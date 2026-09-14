"""Plan textual state/action traces with RAP's lazy MCTS and world model.

Adapted from maitrix-org/llm-reasoners (Apache-2.0); see README.md and LICENSE.
Modified for async Slick prompts, task-independent states, and explicit failures.
"""

import math
from collections import defaultdict
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass, field
from statistics import fmean
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Action(BaseModel, extra="forbid", frozen=True):
    content: Text


class State(BaseModel, extra="forbid", frozen=True):
    content: Text
    terminal: bool = Field(default=False, strict=True)
    answer: Text | None = None


class Score(BaseModel, extra="forbid"):
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


@dataclass(frozen=True)
class Transition:
    source: State
    action: str
    target: State
    confidence: float
    helpfulness: float


@dataclass(eq=False)
class Node:
    state: State | None = None
    action: str | None = None
    parent: "Node | None" = field(default=None, repr=False)
    depth: int = 0
    children: "list[Node] | None" = field(default=None, repr=False)
    helpfulness: float = 0.0
    fast_reward: float = 0.0
    reward: float = 0.0
    transition: Transition | None = None
    returns: list[float] = field(default_factory=list)

    @property
    def visits(self) -> int:
        return len(self.returns)


@dataclass(frozen=True)
class Trace:
    states: tuple[State, ...]
    actions: tuple[str, ...]
    rewards: tuple[float, ...]
    score: float
    reason: Literal["terminal", "depth_limit", "dead_end"]


@dataclass(frozen=True)
class Result:
    best: Trace | None
    partial: Trace | None
    answer: str | None
    answer_weights: dict[str, float]
    traces: tuple[Trace, ...]
    iterations: int


class RAP:
    """Own one simulated search at a time; no actions are executed externally.

    evaluate(transition) optionally replaces the geometric step reward with a
    finite, higher-is-better score. The callback owns any execution isolation.
    State contents must retain all context needed for subsequent steps. Calls
    are independent and sequential. Errors propagate; callers own retries,
    model settings, transport budgets, and durable logging.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Transition], Awaitable[float]] | None = None,
        *,
        actions: str = "One incremental reasoning step, subquestion, or proposed planning action",
        world: str = "Track established facts, intermediate results, and unresolved requirements",
        criteria: str = "Correctness and useful progress toward satisfying every task requirement",
        state_key: Callable[[State], Hashable] | None = None,
        answer_key: Callable[[str], str] = str.strip,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.actions, self.world, self.criteria = actions, world, criteria
        self.state_key = state_key or (lambda state: (state.content, state.terminal, state.answer))
        self.answer_key = answer_key
        self.calls: list[dict] = []
        self.evaluations: list[dict] = []
        self.exploration_weight = 1.0

    @prompt(template="propose.j2", output_type=Action)
    async def propose(self, state: State, *, generated: Action) -> str:
        """Sample one possible action conditioned on the current state."""
        return generated.content

    @prompt(template="assess.j2", output_type=Score)
    async def assess(self, state: State, action: str, *, generated: Score) -> float:
        """Estimate action helpfulness without a world-model rollout."""
        return generated.score

    @prompt(template="predict.j2", output_type=State)
    async def predict(self, state: State, action: str, *, generated: State) -> State:
        """Simulate the next state; a terminal flag is a model claim."""
        if generated.answer is not None and not generated.terminal:
            raise ValueError("a generated final answer requires a terminal state")
        return generated

    async def run(
        self,
        initial_state: State | None = None,
        *,
        iterations: int = 10,
        candidates: int = 4,
        depth: int = 5,
        confidence_samples: int = 4,
        exploration_weight: float = 1.0,
        reward_alpha: float = 0.5,
        prior_confidence: float = 0.8,
        cum_reward: Callable[[list[float]], float] = fmean,
        calc_q: Callable[[list[float]], float] = max,
        aggregate: bool = False,
    ) -> Result:
        """Select, expand, greedily simulate, and back up suffix rewards.

        Default Q is the maximum observed mean suffix reward (paper Eq. 2).
        Depth counts actions. A cutoff is a partial trace, not a terminal answer.
        Revisited states and terminal paths reuse cached transitions. Each run
        resets the tree and in-memory records; there are no hidden retries.
        """
        self.candidates, self.depth = candidates, depth
        self.confidence_samples = confidence_samples
        self.exploration_weight = exploration_weight
        self.reward_alpha, self.prior_confidence = reward_alpha, prior_confidence
        self.cum_reward, self.calc_q = cum_reward, calc_q
        self.calls, self.evaluations = [], []
        self.root = Node(
            state=initial_state if initial_state is not None else State(content=self.task)
        )
        traces = []
        if self.root.state.terminal:
            return self._result([self._trace([self.root])], aggregate, 0)
        for _ in range(iterations):
            path = self._select()
            await self._expand(path[-1])
            await self._simulate(path)
            self._backpropagate(path)
            traces.append(self._trace(path))
        return self._result(traces, aggregate, len(traces))

    def _select(self) -> list[Node]:
        node = self.root
        path = [node]
        while node.children and not node.state.terminal and node.depth < self.depth:
            node = max(node.children, key=self._uct)
            path.append(node)
        return path

    def _uct(self, node: Node) -> float:
        q = self.calc_q(node.returns) if node.visits else node.fast_reward
        # Official RAP gives unvisited actions a finite local prior, not infinity.
        return q + self.exploration_weight * math.sqrt(
            math.log(max(1, node.parent.visits)) / max(1, node.visits)
        )

    async def _expand(self, node: Node) -> None:
        if node.state is None:
            await self._materialize(node)
        if node.state.terminal or node.depth >= self.depth or node.children is not None:
            return
        samples = [await self._invoke(self.propose, node.state) for _ in range(self.candidates)]
        node.children = []
        for action in dict.fromkeys(samples):
            helpfulness = await self._invoke(self.assess, node.state, action)
            prior = helpfulness**self.reward_alpha * self.prior_confidence ** (
                1 - self.reward_alpha
            )
            node.children.append(
                Node(
                    action=action,
                    parent=node,
                    depth=node.depth + 1,
                    helpfulness=helpfulness,
                    fast_reward=prior,
                )
            )

    async def _materialize(self, node: Node) -> None:
        samples = [
            await self._invoke(self.predict, node.parent.state, node.action)
            for _ in range(self.confidence_samples)
        ]
        groups = defaultdict(list)
        for state in samples:
            groups[self.state_key(state)].append(state)
        majority = max(groups.values(), key=len)
        transition = Transition(
            node.parent.state,
            node.action,
            majority[0],
            len(majority) / len(samples),
            node.helpfulness,
        )
        record = {"transition": transition}
        self.evaluations.append(record)
        try:
            reward = (
                await self.evaluate(transition)
                if self.evaluate is not None
                else transition.helpfulness**self.reward_alpha
                * transition.confidence ** (1 - self.reward_alpha)
            )
            record["reward"] = reward
            if not math.isfinite(reward):
                raise ValueError("transition reward must be finite")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        node.state, node.transition, node.reward = transition.target, transition, reward

    async def _simulate(self, path: list[Node]) -> None:
        node = path[-1]
        while node.children and not node.state.terminal and node.depth < self.depth:
            node = max(node.children, key=lambda child: child.fast_reward)
            path.append(node)
            await self._expand(node)

    def _backpropagate(self, path: list[Node]) -> None:
        # ponytail: arbitrary suffix reducers cost O(depth²); use running sums for deep trees.
        rewards = []
        for node in reversed(path[1:]):
            rewards.insert(0, node.reward)
            node.returns.append(self.cum_reward(rewards))
        # The root has no incoming action or reward; don't dilute mean returns with zero.
        path[0].returns.append(self.cum_reward(rewards) if rewards else 0.0)

    def _trace(self, path: list[Node]) -> Trace:
        rewards = [node.reward for node in path[1:]]
        leaf = path[-1]
        if leaf.state.terminal:
            reason = "terminal" if leaf.state.answer is not None else "dead_end"
        else:
            reason = "depth_limit" if leaf.depth >= self.depth else "dead_end"
        return Trace(
            tuple(node.state for node in path),
            tuple(node.action for node in path[1:]),
            tuple(rewards),
            self.cum_reward(rewards) if rewards else 0.0,
            reason,
        )

    def _aggregate(self) -> dict[str, float]:
        weights = defaultdict(float)

        def visit(node: Node) -> set[str]:
            if node.state is None:
                return set()
            if node.state.terminal:
                answers = (
                    {self.answer_key(node.state.answer)} if node.state.answer is not None else set()
                )
            else:
                answers = set()
                for child in node.children or []:
                    answers.update(visit(child))
            # Each incoming edge contributes once per distinct descendant answer.
            for answer in sorted(answers):
                weights[answer] += node.reward
            return answers

        visit(self.root)
        return dict(weights)

    def _result(self, traces: list[Trace], aggregate: bool, iterations: int) -> Result:
        best = max(
            (t for t in traces if t.reason == "terminal"), key=lambda t: t.score, default=None
        )
        partial = max(
            (t for t in traces if t.reason != "terminal"), key=lambda t: t.score, default=None
        )
        weights = self._aggregate() if aggregate else {}
        answer = (
            max(weights, key=weights.get) if weights else (best.states[-1].answer if best else None)
        )
        return Result(best, partial, answer, weights, tuple(traces), iterations)

    async def _invoke(self, operation, *args):
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Capture raw responses before Slick's parsing and postprocessing."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("RAP prompt operations require text, not tool requests")
        return response, requests
