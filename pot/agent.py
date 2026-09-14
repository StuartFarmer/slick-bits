"""Search action–observation histories with UCT and sequential thought rollouts."""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Action = Literal["continue", "rollback", "think"]
Judgment = Literal["sure", "likely", "impossible"]
Evaluator = Callable[[tuple[str, ...]], Awaitable[float]]


class Thought(BaseModel, extra="forbid", frozen=True):
    content: Text
    terminal: bool = Field(default=False, strict=True)


class Observation(BaseModel, extra="forbid"):
    judgment: Judgment


class Evaluation(BaseModel, extra="forbid"):
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)


@dataclass(frozen=True)
class State:
    prefix: tuple[str, ...] = ()
    current: Thought | None = None

    @property
    def trajectory(self) -> tuple[str, ...]:
        return self.prefix + ((self.current.content,) if self.current is not None else ())


@dataclass(frozen=True)
class Candidate:
    trajectory: tuple[str, ...]
    terminal: bool
    probability: float
    reward: float

    @property
    def answer(self) -> str:
        return self.trajectory[-1]


@dataclass(frozen=True)
class Result:
    best: Candidate | None
    state: State
    solved: bool
    reason: str
    calls: int
    simulations: int
    actions: int


@dataclass
class _ActionNode:
    visits: int = 0
    value: float = 0.0
    children: dict[tuple[State, Judgment], "_HistoryNode"] = field(default_factory=dict)


@dataclass
class _HistoryNode:
    judgment: Judgment = "likely"
    visits: int = 0
    actions: dict[Action, _ActionNode] = field(default_factory=dict)


class _BudgetExhausted(Exception):
    pass


class PlanOfThoughts:
    """Own one PoT search; task text defines a step and a complete artifact.

    The optional evaluator receives the whole trajectory and returns success
    probability in [0, 1]. Otherwise a Slick judge estimates that probability.
    Providers own decoding settings and transport retries. Calls are independent,
    sequential, and recorded before parsing. No Session history or code execution
    is used. Each run resets the tree, incumbent, and records.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Evaluator | None = None,
        *,
        judge_provider: Provider | None = None,
        rollout_provider: Provider | None = None,
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.judge_provider = judge_provider if judge_provider is not None else provider
        self.rollout_provider = rollout_provider if rollout_provider is not None else provider
        self.calls: list[dict] = []
        self.history: list[dict] = []
        self.best: Candidate | None = None
        self.root = _HistoryNode()

    @prompt(template="think.j2", output_type=Thought)
    async def think(self, prefix: tuple[str, ...], previous: str, *, generated: Thought) -> Thought:
        """Sample a next thought, optionally replacing a pending proposal."""
        return generated

    @prompt(template="rollout_step.j2", output_type=Thought)
    async def rollout_step(self, trajectory: tuple[str, ...], *, generated: Thought) -> Thought:
        """Extend one trajectory with the generator's best next step."""
        return generated

    @prompt(template="observe.j2", output_type=Observation)
    async def observe(self, trajectory: tuple[str, ...], *, generated: Observation) -> Judgment:
        """Sample a usefulness label, distinct from final correctness."""
        return generated.judgment

    @prompt(template="judge.j2", output_type=Evaluation)
    async def judge(self, trajectory: tuple[str, ...], *, generated: Evaluation) -> float:
        """Estimate whether the complete trajectory satisfies the task."""
        return generated.probability

    async def run(
        self,
        *,
        depth: int = 5,
        simulations: int = 16,
        max_actions: int = 32,
        max_calls: int = 256,
        time_limit: float | None = None,
        exploration: float = math.sqrt(2),
        success_threshold: float = 0.95,
        reward_min: float = 0.0,
        reward_max: float = 1.0,
    ) -> Result:
        """Plan, execute one action, and retain the matching observation subtree.

        Depth bounds trajectory length; simulations is per executed action.
        Simulation action depth is capped at 2*depth to bound rollback loops.
        max_calls counts provider calls and evaluator invocations, even failures.
        The deadline cancels in-flight async work cooperatively. Exhausted budgets
        return the best assessed candidate, including candidates from simulations;
        generation, evaluation, and transport errors propagate without retries.
        """
        self.calls, self.history, self.best = [], [], None
        self.root = _HistoryNode()
        self.depth, self.max_calls = depth, max_calls
        self.exploration, self.success_threshold = exploration, success_threshold
        self.reward_min, self.reward_max = reward_min, reward_max
        self.deadline = math.inf if time_limit is None else time.monotonic() + time_limit
        self.simulations = 0
        self.state = State()
        reason = "actions"
        try:
            self.state, judgment = await self._transition(self.state, "think")
            self.root.judgment = judgment
            if self.state.current.terminal or len(self.state.trajectory) >= self.depth:
                await self._assess(self.state)
            for _ in range(max_actions):
                if self._solved():
                    break
                await self._plan(simulations)
                if self._solved():
                    break
                await self._advance()
        except _BudgetExhausted as exc:
            reason = str(exc)
        return Result(
            self.best,
            self.state,
            self._solved(),
            "solved" if self._solved() else reason,
            len(self.calls),
            self.simulations,
            len(self.history),
        )

    def _solved(self) -> bool:
        return (
            self.best is not None
            and self.best.terminal
            and self.best.probability >= self.success_threshold
        )

    async def _invoke(self, operation: Callable[[], Awaitable], record: dict):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise _BudgetExhausted("time")
        if len(self.calls) >= self.max_calls:
            raise _BudgetExhausted("calls")
        self.calls.append(record)
        pending = None
        try:
            if math.isinf(remaining):
                return await operation()
            pending = asyncio.ensure_future(operation())
            done, _ = await asyncio.wait((pending,), timeout=remaining)
            if not done:
                raise _BudgetExhausted("time")
            return pending.result()
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def _generate(self, operation, provider: Provider, *args):
        record = {"operation": operation.__name__}

        async def acall(context):
            record["prompt"] = context
            response, requests = await provider.acall(context)
            record["response"] = response
            return response, requests

        return await self._invoke(
            lambda: operation(*args, provider=SimpleNamespace(acall=acall)), record
        )

    async def _transition(self, state: State, action: Action) -> tuple[State, Judgment]:
        if action == "rollback":
            next_state = State(state.prefix[:-1], Thought(content=state.prefix[-1]))
        else:
            prefix = state.trajectory if action == "continue" else state.prefix
            previous = state.current.content if action == "think" and state.current else ""
            thought = await self._generate(self.think, self.provider, prefix, previous)
            next_state = State(prefix, thought)
        judgment = await self._generate(self.observe, self.judge_provider, next_state.trajectory)
        return next_state, judgment

    def _actions(self, state: State, judgment: Judgment) -> list[Action]:
        # Observation guides first exploration and ties; it never prunes an action.
        order: list[Action] = (
            ["rollback", "think", "continue"]
            if judgment == "impossible"
            else ["continue", "think", "rollback"]
        )
        return [
            action
            for action in order
            if (action != "rollback" or state.prefix)
            and (
                action != "continue"
                or (
                    state.current is not None
                    and not state.current.terminal
                    and len(state.trajectory) < self.depth
                )
            )
        ]

    def _select(self, node: _HistoryNode, state: State, *, explore: bool) -> Action:
        actions = self._actions(state, node.judgment)
        for action in actions:
            node.actions.setdefault(action, _ActionNode())
        if explore:
            for action in actions:
                if node.actions[action].visits == 0:
                    return action
            return max(
                actions,
                key=lambda action: (
                    node.actions[action].value
                    + self.exploration
                    * math.sqrt(math.log(node.visits) / node.actions[action].visits)
                ),
            )
        visited = [action for action in actions if node.actions[action].visits]
        return (
            max(visited, key=lambda action: node.actions[action].value) if visited else actions[0]
        )

    async def _plan(self, simulations: int):
        for _ in range(simulations):
            await self._simulate(self.state, self.root, 2 * self.depth)
            self.simulations += 1
            if self._solved():
                break

    async def _simulate(self, state: State, node: _HistoryNode, remaining: int) -> float:
        if remaining <= 0:
            return await self._rollout(state)
        action = self._select(node, state, explore=True)
        edge = node.actions[action]
        next_state, judgment = await self._transition(state, action)
        key = (next_state, judgment)
        child = edge.children.get(key)
        if child is None:
            edge.children[key] = _HistoryNode(judgment)
            value = await self._rollout(next_state)
        elif next_state.current.terminal or len(next_state.trajectory) >= self.depth:
            value = await self._assess(next_state)
        else:
            value = await self._simulate(next_state, child, remaining - 1)
        node.visits += 1
        edge.visits += 1
        edge.value += (value - edge.value) / edge.visits
        return value

    async def _rollout(self, state: State) -> float:
        while not state.current.terminal and len(state.trajectory) < self.depth:
            thought = await self._generate(
                self.rollout_step, self.rollout_provider, state.trajectory
            )
            state = State(state.trajectory, thought)
        return await self._assess(state)

    async def _assess(self, state: State) -> float:
        if self.evaluate is None:
            probability = await self._generate(self.judge, self.judge_provider, state.trajectory)
        else:
            record = {"operation": "evaluate", "trajectory": state.trajectory}

            async def evaluate():
                probability = float(await self.evaluate(state.trajectory))
                record["probability"] = probability
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("evaluation probability must be finite and in [0, 1]")
                return probability

            probability = await self._invoke(evaluate, record)
        # Standard expected reward: the paper's division by Rmax+Rmin is undefined
        # for symmetric rewards. Gamma is one; only final reward is backed up.
        reward = self.reward_max * probability + self.reward_min * (1 - probability)
        candidate = Candidate(state.trajectory, state.current.terminal, probability, reward)
        if self.best is None or (candidate.terminal, candidate.probability) > (
            self.best.terminal,
            self.best.probability,
        ):
            self.best = candidate
        return reward

    async def _advance(self):
        action = self._select(self.root, self.state, explore=False)
        state, judgment = await self._transition(self.state, action)
        edge = self.root.actions[action]
        self.root = edge.children.setdefault((state, judgment), _HistoryNode(judgment))
        self.state = state
        self.history.append({"action": action, "state": state, "judgment": judgment})
        if state.current.terminal or len(state.trajectory) >= self.depth:
            await self._assess(state)
