"""Search reversible language-action trajectories with LATS and external feedback."""

import math
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from slick import prompt
from slick.providers import Provider


class Action(BaseModel, extra="forbid", frozen=True):
    thought: str
    content: str = Field(min_length=1)


class Value(BaseModel, extra="forbid"):
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


@dataclass(frozen=True)
class Feedback:
    observation: str
    reward: float = 0.0
    terminal: bool = False
    success: bool = False


@dataclass(frozen=True)
class Step:
    action: Action
    feedback: Feedback


@dataclass(eq=False)
class Node:
    trajectory: tuple[Step, ...] = ()
    feedback: Feedback = Feedback("")
    parent: "Node | None" = field(default=None, repr=False)
    children: list["Node"] = field(default_factory=list, repr=False)
    heuristic: float = 0.0
    value: float = 0.0
    visits: int = 0
    expanded: bool = False
    exhausted: bool = False

    def uct(self, exploration_weight: float) -> float:
        """Use the empirical mean once; give unvisited branches priority."""
        if self.visits == 0:
            return math.inf
        return self.value + exploration_weight * math.sqrt(
            math.log(max(1, self.parent.visits)) / self.visits
        )


@dataclass(frozen=True)
class Reflection:
    trajectory: tuple[Step, ...]
    reward: float
    reason: str
    text: str


@dataclass(frozen=True)
class Result:
    best: Node | None
    iterations: int
    expansions: int
    stop_reason: Literal["success", "budget", "exhausted"]


class LATS:
    """Own one search at a time, with independent sequential Slick calls.

    evaluate(history, action) must restore the environment to history before
    applying action, or evaluate that trajectory without mutable shared state.
    Rewards are finite, higher-is-better outcome scores, not incremental rewards.
    Only environment feedback establishes success. Callers own execution isolation,
    provider configuration, transport retries, and durable logging.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[tuple[Step, ...], str], Awaitable[Feedback]],
        *,
        actions: str = "A textual reasoning step, environment command, or complete candidate answer",
        criteria: str = "Progress toward satisfying the task and all its constraints",
        observation: str = "",
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.actions, self.criteria, self.observation = actions, criteria, observation
        self.calls: list[dict] = []
        self.assessments: list[dict] = []
        self.memory: list[Reflection] = []

    @prompt(template="sample.j2", output_type=Action)
    async def sample(self, node: Node, *, generated: Action) -> Action:
        """Sample one thought/action pair; preserve action bytes for the evaluator."""
        if not generated.content.strip():
            raise ValueError("generated action is blank")
        return generated

    @prompt(template="value.j2", output_type=Value)
    async def value(self, node: Node, *, generated: Value) -> float:
        """Score progress after observing the environment's response."""
        return generated.score

    @prompt(template="reflect.j2")
    async def reflect(self, node: Node, reward: float, reason: str, *, generated: str) -> str:
        """Describe an unsuccessful trial and how a subsequent trial could improve."""
        if not generated.strip():
            raise ValueError("generated reflection is blank")
        return generated

    async def run(
        self,
        *,
        iterations: int = 50,
        candidates: int = 5,
        depth: int = 7,
        exploration_weight: float = 1.0,
        lm_weight: float = 0.5,
        simulate: bool = True,
        cutoff_reward: float = 0.0,
    ) -> Result:
        """Select, expand/evaluate, simulate, backpropagate, and reflect.

        Without simulation, each new candidate's measured reward is backed up
        directly (paper section 5.2). Nonterminal candidates remain revisable.
        Depth counts actions from the root. Each run resets memory and records;
        malformed generation and evaluator failures propagate without retries.
        """
        self.candidates, self.depth, self.lm_weight = candidates, depth, lm_weight
        self.exploration_weight = exploration_weight
        self.calls, self.assessments, self.memory = [], [], []
        self.root = Node(feedback=Feedback(self.observation), exhausted=depth == 0)
        self.nodes: list[Node] = []
        self.iterations, self.expansions = 0, 0
        for _ in range(iterations):
            node = self._select()
            if node is None:
                break
            self.iterations += 1
            endpoints = [await self._simulate(node)] if simulate else await self._expand(node)
            for endpoint in endpoints:
                stopped = endpoint.feedback.terminal or endpoint.feedback.success
                reward = cutoff_reward if simulate and not stopped else endpoint.feedback.reward
                self._backpropagate(endpoint, reward)
                self._close(endpoint, simulate or stopped or len(endpoint.trajectory) >= depth)
                if endpoint.feedback.success:
                    return Result(endpoint, self.iterations, self.expansions, "success")
                reason = "terminal failure" if stopped else "candidate feedback"
                if simulate and not stopped:
                    reason = "depth limit" if len(endpoint.trajectory) >= depth else "no actions"
                await self._remember(endpoint, reward, reason)
        # Prefer completed trajectories when available; incomplete fallbacks stay explicit.
        completed = [node for node in self.nodes if node.feedback.terminal]
        pool = completed if simulate and completed else self.nodes
        best = max(pool, key=lambda n: (n.feedback.reward, n.heuristic), default=None)
        reason = "exhausted" if self.root.exhausted else "budget"
        return Result(best, self.iterations, self.expansions, reason)

    def _select(self) -> Node | None:
        node = self.root
        while not node.exhausted:
            if not node.expanded:
                return node
            available = [child for child in node.children if not child.exhausted]
            if not available:
                node.exhausted = True
                node = node.parent or self.root
                continue
            node = max(available, key=lambda child: child.uct(self.exploration_weight))
        return None

    async def _expand(self, node: Node) -> list[Node]:
        if node.feedback.terminal or node.feedback.success or len(node.trajectory) >= self.depth:
            return [node]
        self.expansions += 1
        samples = [await self._invoke(self.sample, node) for _ in range(self.candidates)]
        counts = Counter(action.content for action in samples)
        unique = {action.content: action for action in reversed(samples)}
        node.expanded = True
        for content, count in counts.items():
            action = unique[content]
            feedback = await self._assess(node.trajectory, content)
            child = Node(node.trajectory + (Step(action, feedback),), feedback, parent=node)
            node.children.append(child)
            self.nodes.append(child)
            if feedback.terminal or feedback.success:
                child.heuristic = feedback.reward
            else:
                score = await self._invoke(self.value, child) if self.lm_weight else 0.0
                child.heuristic = self.lm_weight * score + (1 - self.lm_weight) * count / len(
                    samples
                )
            child.value = child.heuristic
            if feedback.success:
                return [child]
        return node.children or [node]

    async def _simulate(self, node: Node) -> Node:
        while not node.feedback.terminal and not node.feedback.success:
            if len(node.trajectory) >= self.depth:
                break
            children = await self._expand(node)
            if children == [node]:
                break
            node = max(children, key=lambda child: child.heuristic)
        return node

    @staticmethod
    def _backpropagate(node: Node, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += (reward - node.value) / node.visits
            node = node.parent

    @staticmethod
    def _close(node: Node, exhausted: bool) -> None:
        node.exhausted = exhausted or (node.expanded and not node.children)
        while node.parent is not None:
            node = node.parent
            node.exhausted = all(child.exhausted for child in node.children)

    async def _remember(self, node: Node, reward: float, reason: str) -> None:
        text = await self._invoke(self.reflect, node, reward, reason)
        # ponytail: full failure context grows with the budget; summarize for large searches.
        self.memory.append(Reflection(node.trajectory, reward, reason, text))

    async def _assess(self, trajectory: tuple[Step, ...], content: str) -> Feedback:
        record = {"trajectory": trajectory, "action": content}
        self.assessments.append(record)
        try:
            feedback = await self.evaluate(trajectory, content)
            record["feedback"] = feedback
            if not math.isfinite(feedback.reward):
                raise ValueError("environment reward must be finite")
            return feedback
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _invoke(self, operation, *args):
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Capture raw responses before Slick parses or validates generated data."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("LATS prompt operations require text, not provider tool requests")
        return response, requests
