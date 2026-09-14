"""Search independent thought trees, repair uncertain candidates, and select by consensus."""

import math
from collections import Counter
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Candidate(BaseModel, extra="forbid", frozen=True):
    content: Text
    answer: Text | None = None


class Candidates(BaseModel, extra="forbid"):
    candidates: list[Candidate]


class Evaluation(BaseModel, extra="forbid", frozen=True):
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    valid: bool = Field(default=True, strict=True)
    feedback: str = ""


class Choice(BaseModel, extra="forbid"):
    choice: int = Field(strict=True, ge=1)


@dataclass
class Node:
    candidate: Candidate
    evaluation: Evaluation
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    @property
    def reward(self) -> float:
        # MCTSr's conservative reward, on the official implementation's 0–100 scale.
        return 50 * (min(self.rewards) + sum(self.rewards) / len(self.rewards))


@dataclass(frozen=True)
class Tree:
    index: int
    active: bool
    solution: Candidate | None
    reason: str


@dataclass(frozen=True)
class Result:
    solution: Candidate | None
    trees: tuple[Tree, ...]
    decision: str
    calls: int
    budget_exhausted: bool

    @property
    def answer(self) -> str | None:
        return self.solution.answer if self.solution is not None else None


class _BudgetReached(Exception):
    pass


class ForestOfThought:
    """Own one run at a time; callers supply task semantics and execution resources.

    Evaluations score complete candidate content, including partial solutions.
    Confidence is separate from quality. Without an evaluator both are LM estimates.
    Calls are sequential and independent, with no shared conversational Session.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Candidate], Awaitable[Evaluation]] | None = None,
        *,
        retrieve: Callable[[str], Awaitable[str]] | None = None,
        correct: Callable[[Candidate, str], Awaitable[Candidate | None]] | None = None,
        verify: Callable[[Candidate], Awaitable[bool]] | None = None,
        answer_key: Callable[[str], Hashable | None] = str.strip,
        criteria: str = "Correctness, feasibility, and consistency with all task constraints",
        answer_format: str = "A complete answer satisfying the task",
        perspectives: tuple[str, ...] = (
            "Build a direct constructive solution",
            "Work backward from the desired outcome",
            "Decompose the task into independently checkable parts",
            "Test assumptions and seek an alternative approach",
        ),
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.retrieve, self.correct, self.verify = retrieve, correct, verify
        self.answer_key, self.criteria, self.answer_format = answer_key, criteria, answer_format
        self.perspectives = perspectives
        self.knowledge, self.perspective = "", ""
        self.calls: list[dict] = []
        self.assessments: list[dict] = []
        self.corrections: list[dict] = []
        self.callbacks: list[dict] = []
        self.history: list[dict] = []
        self.nodes: dict[int, list[Node]] = {}

    @prompt(template="propose.j2", output_type=Candidates)
    async def propose(self, previous: dict, count: int, *, generated: Candidates) -> list[Candidate]:
        """Produce sibling continuations; an empty list represents a dead end."""
        if len(generated.candidates) > count:
            raise ValueError("proposal exceeds candidate limit")
        return generated.candidates

    @prompt(template="initialize.j2", output_type=Candidate)
    async def initialize(self, *, generated: Candidate) -> Candidate:
        """Generate the initial complete solution for MCTSr."""
        if generated.answer is None:
            raise ValueError("initial solution requires an answer")
        return generated

    @prompt(template="assess.j2", output_type=Evaluation)
    async def assess(self, candidate: Candidate, *, generated: Evaluation) -> Evaluation:
        """Estimate quality, confidence, validity, and actionable feedback."""
        return generated

    @prompt(template="correct.j2", output_type=Candidate)
    async def correct_candidate(
        self, candidate: Candidate, feedback: str, *, generated: Candidate
    ) -> Candidate:
        """Repair a low-confidence candidate using feedback and prior knowledge."""
        if candidate.answer is not None and generated.answer is None:
            raise ValueError("correction dropped the final answer")
        return generated

    @prompt(template="refine.j2", output_type=Candidate)
    async def refine(
        self, candidate: Candidate, feedback: str, *, generated: Candidate
    ) -> Candidate:
        """Expand an MCTSr node with a new complete solution."""
        if generated.answer is None:
            raise ValueError("refinement requires an answer")
        return generated

    @prompt(template="finish.j2", output_type=Candidate)
    async def finish(self, previous: dict, *, generated: Candidate) -> Candidate:
        """Complete the best beam leaf, or solve directly when depth is zero."""
        if generated.answer is None:
            raise ValueError("final solution requires an answer")
        return generated

    @prompt(template="choose.j2", output_type=Choice)
    async def choose(self, candidates: list[Candidate], *, generated: Choice) -> Candidate:
        """Select an existing active-tree solution, preserving its supporting content."""
        if generated.choice > len(candidates):
            raise ValueError("expert selected an unknown candidate")
        return candidates[generated.choice - 1]

    async def run(
        self,
        *,
        trees: int = 4,
        search: Literal["tot", "mctsr"] = "tot",
        depth: int = 3,
        breadth: int = 2,
        candidates: int = 3,
        rollouts: int = 4,
        exploration: float = 1.4,
        correction_threshold: float = 0.5,
        consensus: Literal["majority", "plurality"] = "majority",
        max_calls: int = 1000,
    ) -> Result:
        """Build trees, stop on verification/locked majority, then use CGED.

        `max_calls` counts attempted prompt operations, including assessment and
        expert selection. Exhaustion returns explicit partial records. Other errors
        propagate without retries; the caller owns transport and evaluation budgets.
        """
        self.depth, self.breadth, self.candidates = depth, breadth, candidates
        self.rollouts, self.exploration = rollouts, exploration
        self.threshold, self.max_calls = correction_threshold, max_calls
        self.calls, self.assessments, self.corrections, self.callbacks = [], [], [], []
        self.history, self.nodes, self.verified = [], {}, None
        self.knowledge = await self._external("retrieve", self.retrieve, self.task) if self.retrieve else ""
        forest, solutions, exhausted = [], [], False
        build_tree = {"tot": self._beam_tree, "mctsr": self._mctsr_tree}[search]
        for self.tree_index in range(trees):
            self.perspective = self.perspectives[self.tree_index % len(self.perspectives)]
            try:
                solution = await build_tree()
            except _BudgetReached:
                forest.append(Tree(self.tree_index, False, None, "budget_exhausted"))
                exhausted = True
                break
            forest.append(Tree(self.tree_index, solution is not None, solution, "complete" if solution else "inactive"))
            if solution is not None:
                solutions.append(solution)
            if self.verified is not None:
                return Result(self.verified, tuple(forest), "verified", len(self.calls), False)
            winner = self._consensus(solutions, trees, "majority")
            if winner is not None:
                return Result(winner, tuple(forest), "consensus", len(self.calls), False)
        return await self._decide(solutions, forest, consensus, exhausted)

    async def _beam_tree(self) -> Candidate | None:
        frontier: list[tuple[Candidate | None, float]] = [(None, 0)]
        for _ in range(self.depth):
            children, seen = [], set()
            for parent, score in frontier:
                if parent is not None and parent.answer is not None:
                    children.append((parent, score))
                    continue
                previous = parent.model_dump() if parent is not None else {}
                for candidate in await self._invoke(self.propose, previous, self.candidates):
                    prepared = await self._prepare(candidate)
                    if self.verified is not None:
                        return self.verified
                    if prepared is None:
                        continue
                    candidate, evaluation = prepared
                    key = (candidate.content, candidate.answer)
                    if key not in seen:
                        seen.add(key)
                        children.append((candidate, evaluation.score))
            frontier = sorted(children, key=lambda item: item[1], reverse=True)[: self.breadth]
            self.history.append({"tree": self.tree_index, "frontier": tuple(c for c, _ in frontier)})
            if not frontier:
                return None
            if all(candidate.answer is not None for candidate, _ in frontier):
                break
        best = frontier[0][0]
        if best is not None and best.answer is not None:
            return best
        previous = best.model_dump() if best is not None else {}
        prepared = await self._prepare(await self._invoke(self.finish, previous))
        return prepared[0] if prepared is not None else None

    async def _mctsr_tree(self) -> Candidate | None:
        prepared = await self._prepare(await self._invoke(self.initialize))
        if prepared is None:
            return None
        candidate, evaluation = prepared
        nodes = self.nodes[self.tree_index] = [Node(candidate, evaluation, rewards=[evaluation.score])]
        for _ in range(self.rollouts):
            if self.verified is not None:
                return self.verified
            ucb = self._ucb(nodes)
            eligible = [
                i for i, node in enumerate(nodes)
                if len(node.children) < self.candidates
                or max(nodes[j].reward for j in node.children) < node.reward
            ]
            if not eligible:
                break
            index = max(eligible, key=lambda i: ucb[i])
            parent = nodes[index]
            parent.evaluation = await self._evaluate(parent.candidate)
            parent.rewards.append(parent.evaluation.score)
            child = await self._invoke(self.refine, parent.candidate, parent.evaluation.feedback)
            prepared = await self._prepare(child)
            if prepared is None:
                return None
            candidate, evaluation = prepared
            parent.children.append(len(nodes))
            nodes.append(Node(candidate, evaluation, index, rewards=[evaluation.score]))
        if self.verified is not None:
            return self.verified
        ucb = self._ucb(nodes)
        best = max(
            range(len(nodes)),
            key=lambda i: 50 * min(nodes[i].rewards) + 0.3 * len(nodes[i].rewards) + 0.2 * ucb[i],
        )
        return nodes[best].candidate

    def _ucb(self, nodes: list[Node]) -> list[float]:
        """Use conservative rewards and the official parent/best-child backup."""
        values = []
        for node in nodes:
            reward = node.reward
            if node.children:
                reward = (reward + max(nodes[i].reward for i in node.children)) / 2
            parent_visits = len(nodes[node.parent].rewards) if node.parent is not None else 0
            values.append(reward + self.exploration * math.sqrt(math.log(parent_visits + 1) / (len(node.rewards) + 1e-5)))
        return values

    async def _prepare(self, candidate: Candidate) -> tuple[Candidate, Evaluation] | None:
        evaluation = await self._evaluate(candidate)
        if evaluation.confidence < self.threshold:
            record = {"tree": self.tree_index, "original": candidate, "accepted": False}
            self.corrections.append(record)
            if self.correct is None:
                revised = await self._invoke(self.correct_candidate, candidate, evaluation.feedback)
            else:
                revised = await self._external("correct", self.correct, candidate, evaluation.feedback)
            record["revised"] = revised
            if revised is None:
                return None
            revised_evaluation = await self._evaluate(revised)
            complete = candidate.answer is None or revised.answer is not None
            if complete and revised_evaluation.valid and (
                not evaluation.valid or revised_evaluation.confidence > evaluation.confidence
            ):
                candidate, evaluation = revised, revised_evaluation
                record["accepted"] = True
        if not evaluation.valid:
            return None
        if candidate.answer is not None:
            if self.answer_key(candidate.answer) is None:
                return None
            if self.verify is not None and await self._external("verify", self.verify, candidate):
                self.verified = candidate
        return candidate, evaluation

    async def _evaluate(self, candidate: Candidate) -> Evaluation:
        record = {"tree": self.tree_index, "candidate": candidate}
        self.assessments.append(record)
        try:
            evaluation = await self.evaluate(candidate) if self.evaluate else await self._invoke(self.assess, candidate)
            record["evaluation"] = evaluation
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (evaluation.score, evaluation.confidence)):
                raise ValueError("measured score and confidence must be finite and in [0, 1]")
            return evaluation
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def _consensus(self, solutions: list[Candidate], total: int, policy: str) -> Candidate | None:
        if not solutions:
            return None
        keys = [self.answer_key(solution.answer) for solution in solutions]
        counts = Counter(keys)
        key, count = counts.most_common(1)[0]
        agreement = count > total / 2 if policy == "majority" else sum(n == count for n in counts.values()) == 1
        return solutions[keys.index(key)] if agreement else None

    async def _decide(self, solutions, forest, consensus, exhausted) -> Result:
        solution, decision = None, "no_solution"
        if solutions:
            solution = self._consensus(solutions, len(solutions), consensus)
            decision = "consensus"
            if solution is None:
                try:
                    solution = await self._invoke(self.choose, solutions)
                    decision = "expert"
                except _BudgetReached:
                    exhausted, decision = True, "budget_exhausted"
        elif exhausted:
            decision = "budget_exhausted"
        return Result(solution, tuple(forest), decision, len(self.calls), exhausted)

    async def _external(self, name, callback, *args):
        record = {"operation": name}
        self.callbacks.append(record)
        try:
            result = await callback(*args)
            record["result"] = result
            return result
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _invoke(self, operation, *args):
        if len(self.calls) >= self.max_calls:
            raise _BudgetReached
        record = {"tree": self.tree_index, "operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Record raw responses before Slick parses or checks generated output."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(context, tools=tools, tool_results=tool_results)
        record["response"] = response
        if requests:
            raise ValueError("FoT requires text responses, not tool requests")
        return response, requests
