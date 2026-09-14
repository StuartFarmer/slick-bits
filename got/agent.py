"""Execute Graph of Thoughts operation plans with explicit dependency and score state."""

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from graphlib import TopologicalSorter
from statistics import fmean
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(min_length=1)]


class Decomposition(BaseModel, extra="forbid"):
    thoughts: list[Text]


class Score(BaseModel, extra="forbid"):
    score: float = Field(allow_inf_nan=False)


class Validation(BaseModel, extra="forbid"):
    valid: bool = Field(strict=True)
    feedback: str


@dataclass(frozen=True)
class Thought:
    id: int
    content: str
    parents: tuple[int, ...] = ()
    kind: str = "solution"
    score: float | None = None
    valid: bool | None = None
    feedback: str = ""


Evaluator = Callable[[Thought, Mapping[int, Thought]], Awaitable[float]]
Validator = Callable[[Thought, Mapping[int, Thought]], Awaitable[Validation]]
Selector = Callable[[tuple[Thought, ...]], Sequence[Thought]]


@dataclass(frozen=True)
class Operation:
    """One static GoO step; predecessors name other steps in the plan.

    count means independent outputs for generate/aggregate/improve, parts for
    decompose, LLM samples for score, retained items for keep_best, and maximum
    repairs for validate_and_improve. selector routes existing thoughts locally.
    """

    kind: Literal[
        "generate",
        "decompose",
        "aggregate",
        "improve",
        "score",
        "validate_and_improve",
        "keep_best",
        "keep_valid",
        "select",
    ]
    predecessors: tuple[str, ...] = ()
    count: int = 1
    instruction: str = ""
    thought_kind: str = "solution"
    selector: Selector | None = None


@dataclass(frozen=True)
class Result:
    final: tuple[Thought, ...]
    thoughts: Mapping[int, Thought]
    outputs: Mapping[str, tuple[Thought, ...]]
    calls: int

    def volume(self, thought_id: int) -> int:
        """Count distinct preceding generated thoughts, excluding input and self."""
        seen = {thought_id, 0}
        pending = list(self.thoughts[thought_id].parents)
        ancestors = set()
        while pending:
            parent = pending.pop()
            if parent not in seen:
                seen.add(parent)
                ancestors.add(parent)
                pending.extend(self.thoughts[parent].parents)
        return len(ancestors)


class CallBudgetExceeded(RuntimeError):
    """No further provider calls are allowed; partial state remains on the agent."""


def default_plan() -> dict[str, Operation]:
    """Generate, rank, aggregate, then refine with incumbent-preserving selection."""
    return {
        "generate": Operation("generate", count=5),
        "score": Operation("score", ("generate",)),
        "selected": Operation("keep_best", ("score",), count=3),
        "aggregate": Operation("aggregate", ("selected",), count=3),
        "aggregate_score": Operation("score", ("aggregate",)),
        "incumbent": Operation("keep_best", ("selected", "aggregate_score")),
        "improve": Operation("improve", ("incumbent",), count=2),
        "improve_score": Operation("score", ("improve",)),
        "best": Operation("keep_best", ("incumbent", "improve_score")),
    }


def nonblank(text: str) -> str:
    if not text.strip():
        raise ValueError("generated thought is blank")
    return text


class GraphOfThoughts:
    """Own one sequential run at a time; the task defines the meaning of thoughts.

    Optional async evaluate/validate callbacks receive an immutable thought and a
    graph snapshot. Otherwise Slick scores/validates with the supplied provider.
    Model configuration, transport retries, execution isolation and persistence
    belong to the caller. No shared conversation or automatic generation retry.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Evaluator | None = None,
        *,
        validate: Validator | None = None,
        criteria: str = "Correctness, completeness, and satisfaction of the task constraints",
        higher_is_better: bool = True,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.validator, self.criteria = validate, criteria
        self.higher_is_better = higher_is_better
        self.score_direction = "higher" if higher_is_better else "lower"
        self.input = ""
        self.thoughts: dict[int, Thought] = {}
        self.outputs: dict[str, tuple[Thought, ...]] = {}
        self.calls: list[dict] = []
        self.assessments: list[dict] = []
        self.execution: list[dict] = []

    @prompt(template="generate.j2")
    async def generate(self, thought: Thought, instruction: str, *, generated: str) -> str:
        """Generate one continuation; repeated calls create independent alternatives."""
        return nonblank(generated)

    @prompt(template="decompose.j2", output_type=Decomposition)
    async def decompose(
        self, thought: Thought, count: int, instruction: str, *, generated: Decomposition
    ) -> list[str]:
        """Split a thought into exactly the requested number of textual parts."""
        if len(generated.thoughts) != count:
            raise ValueError("decomposition must contain exactly the requested number of thoughts")
        return [nonblank(text) for text in generated.thoughts]

    @prompt(template="aggregate.j2")
    async def aggregate(
        self, thoughts: tuple[Thought, ...], instruction: str, *, generated: str
    ) -> str:
        """Combine all supplied thoughts in one new artifact."""
        return nonblank(generated)

    @prompt(template="improve.j2")
    async def improve(self, thought: Thought, instruction: str, *, generated: str) -> str:
        """Refine a thought using its current validation feedback."""
        return nonblank(generated)

    @prompt(template="score.j2", output_type=Score)
    async def score(self, thought: Thought, instruction: str, *, generated: Score) -> float:
        """Obtain one finite model assessment."""
        return generated.score

    @prompt(template="validate.j2", output_type=Validation)
    async def validate(
        self, thought: Thought, instruction: str, *, generated: Validation
    ) -> Validation:
        """Check validity without equating a model judgment with ground truth."""
        return generated

    async def run(
        self, input: str, *, plan: Mapping[str, Operation] | None = None, max_calls: int = 100
    ) -> Result:
        """Execute each operation once and return the union of its terminal outputs.

        Failures and budget exhaustion raise, preserving partial state and records.
        max_calls includes scoring/validation calls, excluding provider-internal
        transport retries. An explicit empty plan returns no final thoughts.
        """
        self.input, self.max_calls = input, max_calls
        self.thoughts = {0: Thought(0, input, kind="input")}
        self.outputs, self.calls, self.assessments, self.execution = {}, [], [], []
        plan = default_plan() if plan is None else dict(plan)
        order = tuple(
            TopologicalSorter({key: op.predecessors for key, op in plan.items()}).static_order()
        )
        scheduled = [(name, plan[name]) for name in order]
        for name, operation in scheduled:
            await self._execute(name, operation)
        predecessors = {name for op in plan.values() for name in op.predecessors}
        final = self._union(tuple(name for name in plan if name not in predecessors))
        return Result(
            final,
            MappingProxyType(dict(self.thoughts)),
            MappingProxyType(dict(self.outputs)),
            len(self.calls),
        )

    def _union(self, predecessors: tuple[str, ...]) -> tuple[Thought, ...]:
        # A shared thought contributes once at a join; first predecessor wins annotations.
        selected = {}
        for name in predecessors:
            for thought in self.outputs[name]:
                selected.setdefault(thought.id, thought)
        return tuple(selected.values())

    async def _execute(self, name: str, operation: Operation) -> None:
        record = {"operation": name, "kind": operation.kind}
        self.execution.append(record)
        try:
            inputs = (
                self._union(operation.predecessors)
                if operation.predecessors
                else (Thought(0, self.input, kind="input"),)
            )
            if operation.kind in ("generate", "decompose", "aggregate", "improve"):
                outputs = await self._transform(operation, inputs)
            elif operation.kind == "score":
                outputs = [await self._assess(thought, operation) for thought in inputs]
            elif operation.kind == "validate_and_improve":
                outputs = [await self._repair(thought, operation) for thought in inputs]
            elif operation.kind == "keep_best":
                if any(thought.score is None for thought in inputs):
                    raise ValueError("keep_best requires scored thoughts")
                outputs = sorted(inputs, key=lambda t: t.score, reverse=self.higher_is_better)[
                    : operation.count
                ]
            elif operation.kind == "keep_valid":
                outputs = [thought for thought in inputs if thought.valid is True]
            elif operation.kind == "select":
                outputs = operation.selector(inputs)
            else:
                raise ValueError(f"unknown operation: {operation.kind}")
            self.outputs[name] = tuple(outputs)
            record["outputs"] = tuple(t.id for t in outputs)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _transform(self, operation: Operation, inputs: tuple[Thought, ...]) -> list[Thought]:
        outputs = []
        if not inputs:
            return outputs
        if operation.kind == "decompose":
            for thought in inputs:
                parts = await self._invoke(
                    self.decompose, thought, operation.count, operation.instruction
                )
                outputs.extend(self._create(text, (thought,), operation) for text in parts)
            return outputs
        method = {"generate": self.generate, "aggregate": self.aggregate, "improve": self.improve}[
            operation.kind
        ]
        groups = [inputs] if operation.kind == "aggregate" else [(thought,) for thought in inputs]
        for parents in groups:
            argument = parents if operation.kind == "aggregate" else parents[0]
            for _ in range(operation.count):
                text = await self._invoke(method, argument, operation.instruction)
                outputs.append(self._create(text, parents, operation))
        return outputs

    def _create(self, text: str, parents: tuple[Thought, ...], operation: Operation) -> Thought:
        # Every generation also sees the original input. Refinements are versioned,
        # preserving feedback loops as edges without overwriting earlier artifacts.
        ids = tuple(dict.fromkeys((0, *(thought.id for thought in parents))))
        thought = Thought(len(self.thoughts), text, ids, operation.thought_kind)
        self.thoughts[thought.id] = thought
        return thought

    async def _assess(self, thought: Thought, operation: Operation) -> Thought:
        record = {"kind": "score", "thought": thought.id}
        self.assessments.append(record)
        try:
            if self.evaluate is not None:
                score = float(
                    await self.evaluate(
                        thought, MappingProxyType({**self.thoughts, thought.id: thought})
                    )
                )
            else:
                score = fmean(
                    [
                        await self._invoke(self.score, thought, operation.instruction)
                        for _ in range(operation.count)
                    ]
                )
            record["score"] = score
            if not math.isfinite(score):
                raise ValueError("evaluation score must be finite")
            scored = replace(thought, score=score)
            self.thoughts[thought.id] = scored
            return scored
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _repair(self, thought: Thought, operation: Operation) -> Thought:
        for attempt in range(operation.count + 1):
            record = {"kind": "validate", "thought": thought.id}
            self.assessments.append(record)
            try:
                if self.validator is None:
                    verdict = await self._invoke(self.validate, thought, operation.instruction)
                else:
                    verdict = await self.validator(
                        thought, MappingProxyType({**self.thoughts, thought.id: thought})
                    )
                thought = replace(thought, valid=verdict.valid, feedback=verdict.feedback)
                self.thoughts[thought.id] = thought
                record["valid"], record["feedback"] = verdict.valid, verdict.feedback
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            if thought.valid or attempt == operation.count:
                return thought
            text = await self._invoke(self.improve, thought, operation.instruction)
            thought = self._create(text, (thought,), operation)
        return thought

    async def _invoke(self, method, *args):
        if len(self.calls) >= self.max_calls:
            raise CallBudgetExceeded(f"provider call budget exhausted ({self.max_calls})")
        record = {"operation": method.__name__}
        self.calls.append(record)
        try:
            return await method(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Capture raw output before Slick parsing or generated-output validation."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("GoT prompt operations require text, not tool requests")
        return response, requests
