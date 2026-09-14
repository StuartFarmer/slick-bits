"""Align generated text with an editor profile through judge-guided meta prompts."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from math import prod
from statistics import covariance, fmean
from types import SimpleNamespace
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False, strict=True)]


@dataclass(frozen=True)
class Dimension:
    name: str
    description: str


class Judgement(BaseModel, extra="forbid"):
    scores: dict[str, Score]
    feedback: Text


@dataclass
class Profile:
    """Caller-owned targets, replaced by means of judged human edits on observe().

    Keep one profile per editor and a consistent dimension set and rubric.
    Initial targets are a cold start, not an extra human observation.
    """

    targets: dict[str, float]
    samples: list[dict[str, float]] = field(default_factory=list)

    def observe(self, scores: dict[str, float]) -> None:
        self.samples.append(dict(scores))
        self.targets = {name: fmean(row[name] for row in self.samples) for name in self.targets}

    def weights(self) -> dict[str, float]:
        """Use the maximum off-diagonal 1-|sample covariance| per normalized axis."""
        if len(self.samples) < 2 or len(self.targets) == 1:
            return dict.fromkeys(self.targets, 1.0)
        columns = {name: [row[name] / 100 for row in self.samples] for name in self.targets}
        return {
            name: max(
                1 - abs(covariance(values, other)) for key, other in columns.items() if key != name
            )
            for name, values in columns.items()
        }


@dataclass(frozen=True)
class Alignment:
    area: float
    expected_area: float
    tma: float
    tmd: float
    loss: float


def alignment(
    scores: dict[str, float], targets: dict[str, float], weights: dict[str, float] | None = None
) -> Alignment:
    """Equations 11, 12 and literal 17 on normalized, scaled diagonal vertices.

    The paper leaves f(G) unspecified. Here its diagonal entries are score/100
    times the profile's covariance weight. A zero expected determinant uses the
    unit hypercube as its normalization volume; see README for this extension.
    """
    scales = dict.fromkeys(targets, 1.0) if weights is None else weights
    current = [scores[name] / 100 * scales[name] for name in targets]
    expected = [targets[name] / 100 * scales[name] for name in targets]
    area, expected_area = prod(current), prod(expected)
    tma = abs(expected_area - area)
    tmd = fmean(abs(a - b) for a, b in zip(current, expected))
    # ponytail: diagonal geometry; revisit f(G) if the authors release its implementation.
    relative_area = tma / (expected_area if expected_area != 0 else 1.0)
    return Alignment(
        area, expected_area, tma, tmd, (relative_area**2 + abs(relative_area)) / 4 + tmd
    )


@dataclass(frozen=True)
class Iteration:
    content: str
    instruction: str
    judgement: Judgement
    alignment: Alignment


@dataclass(frozen=True)
class Result:
    best: Iteration | None
    history: tuple[Iteration, ...]
    stop_reason: Literal["converged", "iterations", "timeout"]


def nonblank(generated: str) -> str:
    if not generated.strip():
        raise ValueError("generated text is blank")
    return generated


class TheoryOfMind:
    """Own independent generation, judgement and prompt-editing calls.

    Configure Slick's template root at startup. Providers own transport retries;
    this class makes no retries and uses no shared Session. Errors propagate with
    raw responses and partial history retained. Timeout returns completed work.
    Use one operation at a time per instance; run() resets call records/history.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        dimensions: Sequence[Dimension],
        profile: Profile,
        *,
        context: str = "",
        evaluate: Callable[[str], Awaitable[Judgement]] | None = None,
        judge_provider: Provider | None = None,
        editor_provider: Provider | None = None,
        max_iterations: int = 21,
        timeout: float = 120,
        threshold: float = 0.05,
    ):
        self.task, self.context = task, context
        self.provider = provider
        self.judge_provider = provider if judge_provider is None else judge_provider
        self.editor_provider = provider if editor_provider is None else editor_provider
        self.dimensions, self.profile, self.evaluate = tuple(dimensions), profile, evaluate
        self.max_iterations, self.timeout, self.threshold = max_iterations, timeout, threshold
        self.history: list[Iteration] = []
        self.calls: list[dict] = []

    @prompt(template="generate.j2")
    async def generate(self, instruction: str, *, generated: str) -> str:
        """Generate the initial task output."""
        return nonblank(generated)

    @prompt(template="regenerate.j2")
    async def regenerate(self, instruction: str, previous: str, *, generated: str) -> str:
        """Apply the editor's revised instructions to the previous output."""
        return nonblank(generated)

    @prompt(template="judge.j2", output_type=Judgement)
    async def judge(self, content: str, *, generated: Judgement) -> Judgement:
        """Measure traits without revealing target scores to the judge."""
        return self._check_judgement(generated)

    @prompt(template="rewrite_prompt.j2")
    async def rewrite_prompt(self, previous: dict, feedback: list[str], *, generated: str) -> str:
        """Produce a new generation instruction from trait deltas and geometric loss."""
        return nonblank(generated)

    def _check_judgement(self, judgement: Judgement) -> Judgement:
        # Revalidate measured output even when an injected evaluator constructed it unchecked.
        checked = Judgement.model_validate(judgement.model_dump())
        if set(checked.scores) != {dimension.name for dimension in self.dimensions}:
            raise ValueError("judgement must contain exactly the configured dimensions")
        return checked

    async def learn(self, edited_content: str) -> Judgement:
        """Judge a human edit against this task/context and update the supplied profile.

        This explicit feedback operation is outside run()'s iteration/time budget.
        Reuse a profile across task instances; generation never updates it implicitly.
        """
        judgement = await self._assess(edited_content)
        self.profile.observe(judgement.scores)
        return judgement

    async def run(self, instruction: str | None = None) -> Result:
        """Evaluate at most max_iterations drafts, including the initial generation.

        Revisions follow the latest draft, as in the paper. Also retain the lowest
        loss draft for caller selection; ties keep the earlier draft. Deadline
        cancellation is cooperative and includes generation, judging and editing.
        """
        self.calls, self.history = [], []
        targets, weights = dict(self.profile.targets), self.profile.weights()
        instruction = self.task if instruction is None else instruction
        deadline = asyncio.get_running_loop().time() + self.timeout
        best, previous = None, None
        reason = "iterations"
        try:
            for _ in range(self.max_iterations):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    reason = "timeout"
                    break
                current = await asyncio.wait_for(
                    self._generate_and_assess(instruction, previous, targets, weights), remaining
                )
                self.history.append(current)
                if best is None or current.alignment.loss < best.alignment.loss:
                    best = current
                if current.alignment.loss < self.threshold:
                    reason = "converged"
                    break
                if len(self.history) == self.max_iterations:
                    break
                instruction = await asyncio.wait_for(
                    self._call(
                        self.rewrite_prompt,
                        self.editor_provider,
                        {**asdict(current), "judgement": current.judgement.model_dump()},
                        self._feedback(current, targets),
                    ),
                    max(0, deadline - asyncio.get_running_loop().time()),
                )
                previous = current.content
        except asyncio.TimeoutError:
            reason = "timeout"
        return Result(best, tuple(self.history), reason)

    async def _generate_and_assess(self, instruction, previous, targets, weights) -> Iteration:
        if previous is None:
            content = await self._call(self.generate, self.provider, instruction)
        else:
            content = await self._call(self.regenerate, self.provider, instruction, previous)
        judgement = await self._assess(content)
        return Iteration(
            content, instruction, judgement, alignment(judgement.scores, targets, weights)
        )

    def _feedback(self, current: Iteration, targets: dict[str, float]) -> list[str]:
        feedback = []
        for name, target in targets.items():
            delta = current.judgement.scores[name] - target
            if delta == 0:
                feedback.append(f"{name} matches expectations ({target:g}); preserve it.")
            else:
                position, direction = ("above", "decrease") if delta > 0 else ("below", "increase")
                feedback.append(
                    f"{name} is {abs(delta):g} percentage points {position} "
                    f"expectations ({target:g}); {direction} {name}."
                )
        return feedback

    async def _assess(self, content: str) -> Judgement:
        if self.evaluate is None:
            return await self._call(self.judge, self.judge_provider, content)
        record = {"operation": "evaluate", "content": content}
        self.calls.append(record)
        try:
            judgement = await self.evaluate(content)
            record["response"] = judgement.model_dump()
            return self._check_judgement(judgement)
        except (Exception, asyncio.CancelledError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _call(self, operation, source: Provider, *args):
        record = {"operation": operation.__name__}
        self.calls.append(record)

        async def acall(context):
            record["prompt"] = context
            text, requests = await source.acall(context)
            record["response"] = text
            if requests:
                raise ValueError("TheoryOfMind requires text responses, not tool requests")
            return text, requests

        try:
            return await operation(*args, provider=SimpleNamespace(acall=acall))
        except (Exception, asyncio.CancelledError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
