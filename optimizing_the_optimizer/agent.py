"""Improve complete optimizer implementations through measured, explicit dialogue."""

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Proposal(BaseModel, extra="forbid"):
    code: str = Field(min_length=1)
    rationale: Text


@dataclass(frozen=True)
class Evaluation:
    """A measured score, or None with feedback explaining candidate rejection."""

    score: float | None
    feedback: str = ""


@dataclass(frozen=True)
class Candidate:
    code: str
    score: float
    feedback: str = ""


class _RejectedProposal(ValueError):
    """A generated response cannot be evaluated as source code."""


def _code(proposal: Proposal) -> str:
    if not proposal.code.strip():
        raise _RejectedProposal("generated code is blank")
    return proposal.code


class OptimizingTheOptimizer:
    """Own the paper's heuristic dialogue and optional performance branches.

    The evaluator owns compilation, interface checks, isolated execution and
    measurement. Each proposal contains the complete replacement implementation.
    No generated code is executed here; arbitrary source languages are supported.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        *,
        maximize: bool = True,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.maximize = maximize
        self.score_direction = "Higher" if maximize else "Lower"
        self.target = ""
        self.attempts: list[dict] = []
        self.evaluations = 0
        self.best: Candidate | None = None

    @prompt(template="improve_heuristic.j2", output_type=Proposal)
    async def improve_heuristic(
        self, source: str, feedback: str, history: list[dict], *, generated: Proposal
    ) -> str:
        """Discover a heuristic using underutilized algorithm state."""
        return _code(generated)

    @prompt(template="revise.j2", output_type=Proposal)
    async def revise(
        self, source: str, feedback: str, history: list[dict], *, generated: Proposal
    ) -> str:
        """Refine the previous heuristic or repair it using measured feedback."""
        return _code(generated)

    @prompt(template="optimize_code.j2", output_type=Proposal)
    async def optimize_code(
        self, source: str, feedback: str, history: list[dict], *, generated: Proposal
    ) -> str:
        """Seek implementation efficiency while preserving heuristic behavior."""
        return _code(generated)

    async def run(
        self, initial: str, *, target: str, rounds: int = 2, performance: bool = False
    ) -> Candidate:
        """Measure the seed, refine heuristics, and return the best measured code.

        Each round consumes one generation, including malformed/rejected outputs.
        performance adds one separate call for each valid heuristic variant.
        Ties retain the incumbent. Provider/evaluator exceptions abort; there are
        no hidden retries. One instance supports one run at a time.
        """
        self.target = target
        self.attempts, self.evaluations, self.best = [], 0, None
        baseline = await self._measure(initial)
        if baseline.score is None:
            raise ValueError(f"initial implementation rejected: {baseline.feedback}")
        self.best = Candidate(initial, baseline.score, baseline.feedback)
        source, feedback = initial, self._feedback(baseline)
        for round_index in range(rounds):
            operation = self.improve_heuristic if round_index == 0 else self.revise
            source, feedback, candidate = await self._attempt(operation, source, feedback)
            if performance and candidate is not None:
                await self._attempt(self.optimize_code, source, feedback)
        return self.best

    async def _measure(self, source: str) -> Evaluation:
        self.evaluations += 1
        evaluation = await self.evaluate(source)
        if evaluation.score is not None and not math.isfinite(evaluation.score):
            return Evaluation(None, f"measured score must be finite; {evaluation.feedback}")
        return evaluation

    def _feedback(self, evaluation: Evaluation) -> str:
        return (
            f"Measured score: {evaluation.score}. {self.score_direction} is better. "
            f"{evaluation.feedback}"
        )

    async def _attempt(self, operation, source: str, feedback: str):
        history = [dict(row) for row in self.attempts]
        record = {"operation": operation.__name__, "source": source, "feedback": feedback}
        self.attempts.append(record)

        # Capture before Slick parses, so malformed outputs remain available for repair.
        async def acall(context):
            response, requests = await self.provider.acall(context)
            record["response"] = response
            if requests:
                raise _RejectedProposal("expected source code, not tool requests")
            return response, requests

        try:
            try:
                source = await operation(
                    source, feedback, history, provider=SimpleNamespace(acall=acall)
                )
            except (ValidationError, _RejectedProposal) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                return source, record["error"], None
            record["code"] = source
            evaluation = await self._measure(source)
            feedback = self._feedback(evaluation)
            record["score"], record["evaluation_feedback"] = evaluation.score, evaluation.feedback
            if evaluation.score is None:
                record["error"] = evaluation.feedback or "evaluator rejected the candidate"
                return source, record["error"], None
            candidate = Candidate(source, evaluation.score, evaluation.feedback)
            improves = (
                candidate.score > self.best.score
                if self.maximize
                else candidate.score < self.best.score
            )
            record["improved"] = improves
            if improves:
                self.best = candidate
            return source, feedback, candidate
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
