"""Recursively improve optimizer source against a local downstream meta-utility."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from statistics import fmean

from pydantic import BaseModel, Field
from slick import prompt
from slick.providers import Provider

SEED_IMPROVER = """async def improve(initial, capabilities):
    candidates = []
    count = min(capabilities.generations_left, capabilities.evaluations_left)
    for _ in range(count):
        candidate = await capabilities.suggest(initial)
        candidates.append((await capabilities.evaluate(candidate), candidate))
    return max(candidates, key=lambda pair: pair[0])[1]
"""


class Program(BaseModel, extra="forbid"):
    code: str = Field(min_length=1)


class ImproverExecutionError(Exception):
    """The isolated executor reports generated-code failure, including timeout."""


class BudgetExhausted(ImproverExecutionError):
    pass


@dataclass(frozen=True)
class Problem:
    initial: str
    utility_description: str
    evaluate: Callable[[str], Awaitable[float]]


class Capabilities:
    """Expose budgeted generation/evaluation endpoints to an isolated improver."""

    def __init__(self, suggest, evaluate, generations, evaluations):
        self._suggest, self._evaluate = suggest, evaluate
        self.generations_left, self.evaluations_left = generations, evaluations

    async def suggest(self, initial: str, guidance: str = "") -> str:
        if self.generations_left <= 0:
            raise BudgetExhausted("generation budget exhausted")
        self.generations_left -= 1
        return await self._suggest(initial, guidance)

    async def evaluate(self, source: str) -> float:
        if self.evaluations_left <= 0:
            raise BudgetExhausted("utility budget exhausted")
        self.evaluations_left -= 1
        score = await self._evaluate(source)
        if not math.isfinite(score):
            raise ValueError("utility must be finite")
        return score


class STOP:
    """The executor runs source with RPC access to capabilities in caller-owned isolation.

    It must not execute source in the host process. Only explicit generated-program
    failures are converted to zero utility; transport and evaluator failures propagate.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        execute: Callable[[str, str, Capabilities], Awaitable[str]],
        problems: Sequence[Problem],
    ):
        self.task, self.provider, self.execute = task, provider, execute
        self.problems = tuple(problems)

    @prompt(template="improve.j2", output_type=Program)
    async def improve(
        self, initial: str, utility: str, guidance: str, *, generated: Program
    ) -> str:
        if not generated.code.strip():
            raise ValueError("empty generated program")
        return generated.code

    def capabilities(self, utility, description, generations, evaluations):
        async def suggest(initial, guidance):
            self.generation_calls += 1
            return await self.improve(initial, description, guidance, provider=self.provider)

        async def measure(source):
            self.utility_calls += 1
            return await utility(source)

        return Capabilities(suggest, measure, generations, evaluations)

    async def meta_utility(self, source):
        self.meta_calls += 1
        scores = []
        for problem in self.problems:
            caps = self.capabilities(
                problem.evaluate,
                problem.utility_description,
                self.inner_generations,
                self.inner_evaluations,
            )
            try:
                improved = await self.execute(source, problem.initial, caps)
            except ImproverExecutionError:
                return 0.0
            if not improved.strip():
                return 0.0
            # Final checking is separate from the improver's search budget, as upstream.
            score = await problem.evaluate(improved)
            self.utility_calls += 1
            if not math.isfinite(score):
                raise ValueError("downstream utility must be finite")
            scores.append(score)
        return fmean(scores)

    async def self_improve(self, optimizer, current, description, generations, evaluations):
        caps = self.capabilities(self.meta_utility, description, generations, evaluations)
        try:
            candidate = await self.execute(optimizer, current, caps)
        except ImproverExecutionError:
            return current, None
        if not candidate.strip():
            return current, None
        score = await self.meta_utility(candidate)
        return (candidate, score) if score != 0 else (current, None)

    async def run(
        self,
        initial: str = SEED_IMPROVER,
        *,
        rounds: int = 3,
        generations: int = 4,
        evaluations: int = 4,
        inner_generations: int = 4,
        inner_evaluations: int = 4,
    ) -> dict:
        self.inner_generations, self.inner_evaluations = inner_generations, inner_evaluations
        self.generation_calls = self.utility_calls = self.meta_calls = 0
        current = optimizer = previous_optimizer = initial
        history = []
        description = (
            "Mean downstream utility after this improver improves the supplied initial "
            "solutions using budgeted capabilities. Implement async improve(initial, capabilities)."
        )
        for _ in range(rounds):
            candidate, score = await self.self_improve(
                optimizer, current, description, generations, evaluations
            )
            if score is None:
                optimizer = previous_optimizer
            else:
                previous_optimizer, optimizer, current = optimizer, candidate, candidate
            history.append({"source": current, "optimizer": optimizer, "checked_score": score})
        return {
            "improver": current,
            "history": history,
            "generation_calls": self.generation_calls,
            "utility_calls": self.utility_calls,
            "meta_calls": self.meta_calls,
        }
