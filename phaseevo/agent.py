"""Jointly evolve instructions and demonstrations through four ordered phases."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Candidate(BaseModel, extra="forbid"):
    instruction: Text
    examples: list[Text]


class Feedback(BaseModel, extra="forbid"):
    text: Text


@dataclass(frozen=True)
class Evaluation:
    score: float
    outcomes: tuple[bool, ...]
    errors: str


@dataclass(frozen=True)
class Individual:
    candidate: Candidate
    evaluation: Evaluation


class PhaseEvo:
    """Maximize finite scores; evaluator supplies aligned development outcomes/errors.

    All failures propagate. Minimum rounds and stalled-round patience control phase
    transitions, with a hard cap per phase to bound successful searches as well.
    """

    def __init__(
        self, task: str, provider: Provider, evaluate: Callable[[Candidate], Awaitable[Evaluation]]
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate

    @prompt(template="initialize.j2", output_type=Candidate)
    async def initialize(self, examples: Sequence[str], *, generated: Candidate) -> Candidate:
        return generated

    @prompt(template="examine.j2", output_type=Feedback)
    async def examine(self, candidate: Candidate, errors: str, *, generated: Feedback) -> str:
        return generated.text

    @prompt(template="improve.j2", output_type=Candidate)
    async def improve(
        self, candidate: Candidate, feedback: str, *, generated: Candidate
    ) -> Candidate:
        return generated

    @prompt(template="crossover.j2", output_type=Candidate)
    async def crossover(
        self, first: Candidate, second: Candidate, *, generated: Candidate
    ) -> Candidate:
        if not set(generated.examples) <= set(first.examples + second.examples):
            raise ValueError("crossover invented an example outside its parents")
        return generated

    @prompt(template="distribution.j2", output_type=Candidate)
    async def distribution(
        self, parents: Sequence[Candidate], *, generated: Candidate
    ) -> Candidate:
        if not set(generated.examples) <= {example for p in parents for example in p.examples}:
            raise ValueError("distribution mutation invented an example outside its parents")
        return generated

    @prompt(template="semantic.j2", output_type=Candidate)
    async def semantic(self, candidate: Candidate, *, generated: Candidate) -> Candidate:
        if not set(generated.examples) <= set(candidate.examples):
            raise ValueError("semantic mutation added an example")
        return generated

    async def _assess(self, candidate):
        evaluation = await self.evaluate(candidate)
        self.evaluations += 1
        if not math.isfinite(evaluation.score):
            raise ValueError("measured score must be finite")
        return Individual(candidate, evaluation)

    async def _feedback_round(self, population):
        children = []
        for parent in population:
            feedback = await self.examine(
                parent.candidate, parent.evaluation.errors, provider=self.provider
            )
            candidate = await self.improve(parent.candidate, feedback, provider=self.provider)
            children.append(await self._assess(candidate))
        return children

    def _distinct_parents(self, population, count):
        anchor = self.rng.choice(population)
        others = [p for p in population if p is not anchor]
        # Maximal Hamming distance chooses complementary error patterns.
        others.sort(
            key=lambda p: sum(
                a != b
                for a, b in zip(anchor.evaluation.outcomes, p.evaluation.outcomes, strict=True)
            ),
            reverse=True,
        )
        return [anchor, *others[: count - 1]]

    async def _evolution_round(self, population):
        children = []
        for _ in population:
            parents = self._distinct_parents(population, min(3, len(population)))
            if self.rng.random() < 0.5:
                candidate = await self.crossover(
                    parents[0].candidate, parents[1].candidate, provider=self.provider
                )
            else:
                parents.sort(key=lambda p: p.evaluation.score)
                candidate = await self.distribution(
                    [p.candidate for p in parents], provider=self.provider
                )
            children.append(await self._assess(candidate))
        return children

    async def _semantic_round(self, population):
        return [
            await self._assess(await self.semantic(p.candidate, provider=self.provider))
            for p in population
        ]

    async def _phase(self, population, operation, minimum, patience, cap):
        stalled = 0
        for step in range(cap):
            previous = sum(p.evaluation.score / len(population) for p in population)
            children = await operation(population)
            population = sorted(
                population + children, key=lambda p: p.evaluation.score, reverse=True
            )[: len(population)]
            current = sum(p.evaluation.score / len(population) for p in population)
            stalled = stalled + 1 if current <= previous else 0
            if step + 1 >= minimum and stalled >= patience:
                break
        return population

    async def run(
        self,
        examples: Sequence[str],
        *,
        population_size: int = 8,
        initial: Sequence[Candidate] = (),
        minimum_rounds: int = 1,
        patience: tuple[int, int, int] = (1, 3, 1),
        max_rounds: int = 10,
        seed: int = 0,
    ) -> dict:
        self.rng = random.Random(seed)
        self.evaluations = 0
        candidates = list(initial)
        for _ in range(population_size - len(candidates)):
            candidate = await self.initialize(examples, provider=self.provider)
            candidates.append(candidate)
        population = [await self._assess(c) for c in candidates]
        history = [max(p.evaluation.score for p in population)]
        for operation, tolerance in zip(
            (self._feedback_round, self._evolution_round, self._semantic_round),
            patience,
            strict=True,
        ):
            population = await self._phase(
                population, operation, minimum_rounds, tolerance, max_rounds
            )
            history.append(max(p.evaluation.score for p in population))
        return {
            "best": max(population, key=lambda p: p.evaluation.score),
            "population": population,
            "history": history,
            "evaluations": self.evaluations,
        }
