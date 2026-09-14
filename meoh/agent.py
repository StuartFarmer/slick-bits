"""Evolve a Pareto archive using MEOH's dominance-masked code similarity."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from itertools import cycle
from types import SimpleNamespace
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError, field_validator
from slick import prompt
from slick.providers import Provider, ProviderError

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OPERATORS = ("E1", "E2", "M1", "M2", "M3")


class Proposal(BaseModel, extra="forbid", frozen=True):
    description: Text
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, content: str) -> str:
        if not content.strip():
            raise ValueError("candidate content must not be blank")
        return content


class Individual(Proposal):
    id: int
    objectives: tuple[float, ...]


class CandidateRejected(Exception):
    """Signal an infeasible candidate; consume the attempt and retain incumbents."""


def python_similarity(reference: str, candidate: str) -> float:
    """Use LLM4AD's directional CodeBLEU AST match; requires requirements.txt."""
    from codebleu.syntax_match import calc_syntax_match

    return calc_syntax_match([reference], candidate, "python")


def dominates(a: Sequence[float], b: Sequence[float], maximize: tuple[bool, ...]) -> bool:
    """Strict Pareto dominance with a direction for each measured objective."""
    comparisons = [
        (x >= y, x > y) if high else (x <= y, x < y)
        for x, y, high in zip(a, b, maximize, strict=True)
    ]
    return all(weak for weak, _ in comparisons) and any(strict for _, strict in comparisons)


def dominance_scores(
    population: Sequence[Individual],
    maximize: tuple[bool, ...],
    similarity: Callable[[str, str], float],
) -> list[float]:
    """Column sums of negative similarity, masked by strict dominance (Appendix A)."""
    scores = [0.0] * len(population)
    # ponytail: quadratic pair scans; cache deterministic AST matches for large populations.
    for j, candidate in enumerate(population):
        for parent in population:
            if dominates(parent.objectives, candidate.objectives, maximize):
                value = similarity(parent.content, candidate.content)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("measured similarity must be finite and within [0, 1]")
                scores[j] -= value
    return scores


class MEOH:
    """Own generation, evaluation, dominance selection, and a non-dominated archive.

    The caller owns evaluator isolation and provider transport retries. Calls are
    independent (no shared conversation). Use a fresh instance per experiment.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Sequence[float]]],
        *,
        maximize: tuple[bool, ...],
        similarity: Callable[[str, str], float] = python_similarity,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.maximize, self.similarity = maximize, similarity
        self.objective_directions = tuple("maximize" if high else "minimize" for high in maximize)
        self.population: list[Individual] = []
        self.archive: list[Individual] = []
        self.attempts: list[dict] = []
        self.history: list[tuple[Individual, ...]] = []
        self.evaluations = 0

    @prompt(template="initialize.j2", output_type=Proposal)
    async def initialize(self, *, generated: Proposal) -> Proposal:
        """Generate an initial heuristic without parents."""
        return generated

    @prompt(template="explore_diverse.j2", output_type=Proposal)
    async def explore_diverse(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Explore a different heuristic form (E1)."""
        return generated

    @prompt(template="explore_shared.j2", output_type=Proposal)
    async def explore_shared(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Build on the parents' common backbone (E2)."""
        return generated

    @prompt(template="modify_structure.j2", output_type=Proposal)
    async def modify_structure(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Modify one heuristic's structure (M1)."""
        return generated

    @prompt(template="tune_settings.j2", output_type=Proposal)
    async def tune_settings(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Change one heuristic's parameter settings (M2)."""
        return generated

    @prompt(template="simplify.j2", output_type=Proposal)
    async def simplify(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Remove components prone to overfitting, retaining the interface (M3)."""
        return generated

    async def run(
        self,
        population_size: int = 20,
        generations: int = 20,
        parents: int = 5,
        seed: int = 0,
        operators: tuple[str, ...] = OPERATORS,
        init_attempts: int | None = None,
    ) -> list[Individual]:
        """Generate N initial successes, then N attempts per generation.

        Cycle through operators across generations. Each accepted offspring joins
        the parent pool immediately, following Algorithm 1. Truncate once per
        generation and return all non-dominated evaluated artifacts in the archive.
        """
        self.rng = random.Random(seed)
        self.population, self.archive, self.attempts, self.history = [], [], [], []
        self.evaluations = 0
        budget = 3 * population_size if init_attempts is None else init_attempts
        self.population = await self._initialize_population(population_size, budget)
        self.history.append(tuple(self.population))
        schedule = cycle(operators)
        for generation in range(1, generations + 1):
            pool = await self._generate_offspring(generation, population_size, parents, schedule)
            self.population = self._select_survivors(pool, population_size)
            self.history.append(tuple(self.population))
        return list(self.archive)

    async def _initialize_population(self, size: int, budget: int) -> list[Individual]:
        population = []
        for _ in range(budget):
            candidate = await self._attempt("INIT", self.initialize, [], 0)
            if candidate is not None:
                population.append(candidate)
            if len(population) == size:
                return population
        raise RuntimeError("could not initialize a full population; inspect attempts")

    async def _generate_offspring(self, generation, count, parents, schedule):
        strategies = {
            "E1": (self.explore_diverse, parents),
            "E2": (self.explore_shared, parents),
            "M1": (self.modify_structure, 1),
            "M2": (self.tune_settings, 1),
            "M3": (self.simplify, 1),
        }
        pool = list(self.population)
        for _ in range(count):
            operation = next(schedule)
            propose, parent_count = strategies[operation]
            selected = self._select_parents(pool, parent_count)
            candidate = await self._attempt(operation, propose, selected, generation)
            if candidate is not None:
                pool.append(candidate)
        return pool

    def _select_parents(self, population: list[Individual], count: int) -> list[Individual]:
        scores = dominance_scores(population, self.maximize, self.similarity)
        best = max(scores)
        weights = [math.exp(score - best) for score in scores]
        # Repeated independent draws, as in LLM4AD: a parent may occur more than once.
        return self.rng.choices(population, weights=weights, k=count)

    def _select_survivors(self, population: list[Individual], size: int) -> list[Individual]:
        scores = dominance_scores(population, self.maximize, self.similarity)
        order = sorted(range(len(population)), key=scores.__getitem__, reverse=True)
        return [population[index] for index in order[:size]]

    def _update_archive(self, candidate: Individual) -> None:
        if any(
            dominates(item.objectives, candidate.objectives, self.maximize) for item in self.archive
        ):
            return
        self.archive = [
            item
            for item in self.archive
            if not dominates(candidate.objectives, item.objectives, self.maximize)
        ]
        if not any(
            item.content == candidate.content and item.objectives == candidate.objectives
            for item in self.archive
        ):
            self.archive.append(candidate)

    async def _attempt(self, operation, propose, selected, generation) -> Individual | None:
        record = {
            "id": len(self.attempts) + 1,
            "generation": generation,
            "operation": operation,
            "parents": tuple(item.id for item in selected),
        }
        self.attempts.append(record)

        async def recorded_call(context, **kwargs):
            response, requests = await self.provider.acall(context, **kwargs)
            record["raw_response"] = response
            return response, requests

        try:
            try:
                args = () if operation == "INIT" else (selected,)
                proposal = await propose(*args, provider=SimpleNamespace(acall=recorded_call))
            except (ValidationError, ProviderError, TimeoutError) as exc:
                raise CandidateRejected(f"generation failed: {exc}") from exc
            record["proposal"] = proposal
            self.evaluations += 1
            try:
                objectives = tuple(await self.evaluate(proposal.content))
            except TimeoutError as exc:
                raise CandidateRejected(f"evaluation timed out: {exc}") from exc
            record["objectives"] = objectives
            if len(objectives) != len(self.maximize) or not all(map(math.isfinite, objectives)):
                raise CandidateRejected(
                    "objectives must be finite and match the declared dimensions"
                )
            candidate = Individual(**proposal.model_dump(), id=record["id"], objectives=objectives)
            self._update_archive(candidate)
            record["status"] = "accepted"
            return candidate
        except CandidateRejected as exc:
            record.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
            return None
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
