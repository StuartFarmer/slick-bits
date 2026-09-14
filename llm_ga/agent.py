"""Evolve ideas and candidate content with operator-specific rejection feedback."""

import math
import random
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError, field_validator
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OPERATORS = ("E1", "E2", "M1", "M2")


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
    fitness: float = Field(allow_inf_nan=False)


class CandidateRejected(Exception):
    """The evaluator found an invalid candidate; blacklist it and continue."""


class LLMGA:
    """Minimize caller-measured fitness; candidate execution belongs to the evaluator."""

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        *,
        template: str = "",
        information: str = "",
        requirements: str = "",
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.template = template
        self.information = information
        self.requirements = requirements
        self.attempts: list[dict] = []
        self.blacklists: dict[str, list[dict]] = {op: [] for op in OPERATORS}
        self.history: list[tuple[Individual, ...]] = []
        self.evaluations = 0

    @prompt(template="initialize.j2", output_type=Proposal)
    async def initialize(self, *, generated: Proposal) -> Proposal:
        """Generate an initial idea and its implementation."""
        return generated

    @prompt(template="explore_diverse.j2", output_type=Proposal)
    async def explore_diverse(
        self, parents: list[Individual], blacklist: list[dict], *, generated: Proposal
    ) -> Proposal:
        """E1: explore an idea substantially different from its parents."""
        return generated

    @prompt(template="explore_shared.j2", output_type=Proposal)
    async def explore_shared(
        self, parents: list[Individual], blacklist: list[dict], *, generated: Proposal
    ) -> Proposal:
        """E2: recombine the parents' common ideas into a new implementation."""
        return generated

    @prompt(template="modify_structure.j2", output_type=Proposal)
    async def modify_structure(
        self, parents: list[Individual], blacklist: list[dict], *, generated: Proposal
    ) -> Proposal:
        """M1: refine one parent's structure to improve performance."""
        return generated

    @prompt(template="tune_settings.j2", output_type=Proposal)
    async def tune_settings(
        self, parents: list[Individual], blacklist: list[dict], *, generated: Proposal
    ) -> Proposal:
        """M2: tune settings while preserving one parent's structure."""
        return generated

    async def run(
        self,
        population_size: int = 8,
        generations: int = 20,
        parents: int = 2,
        seed: int = 2024,
        init_attempts: int | None = None,
    ) -> list[Individual]:
        """Initialize, evolve four operator batches per generation, and return elites.

        Each batch consumes N proposal attempts against a fixed population snapshot.
        Calls are independent and sequential. Each run resets the in-memory records.
        """
        self.attempts = []
        self.blacklists = {op: [] for op in OPERATORS}
        self.history = []
        self.evaluations = 0
        self.rng = random.Random(seed)
        budget = 3 * population_size if init_attempts is None else init_attempts
        population = await self._initialize_population(population_size, budget)
        self.history.append(tuple(population))
        for generation in range(1, generations + 1):
            for operation in OPERATORS:
                offspring = await self._evolve_batch(population, operation, parents, generation)
                population = sorted(population + offspring, key=lambda p: p.fitness)[
                    :population_size
                ]
            self.history.append(tuple(population))
        return population

    async def _initialize_population(self, size: int, budget: int) -> list[Individual]:
        population = []
        for _ in range(budget):
            candidate = await self._attempt("INIT", [], 0)
            if candidate is not None:
                population.append(candidate)
            if len(population) == size:
                return sorted(population, key=lambda p: p.fitness)
        raise RuntimeError("could not initialize a full population; inspect attempts")

    async def _evolve_batch(
        self, population: list[Individual], operation: str, parents: int, generation: int
    ) -> list[Individual]:
        offspring = []
        count = parents if operation in ("E1", "E2") else 1
        # Official prob_rank.py samples WITH replacement using one-based ranks.
        weights = [1 / (rank + len(population)) for rank in range(1, len(population) + 1)]
        for _ in population:
            selected = self.rng.choices(population, weights=weights, k=count)
            candidate = await self._attempt(operation, selected, generation, population[-1].fitness)
            if candidate is not None:
                offspring.append(candidate)
        return offspring

    async def _attempt(
        self, operation: str, parents: list[Individual], generation: int, worst: float = math.inf
    ) -> Individual | None:
        record = {
            "id": len(self.attempts) + 1,
            "generation": generation,
            "operation": operation,
            "parents": tuple(p.id for p in parents),
        }
        self.attempts.append(record)

        async def recorded_call(context, **kwargs):
            response, requests = await self.provider.acall(context, **kwargs)
            record["raw_response"] = response
            return response, requests

        execution = SimpleNamespace(acall=recorded_call)
        methods = {
            "INIT": self.initialize,
            "E1": self.explore_diverse,
            "E2": self.explore_shared,
            "M1": self.modify_structure,
            "M2": self.tune_settings,
        }
        args = () if operation == "INIT" else (parents, self.blacklists[operation])
        try:
            try:
                proposal = await methods[operation](*args, provider=execution)
            except ValidationError as exc:
                raise CandidateRejected(f"invalid generated proposal: {exc}") from exc
            record["proposal"] = proposal.model_dump()
            self.evaluations += 1
            try:
                fitness = float(await self.evaluate(proposal.content))
            except TimeoutError as exc:
                raise CandidateRejected(f"candidate evaluation timed out: {exc}") from exc
            if not math.isfinite(fitness):
                raise CandidateRejected("evaluated fitness must be finite")
            record["fitness"] = fitness
            if fitness >= worst:
                raise CandidateRejected("fitness must be strictly less than the batch's worst")
            record["status"] = "accepted"
            return Individual(**proposal.model_dump(), id=record["id"], fitness=fitness)
        except CandidateRejected as exc:
            record["status"] = "rejected"
            record["error"] = f"{type(exc).__name__}: {exc}"
            if operation != "INIT":
                self.blacklists[operation].append(record)
            return None
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
