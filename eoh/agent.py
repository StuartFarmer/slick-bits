"""Evolve heuristic descriptions and implementations with EoH's five operators."""

import math
import random
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError, field_validator
from slick import Session, prompt
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
    fitness: float = Field(allow_inf_nan=False)


class CandidateRejected(Exception):
    """The evaluator found an infeasible candidate; consume its attempt and continue."""


class EoH:
    """Own ranked evolution over arbitrary task content and a caller's evaluator."""

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        *,
        maximize: bool = True,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.maximize = maximize
        self.score_direction = "Higher" if maximize else "Lower"
        self.attempts: list[dict] = []
        self.history: list[tuple[Individual, ...]] = []
        self.evaluations = 0

    @prompt(template="initialize.j2", output_type=Proposal)
    async def initialize(self, *, generated: Proposal) -> Proposal:
        """Generate a fresh idea and candidate."""
        return generated

    @prompt(template="explore_diverse.j2", output_type=Proposal)
    async def explore_diverse(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Explore ideas unlike the selected parents (E1)."""
        return generated

    @prompt(template="explore_shared.j2", output_type=Proposal)
    async def explore_shared(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Explore a new variation of the parents' common idea (E2)."""
        return generated

    @prompt(template="modify_structure.j2", output_type=Proposal)
    async def modify_structure(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Modify a parent's structure (M1)."""
        return generated

    @prompt(template="tune_settings.j2", output_type=Proposal)
    async def tune_settings(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Tune a parent's choices while retaining its structure (M2)."""
        return generated

    @prompt(template="simplify.j2", output_type=Proposal)
    async def simplify(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Remove redundant components from a parent (M3)."""
        return generated

    async def run(
        self,
        population_size: int = 20,
        generations: int = 20,
        parents: int = 5,
        seed: int = 0,
        operators: tuple[str, ...] = OPERATORS,
        init_attempts: int | None = None,
        *,
        session: Session | None = None,
    ) -> list[Individual]:
        """Initialize, generate N attempts per selected operator, then select elites.

        Generated schema errors, explicit candidate rejections, nonfinite scores,
        and evaluation timeouts consume attempts. Unexpected errors abort.
        Provider failures consume attempts except with a caller-owned Session,
        whose pending conversation must be resumed by the caller.
        """
        init_attempts = 3 * population_size if init_attempts is None else init_attempts
        self.rng = random.Random(seed)
        self.session = session
        self.attempts, self.history = [], []
        self.evaluations = 0
        population = await self._initialize_population(population_size, init_attempts)
        population = self._select_survivors(population, population_size)
        for generation in range(1, generations + 1):
            offspring = await self._generate_offspring(population, generation, parents, operators)
            population = self._select_survivors(population + offspring, population_size)
        return population

    async def _initialize_population(self, size: int, attempts: int) -> list[Individual]:
        population = []
        for _ in range(attempts):
            candidate = await self._attempt("INIT", self.initialize, [], 0)
            if candidate is not None:
                population.append(candidate)
            if len(population) == size:
                return population
        raise RuntimeError("could not initialize a full valid population; inspect attempts")

    async def _generate_offspring(self, population, generation, parents, operators):
        strategies = {
            "E1": (self.explore_diverse, parents),
            "E2": (self.explore_shared, parents),
            "M1": (self.modify_structure, 1),
            "M2": (self.tune_settings, 1),
            "M3": (self.simplify, 1),
        }
        offspring = []
        for operation in operators:
            propose, count = strategies[operation]
            for _ in range(len(population)):
                selected = self._select_parents(population, count)
                candidate = await self._attempt(operation, propose, selected, generation)
                if candidate is not None:
                    offspring.append(candidate)
        return offspring

    def _select_parents(self, population: list[Individual], count: int) -> list[Individual]:
        # Official v0.1 prob_rank.py: one-based ranks, sampling with replacement.
        weights = [1 / (rank + len(population)) for rank in range(1, len(population) + 1)]
        return self.rng.choices(population, weights=weights, k=count)

    def _select_survivors(self, candidates: list[Individual], size: int) -> list[Individual]:
        population = sorted(candidates, key=lambda item: item.fitness, reverse=self.maximize)[:size]
        self.history.append(tuple(population))
        return population

    async def _attempt(self, operation, propose, selected, generation) -> Individual | None:
        record = {
            "id": len(self.attempts) + 1,
            "generation": generation,
            "operation": operation,
            "parents": tuple(item.id for item in selected),
        }
        self.attempts.append(record)

        async def recorded_call(context, **kwargs):
            if self.session is None:
                response, requests = await self.provider.acall(context, **kwargs)
            else:
                # Finish the Session as text before Slick parses the proposal. A
                # malformed JSON response must not leave its conversation pending.
                response, requests = await self.session.arun(context), []
            record["raw_response"] = response
            return response, requests

        try:
            inputs = (selected,) if selected else ()
            try:
                proposal = await propose(*inputs, provider=SimpleNamespace(acall=recorded_call))
            except ValidationError as exc:
                raise CandidateRejected(f"invalid generated proposal: {exc}") from exc
            except (ProviderError, TimeoutError) as exc:
                if self.session is not None:
                    raise
                raise CandidateRejected(f"generation failed: {exc}") from exc
            record["proposal"] = proposal
            self.evaluations += 1
            try:
                fitness = float(await self.evaluate(proposal.content))
            except TimeoutError as exc:
                raise CandidateRejected(f"evaluation timed out: {exc}") from exc
            if not math.isfinite(fitness):
                raise CandidateRejected("fitness must be finite")
            record["fitness"] = fitness
            record["status"] = "accepted"
            return Individual(**proposal.model_dump(), id=record["id"], fitness=fitness)
        except CandidateRejected as exc:
            record["status"] = "rejected"
            record["error"] = f"{type(exc).__name__}: {exc}"
            return None
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
