"""Evolve textual ideas with ranked exploration and modification operators."""

import math
import random
from collections.abc import Awaitable, Callable
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints
from slick import Session, prompt
from slick.providers import Provider, ProviderError

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OPERATORS = ("E1", "E2", "M1", "M2", "M3")


class Proposal(BaseModel, extra="forbid", frozen=True):
    description: Text
    content: Text


class Individual(Proposal):
    id: int
    fitness: float = Field(allow_inf_nan=False)


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

        Parse, provider, evaluation-value and timeout failures consume attempts.
        Initialization is bounded; other unexpected errors abort.
        """
        init_attempts = 3 * population_size if init_attempts is None else init_attempts
        self.rng = random.Random(seed)
        self.execution = (
            {"session": session} if session is not None else {"provider": self.provider}
        )
        self.attempts, self.history = [], []
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
        # One-based rank weights; sample without replacement from the generation snapshot.
        ranked = population.copy()
        weights = [1 / (rank + len(ranked)) for rank in range(1, len(ranked) + 1)]
        selected = []
        for _ in range(count):
            index = self.rng.choices(range(len(ranked)), weights=weights, k=1)[0]
            selected.append(ranked.pop(index))
            weights.pop(index)
        return selected

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
        try:
            inputs = (selected,) if selected else ()
            proposal = await propose(*inputs, **self.execution)
            record["proposal"] = proposal
            fitness = float(await self.evaluate(proposal.content))
            if not math.isfinite(fitness):
                raise ValueError("fitness must be finite")
            record["fitness"] = fitness
            return Individual(**proposal.model_dump(), id=record["id"], fitness=fitness)
        except (ProviderError, ValueError, TimeoutError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            return None
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
