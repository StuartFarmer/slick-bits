"""Evolve reusable algorithms through probabilistic crossover and mutation."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError, field_validator
from slick import Session, prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Proposal(BaseModel, extra="forbid", frozen=True):
    description: Text
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class Individual(Proposal):
    id: int
    fitness: float = Field(allow_inf_nan=False)


class AEL:
    """Own a task and its evolution; the caller owns evaluation and provider resources."""

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        *,
        maximize: bool = False,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.maximize = maximize
        self.score_direction = "Higher" if maximize else "Lower"
        self.attempts: list[dict] = []
        self.history: list[tuple[Individual, ...]] = []

    @prompt(template="initialization.j2", output_type=Proposal)
    async def initialization(self, *, generated: Proposal) -> Proposal:
        """Generate a fresh candidate."""
        return generated

    @prompt(template="crossover.j2", output_type=Proposal)
    async def crossover(self, parents: list[Proposal], *, generated: Proposal) -> Proposal:
        """Combine the selected parents."""
        return generated

    @prompt(template="mutation.j2", output_type=Proposal)
    async def mutation(self, parents: list[Proposal], *, generated: Proposal) -> Proposal:
        """Revise one parent or newly crossed candidate."""
        return generated

    async def run(
        self,
        population_size: int = 10,
        generations: int = 10,
        parents: int = 2,
        offspring: int = 1,
        crossover: float = 1.0,
        mutation: float = 0.2,
        seed: int = 0,
        init_attempts: int | None = None,
        *,
        initial: Sequence[Proposal] = (),
        session: Session | None = None,
    ) -> list[Individual]:
        """Initialize, vary a generation snapshot, then retain stable elites.

        Direct-provider validation and evaluator ValueError/TimeoutError reject candidates.
        Other failures propagate; initialization has a fixed generation budget.
        Initial algorithms are evaluated first, stopping when the population is full.
        """
        init_attempts = 3 * population_size if init_attempts is None else init_attempts
        self.rng = random.Random(seed)
        self.execution = (
            {"session": session} if session is not None else {"provider": self.provider}
        )
        self.attempts, self.history = [], []
        population = await self._initialize_population(population_size, init_attempts, initial)
        population = self._select_survivors(population, population_size)
        for generation in range(1, generations + 1):
            children = await self._generate_offspring(
                population, generation, parents, offspring, crossover, mutation
            )
            population = self._select_survivors(population + children, population_size)
        return population

    async def _initialize_population(
        self, size: int, attempts: int, initial: Sequence[Proposal]
    ) -> list[Individual]:
        population = []
        for proposal in initial:
            identifier = len(self.attempts) + 1
            self.attempts.append(
                {
                    "id": identifier,
                    "generation": 0,
                    "operation": "seed",
                    "parents": (),
                    "proposal": proposal,
                }
            )
            candidate = await self._score((identifier, proposal))
            if candidate is not None:
                population.append(candidate)
            if len(population) == size:
                return population
        for _ in range(attempts):
            candidate = await self._score(await self._create(self.initialization, [], 0))
            if candidate is not None:
                population.append(candidate)
            if len(population) == size:
                return population
        raise RuntimeError("could not initialize a full valid population; inspect attempts")

    async def _generate_offspring(
        self, population, generation, parents, offspring, crossover, mutation
    ) -> list[Individual]:
        children = []
        for _ in range(len(population)):
            selected = self.rng.sample(population, parents)
            cross = self.rng.random() < crossover
            for _ in range(offspring):
                if cross:
                    draft = await self._create(
                        self.crossover, [(item.id, item) for item in selected], generation
                    )
                else:
                    clone = self.rng.choice(selected)
                    draft = clone.id, clone
                if draft is None:
                    continue
                mutate = self.rng.random() < mutation
                if mutate:
                    draft = await self._create(self.mutation, [draft], generation)
                candidate = await self._score(draft) if cross or mutate else clone
                if candidate is not None:
                    children.append(candidate)
        return children

    def _select_survivors(self, candidates: list[Individual], size: int) -> list[Individual]:
        # Stable ties retain incumbents; replacement happens after the full generation.
        population = sorted(candidates, key=lambda item: item.fitness, reverse=self.maximize)[:size]
        self.history.append(tuple(population))
        return population

    async def _create(self, operation, selected, generation):
        record = {
            "id": len(self.attempts) + 1,
            "generation": generation,
            "operation": operation.__name__,
            "parents": tuple(identifier for identifier, _ in selected),
        }
        self.attempts.append(record)
        try:
            inputs = ([item for _, item in selected],) if selected else ()
            proposal = await operation(*inputs, **self.execution)
            record["proposal"] = proposal
            return record["id"], proposal
        except ValidationError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            # Slick retains the failed parse as a pending conversation; the caller owns recovery.
            if "session" in self.execution:
                raise
            return None
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _score(self, draft) -> Individual | None:
        if draft is None:
            return None
        identifier, proposal = draft
        record = self.attempts[identifier - 1]
        try:
            fitness = float(await self.evaluate(proposal.content))
            if not math.isfinite(fitness):
                raise ValueError("fitness must be finite")
            record["fitness"] = fitness
            return Individual(
                description=proposal.description,
                content=proposal.content,
                id=identifier,
                fitness=fitness,
            )
        except (ValueError, TimeoutError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            return None
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
