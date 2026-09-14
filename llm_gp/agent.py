"""Generate candidates, evolve them, and optionally delegate selection to the model."""

import math
import random
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, StrictInt, ValidationError
from slick import Session, prompt


def _nonblank(content: str) -> str:
    if not content.strip():
        raise ValueError("candidate content must not be blank")
    return content


Text = Annotated[str, Field(min_length=1), AfterValidator(_nonblank)]


class Proposal(BaseModel, extra="forbid"):
    content: Text


class Children(BaseModel, extra="forbid"):
    contents: list[Text] = Field(min_length=2, max_length=2)


class Selection(BaseModel, extra="forbid"):
    ids: list[StrictInt] = Field(min_length=1)


class Individual(BaseModel, extra="forbid", frozen=True):
    content: Text
    score: float = Field(allow_inf_nan=False)


class Result(BaseModel, extra="forbid"):
    best: Individual | None
    designated_best: Individual | None
    population: list[Individual]
    history: list[list[Individual]]
    calls: int
    evaluations: int
    errors: list[str]
    stop_reason: Literal["generations", "budget"]


class BudgetExceeded(Exception):
    """Return the evaluated candidates when the generation budget is exhausted."""


class InvalidSelection(ValueError):
    """Generated IDs do not identify the requested selection."""


class LLMGP:
    """Evolve arbitrary text; the caller owns evaluation and provider retries."""

    def __init__(
        self,
        task: str,
        provider,
        evaluate: Callable[[str], Awaitable[float]],
        *,
        variant: Literal["llm-gp-mu-xo", "llm-gp"] = "llm-gp-mu-xo",
        population_size: int = 10,
        generations: int = 30,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        n_shots: int = 2,
        seed: int = 0,
        max_calls: int = 10000,
        maximize: bool = False,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.variant, self.population_size, self.generations = variant, population_size, generations
        self.crossover_rate, self.mutation_rate = crossover_rate, mutation_rate
        self.n_shots, self.max_calls, self.maximize = n_shots, max_calls, maximize
        self.rng = random.Random(seed)
        self.calls = self.evaluations = 0
        self.errors: list[str] = []
        self.best: Individual | None = None
        self._scores: dict[str, Individual | None] = {}
        self.population: list[Individual] = []
        self.pending: list[Individual] = []
        self.history: list[list[Individual]] = []

    @prompt(template="initialize.j2", output_type=Proposal)
    async def initialize(self, *, generated: Proposal) -> str:
        return generated.content

    @prompt(template="mutate.j2", output_type=Proposal)
    async def mutate(self, candidate: str, samples: list[str], *, generated: Proposal) -> str:
        return generated.content

    @prompt(template="crossover.j2", output_type=Children)
    async def crossover(
        self, parents: list[str], samples: list[str], *, generated: Children
    ) -> list[str]:
        return generated.contents

    @property
    def score_direction(self) -> str:
        return "Higher" if self.maximize else "Lower"

    @prompt(template="select_parents.j2", output_type=Selection)
    async def select_parents(
        self, population: list[Individual], *, generated: Selection
    ) -> list[Individual]:
        return self._resolve_selection(generated, population, 2, unique=False)

    @prompt(template="replace_population.j2", output_type=Selection)
    async def replace_population(
        self, population: list[Individual], *, generated: Selection
    ) -> list[Individual]:
        return self._resolve_selection(generated, population, self.population_size, unique=True)

    @prompt(template="designate_best.j2", output_type=Selection)
    async def designate_best(
        self, population: list[Individual], *, generated: Selection
    ) -> Individual:
        return self._resolve_selection(generated, population, 1, unique=True)[0]

    def _resolve_selection(self, generated, population, count, *, unique):
        if len(generated.ids) != count or any(i < 0 or i >= len(population) for i in generated.ids):
            raise InvalidSelection("wrong count or unknown individual ID")
        if unique and len(set(generated.ids)) != count:
            raise InvalidSelection("replacement and final selection require distinct IDs")
        return [population[i] for i in generated.ids]

    def _rank(self, individual: Individual) -> float:
        return -individual.score if self.maximize else individual.score

    async def _generate(self, operation, *args, session):
        if self.calls >= self.max_calls:
            raise BudgetExceeded
        self.calls += 1
        try:
            resource = {"session": session} if session is not None else {"provider": self.provider}
            return await operation(*args, **resource)
        except (ValidationError, InvalidSelection) as error:
            self.errors.append(f"{operation.__name__}: {error}")
            return None

    async def _assess(self, content: str) -> Individual | None:
        self.evaluations += 1
        if content not in self._scores:
            try:
                score = float(await self.evaluate(content))
                if not math.isfinite(score):
                    raise ValueError("evaluation must return a finite score")
                self._scores[content] = Individual(content=content, score=score)
            except (ValueError, ArithmeticError) as error:
                self.errors.append(f"evaluate {content!r}: {type(error).__name__}: {error}")
                self._scores[content] = None
        individual = self._scores[content]
        if individual is not None and (
            self.best is None or self._rank(individual) < self._rank(self.best)
        ):
            self.best = individual
        return individual

    async def _initialize_population(self, session):
        while len(self.pending) < self.population_size:
            content = await self._generate(self.initialize, session=session)
            if content is not None:
                await self._add_candidate(content)

    async def _add_candidate(self, content):
        individual = await self._assess(content)
        if individual is not None:
            self.pending.append(individual)

    async def _choose_parents(self, session):
        if self.variant == "llm-gp":
            parents = await self._generate(self.select_parents, self.population, session=session)
            return parents or self.rng.choices(self.population, k=2)
        # Upstream tournaments draw distinct competitors; winners may repeat.
        return [
            min(self.rng.sample(self.population, min(2, len(self.population))), key=self._rank)
            for _ in range(2)
        ]

    async def _vary_parents(self, parents, session):
        children = [parent.content for parent in parents]
        samples = self.rng.sample(
            [p.content for p in self.population], min(self.n_shots, len(self.population))
        )
        if self.rng.random() < self.crossover_rate:
            children = (
                await self._generate(self.crossover, children, samples, session=session) or children
            )
        for content in children[: self.population_size - len(self.pending)]:
            if self.rng.random() < self.mutation_rate:
                content = (
                    await self._generate(self.mutate, content, samples, session=session) or content
                )
            await self._add_candidate(content)

    async def _evolve_population(self, session):
        self.pending = []
        while len(self.pending) < self.population_size:
            parents = await self._choose_parents(session)
            await self._vary_parents(parents, session)
        if self.variant == "llm-gp":
            pool = self.population + self.pending
            self.pending = (
                await self._generate(self.replace_population, pool, session=session)
                or sorted(pool, key=self._rank)[: self.population_size]
            )
        else:
            # Tutorial_GP-LLM evaluates n offspring, then adds the old elite and truncates.
            pool = self.pending + [min(self.population, key=self._rank)]
            self.pending = sorted(pool, key=self._rank)[: self.population_size]

    def _keep_generation(self):
        self.population = self.pending
        self.history.append(self.population[:])

    async def run(self, *, session: Session | None = None) -> Result:
        """Initialize, evolve, and designate; use a fresh instance for each run."""
        stop_reason = "generations"
        try:
            await self._initialize_population(session)
            self._keep_generation()
            for _ in range(1, self.generations):
                await self._evolve_population(session)
                self._keep_generation()
            designated = self.best
            if self.variant == "llm-gp":
                designated = await self._generate(
                    self.designate_best, self.population, session=session
                ) or min(self.population, key=self._rank)
        except BudgetExceeded:
            stop_reason = "budget"
            if self.pending and self.pending is not self.population:
                self._keep_generation()
            designated = self.best
        return Result(
            best=self.best,
            designated_best=designated,
            population=self.population,
            history=self.history,
            calls=self.calls,
            evaluations=self.evaluations,
            errors=self.errors,
            stop_reason=stop_reason,
        )
