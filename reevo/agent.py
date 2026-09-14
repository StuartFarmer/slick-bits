"""Reflective Evolution adapted from ai4co/reevo; see README.md and LICENSE."""

import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import combinations

from slick import Provider, Session, prompt


@dataclass(frozen=True)
class Config:
    population_size: int = 10
    initial_size: int = 30
    max_evaluations: int = 100
    crossover_rate: float = 1.0
    mutation_rate: float = 0.5
    short_reflection: bool = True
    long_reflection: bool = True
    maximize: bool = False
    seed: int = 0
    black_box: bool = False


@dataclass
class Individual:
    id: int
    candidate: str
    stage: str
    generation: int
    parents: list[int] = field(default_factory=list)
    score: float | None = None
    error: str = ""


@dataclass
class Result:
    individuals: list[Individual] = field(default_factory=list)
    reflections: list[dict] = field(default_factory=list)
    best: Individual | None = None
    best_history: list[float | None] = field(default_factory=list)
    stop_reason: str = ""


class ReEvo:
    """Reflect on candidate pairs and mutate elites; lower scores win by default.

    Each generated candidate or supplied seed consumes one evaluation
    slot, including evaluator failures. Provider failures abort and cancellation
    propagates. Instances own one run; callers own template and session setup.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        *,
        config: Config | None = None,
        seed_candidate: str | None = None,
        initial_reflection: str = "",
        initial_provider: Provider | None = None,
        reflector_provider: Provider | None = None,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.initial_provider = initial_provider if initial_provider is not None else provider
        self.reflector_provider = reflector_provider if reflector_provider is not None else provider
        self.config = config if config is not None else Config()
        self.seed_candidate = seed_candidate
        self.initial_reflection = initial_reflection
        self.result = Result()
        self.population: list[Individual] = []
        self.seed_individual: Individual | None = None
        self.long_term = initial_reflection
        self.rng = random.Random(self.config.seed)

    @property
    def seed_text(self) -> str:
        return self.seed_candidate or ""

    @prompt(template="initial.j2")
    async def initial(self, *, generated: str) -> str:
        """Propose one initial candidate using the optional seed and guidance."""
        return generated

    @prompt(template="crossover.j2")
    async def crossover(
        self, worse: Individual, better: Individual, reflection: str, *, generated: str
    ) -> str:
        """Combine an ordered worse/better pair using its short reflection."""
        return generated

    @prompt(template="mutate.j2")
    async def mutate(self, elite: Individual, reflection: str, *, generated: str) -> str:
        """Revise the elite using accumulated reflection."""
        return generated

    @prompt(template="reflect_pair.j2")
    async def reflect_pair(self, worse: Individual, better: Individual, *, generated: str) -> str:
        """Compare an ordered worse/better pair."""
        return generated

    @prompt(template="reflect_pair_black_box.j2")
    async def reflect_pair_black_box(
        self, worse: Individual, better: Individual, *, generated: str
    ) -> str:
        """Infer hidden objective structure from relative candidate performance."""
        return generated

    @prompt(template="reflect_long.j2")
    async def reflect_long(self, prior: str, insights: list[str], *, generated: str) -> str:
        """Consolidate short reflections into bounded carried memory."""
        return generated

    def quality(self, individual: Individual) -> float:
        if individual.score is None:
            return math.inf
        return -individual.score if self.config.maximize else individual.score

    @property
    def remaining(self) -> int:
        return self.config.max_evaluations - len(self.result.individuals)

    async def score(self, response, stage, generation, parents=()):
        if self.remaining <= 0:
            raise RuntimeError("evaluation budget exhausted")
        individual = Individual(
            len(self.result.individuals),
            response,
            stage,
            generation,
            list(parents),
        )
        self.result.individuals.append(individual)
        try:
            if not response.strip():
                raise ValueError("candidate must be nonblank text")
            value = float(await self.evaluate(individual.candidate))
            if not math.isfinite(value):
                raise ValueError("score must be finite")
            individual.score = value
        except ValueError as exc:
            individual.error = f"{type(exc).__name__}: {exc}"
        if individual.score is not None and (
            self.result.best is None or self.quality(individual) < self.quality(self.result.best)
        ):
            self.result.best = individual
        self.result.best_history.append(
            self.result.best.score if self.result.best is not None else None
        )
        return individual

    async def seed(self) -> None:
        if self.seed_candidate is not None and self.remaining > 0:
            self.seed_individual = await self.score(self.seed_candidate, "seed", 0)
            if self.seed_individual.error:
                self.result.stop_reason = "invalid_seed"

    async def initialize(self, execution: dict) -> None:
        for _ in range(min(self.config.initial_size, self.remaining)):
            response = await self.initial(**execution)
            self.population.append(await self.score(response, "initial", 0))

    def select_parents(self) -> list[list[Individual]]:
        valid = [item for item in self.population if item.score is not None]
        elite = self.result.best
        if elite is not None and elite not in valid:
            valid.append(elite)
        # Official black-box selection excludes candidates that do not beat the seed.
        if self.config.black_box and self.seed_individual is not None:
            valid = [
                item for item in valid if self.quality(item) < self.quality(self.seed_individual)
            ]
        if not valid:
            self.result.stop_reason = "no_valid_individuals"
            return []
        count = min(int(self.config.population_size * self.config.crossover_rate), self.remaining)
        # ponytail: quadratic pairs fit small populations; sample fitness groups at thousands.
        pairs = [(a, b) for a, b in combinations(valid, 2) if a.score != b.score]
        if count and not pairs:
            self.result.stop_reason = "no_distinct_parents"
            return []
        return [
            sorted(self.rng.choice(pairs), key=self.quality, reverse=True) for _ in range(count)
        ]

    async def reflect(self, selected: list[list[Individual]], execution: dict) -> list[str]:
        insights = []
        reflect_pair = self.reflect_pair_black_box if self.config.black_box else self.reflect_pair
        for worse, better in selected:
            insight = (
                await reflect_pair(worse, better, **execution)
                if self.config.short_reflection
                else ""
            )
            insights.append(insight)
        return insights

    async def cross_population(
        self,
        selected: list[list[Individual]],
        insights: list[str],
        generation: int,
        execution: dict,
    ) -> list[Individual]:
        offspring = []
        for (worse, better), insight in zip(selected, insights):
            response = await self.crossover(worse, better, insight, **execution)
            offspring.append(
                await self.score(response, "crossover", generation, (worse.id, better.id))
            )
        return offspring

    async def update_reflections(
        self,
        selected: list[list[Individual]],
        insights: list[str],
        generation: int,
        execution: dict,
    ) -> None:
        raw_long_term = None
        if self.config.long_reflection and any(insights):
            raw_long_term = await self.reflect_long(self.long_term, insights, **execution)
            self.long_term = " ".join(raw_long_term.split()[:49])
        self.result.reflections.append(
            {
                "generation": generation,
                "pairs": [[a.id, b.id] for a, b in selected],
                "short_term": insights,
                "long_term": self.long_term,
                "raw_long_term": raw_long_term,
            }
        )

    async def mutate_population(self, generation: int, execution: dict) -> list[Individual]:
        # Snapshot once, including current crossover offspring; mutations share this elite.
        elite = self.result.best
        count = min(int(self.config.population_size * self.config.mutation_rate), self.remaining)
        offspring = []
        for _ in range(count):
            response = await self.mutate(elite, self.long_term, **execution)
            offspring.append(await self.score(response, "mutation", generation, (elite.id,)))
        return offspring

    async def run(self, *, session: Session | None = None) -> Result:
        """Run seed evaluation, initialization, and reflective generations on a fresh instance."""
        execution = {"session": session} if session is not None else {"provider": self.provider}
        initialization = execution if session is not None else {"provider": self.initial_provider}
        reflection = execution if session is not None else {"provider": self.reflector_provider}
        await self.seed()
        if self.result.stop_reason:
            return self.result
        await self.initialize(initialization)
        generation = 0
        while self.remaining > 0:
            generation += 1
            selected = self.select_parents()
            if self.result.stop_reason:
                break
            insights = await self.reflect(selected, reflection)
            offspring = await self.cross_population(selected, insights, generation, execution)
            await self.update_reflections(selected, insights, generation, reflection)
            offspring.extend(await self.mutate_population(generation, execution))
            if not offspring:
                self.result.stop_reason = "no_offspring"
                break
            self.population = offspring
        self.result.stop_reason = self.result.stop_reason or "budget"
        return self.result
