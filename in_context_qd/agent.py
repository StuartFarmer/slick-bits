"""Generate quality-diverse candidates from sampled archive elites using Slick."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

from slick import Provider, prompt

Vector = tuple[float, ...]
Cell = tuple[int, ...]


class CandidateRejected(ValueError):
    """A candidate cannot be evaluated or has invalid measured fitness/features."""


@dataclass(frozen=True)
class Evaluation:
    fitness: float
    features: Vector


@dataclass(frozen=True)
class Elite:
    candidate: str
    fitness: float
    features: Vector


@dataclass(frozen=True)
class Grid:
    """Rectangular feature niches; upper endpoints belong to the final cell."""

    bounds: tuple[tuple[float, float], ...]
    bins: tuple[int, ...]
    precision: int = 1000

    def cells(self) -> tuple[Cell, ...]:
        return tuple(product(*(range(n) for n in self.bins)))

    def locate(self, features: Vector) -> Cell:
        if len(features) != len(self.bounds) or any(
            not math.isfinite(v) or not a <= v <= b for v, (a, b) in zip(features, self.bounds)
        ):
            raise CandidateRejected("Expected finite, in-bounds features of the grid dimension")
        return tuple(
            min(n - 1, int((v - a) / (b - a) * n))
            for v, (a, b), n in zip(features, self.bounds, self.bins)
        )

    def centroid(self, cell: Cell) -> Vector:
        return tuple(
            a + (i + 0.5) * (b - a) / n for i, (a, b), n in zip(cell, self.bounds, self.bins)
        )

    def encode(self, features: Vector) -> str:
        return ", ".join(
            str(round((v - a) / (b - a) * self.precision))
            for v, (a, b) in zip(features, self.bounds)
        )


@dataclass(frozen=True)
class NumericSpace:
    """Optional bounded-vector codec: integer CSV to/from original parameter units."""

    bounds: tuple[tuple[float, float], ...]
    precision: int = 1000

    def encode(self, values: Vector) -> str:
        return ", ".join(
            str(round((v - a) / (b - a) * self.precision)) if a != b else "0"
            for v, (a, b) in zip(values, self.bounds)
        )

    def decode(self, candidate: str) -> Vector:
        try:
            integers = tuple(int(v) for v in candidate.split(","))
        except ValueError as exc:
            raise CandidateRejected("Expected comma-separated integer parameters") from exc
        if len(integers) != len(self.bounds) or any(not 0 <= v <= self.precision for v in integers):
            raise CandidateRejected("Wrong parameter dimension or encoded parameter out of bounds")
        return tuple(a + v / self.precision * (b - a) for v, (a, b) in zip(integers, self.bounds))

    def sample(self, rng: random.Random) -> str:
        return self.encode(tuple(rng.uniform(a, b) for a, b in self.bounds))


@dataclass(frozen=True)
class Config:
    batch_size: int = 10
    context_size: int = 30
    template: Literal["qd", "fitness", "feature", "lmx"] = "qd"
    order: Literal["distance", "fitness", "random"] = "distance"
    feature_query: Literal["empty", "uniform"] = "empty"
    improvement: float = 0.2
    minimum_improvement: float = 1e-6
    initial_fitness: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class Query:
    cell: Cell
    features: Vector
    fitness: float
    context: tuple[Elite, ...]


@dataclass
class Attempt:
    generation: int
    query: Query | None = None
    raw: str | None = None
    evaluation: Evaluation | None = None
    cell: Cell | None = None
    accepted: bool = False
    error: str = ""


@dataclass(frozen=True)
class Metrics:
    generation: int
    occupied: int
    coverage: float
    qd_score: float
    max_fitness: float | None


@dataclass
class Result:
    archive: dict[Cell, Elite]
    evaluations: int
    model_calls: int
    history: list[Metrics] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.history[-1].coverage

    @property
    def qd_score(self) -> float:
        return self.history[-1].qd_score

    @property
    def max_fitness(self) -> float | None:
        return self.history[-1].max_fitness


class InContextQD:
    """Maximize caller-measured fitness across a caller-defined feature grid.

    Supply initial candidates sampled from your domain and an async evaluator.
    Candidates stay text; the evaluator owns decoding, constraints, isolation,
    and deadlines. Each run resets state. Provider calls are independent, without
    Session history or internal retries. Configure Slick's template root at startup.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        grid: Grid,
        *,
        config: Config = Config(),
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.grid = grid
        self.config = config
        self.archive: dict[Cell, Elite] = {}
        self.attempts: list[Attempt] = []
        self.history: list[Metrics] = []
        self.evaluations = 0
        self.model_calls = 0
        self.rng = random.Random(config.seed)
        self.cells = grid.cells()

    @prompt(template="qd.j2")
    async def generate_qd(self, query: Query, *, generated: str) -> str:
        """Complete fitness: features: candidate examples."""
        return generated

    @prompt(template="fitness.j2")
    async def generate_fitness(self, query: Query, *, generated: str) -> str:
        """Complete fitness: candidate examples."""
        return generated

    @prompt(template="feature.j2")
    async def generate_feature(self, query: Query, *, generated: str) -> str:
        """Complete features: candidate examples."""
        return generated

    @prompt(template="lmx.j2")
    async def generate_lmx(self, query: Query, *, generated: str) -> str:
        """Complete candidate-only examples without a fitness or feature query."""
        return generated

    async def run(self, initial_candidates: Sequence[str], *, generations: int = 1) -> Result:
        """Evaluate seeds, then consume exactly batch_size model calls per generation.

        Rejected candidates consume their attempt; there are no repair calls or
        random fallbacks. Empty initialization aborts when search is requested.
        Provider and unexpected evaluator errors abort with records retained.
        """
        self.rng = random.Random(self.config.seed)
        self.archive, self.attempts, self.history = {}, [], []
        self.evaluations = self.model_calls = 0
        await self.initialize(initial_candidates)
        if generations and not self.archive:
            raise RuntimeError("Initialization left the archive empty; inspect attempts")
        for generation in range(1, generations + 1):
            queries = self.build_queries()
            await self.generate_batch(queries, generation)
            self.record_metrics(generation)
        return Result(
            dict(self.archive),
            self.evaluations,
            self.model_calls,
            list(self.history),
            list(self.attempts),
        )

    async def initialize(self, candidates: Sequence[str]) -> None:
        for candidate in candidates:
            attempt = Attempt(0, raw=candidate)
            self.attempts.append(attempt)
            await self.assess(attempt)
        self.record_metrics(0)

    def build_queries(self) -> list[Query]:
        """Sample and order contexts against one immutable generation snapshot."""
        config = self.config
        empty = [cell for cell in self.cells if cell not in self.archive]
        if config.feature_query == "empty" and len(empty) >= config.batch_size:
            targets = self.rng.sample(empty, config.batch_size)
        else:
            targets = self.rng.choices(self.cells, k=config.batch_size)
        snapshot = tuple(self.archive.values())
        queries = []
        for cell in targets:
            features = self.grid.centroid(cell)
            context = self.rng.sample(snapshot, min(config.context_size, len(snapshot)))
            if config.order == "distance":
                context.sort(key=lambda p: math.dist(p.features, features), reverse=True)
            elif config.order == "fitness":
                context.sort(key=lambda p: p.fitness)
            # random.sample already gives a random ordering.
            fitness = config.initial_fitness
            if context:
                best = max(p.fitness for p in context)
                fitness = best + max(abs(best) * config.improvement, config.minimum_improvement)
            queries.append(Query(cell, features, fitness, tuple(context)))
        return queries

    async def generate_batch(self, queries: list[Query], generation: int) -> None:
        operation = {
            "qd": self.generate_qd,
            "fitness": self.generate_fitness,
            "feature": self.generate_feature,
            "lmx": self.generate_lmx,
        }[self.config.template]
        for query in queries:
            attempt = Attempt(generation, query)
            self.attempts.append(attempt)
            self.model_calls += 1
            try:
                attempt.raw = await operation(query, provider=self.provider)
            except Exception as exc:
                attempt.error = f"{type(exc).__name__}: {exc}"
                raise
            await self.assess(attempt)

    async def assess(self, attempt: Attempt) -> None:
        """Evaluate once, then compete in the measured cell, never the requested cell."""
        try:
            if not attempt.raw.strip():
                raise CandidateRejected("Candidate must not be blank")
            self.evaluations += 1
            evaluation = await self.evaluate(attempt.raw)
            attempt.evaluation = evaluation
            if not math.isfinite(evaluation.fitness):
                raise CandidateRejected("Measured fitness must be finite")
            cell = self.grid.locate(evaluation.features)
            attempt.cell = cell
            incumbent = self.archive.get(cell)
            if incumbent is None or evaluation.fitness > incumbent.fitness:
                self.archive[cell] = Elite(
                    attempt.raw, evaluation.fitness, tuple(evaluation.features)
                )
                attempt.accepted = True
        except CandidateRejected as exc:
            attempt.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            attempt.error = f"{type(exc).__name__}: {exc}"
            raise

    def record_metrics(self, generation: int) -> None:
        fitness = [p.fitness for p in self.archive.values()]
        self.history.append(
            Metrics(
                generation,
                len(fitness),
                len(fitness) / len(self.cells),
                math.fsum(fitness),
                max(fitness, default=None),
            )
        )
