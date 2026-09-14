"""Optimize bounded real vectors with MOEA/D and the paper's LLM or linear operator."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from slick import prompt
from slick.providers import Provider

Vector = tuple[float, ...]


class CandidateRejected(ValueError):
    """An evaluator cannot score this candidate, or its measured objectives are invalid."""


@dataclass(frozen=True)
class Individual:
    x: Vector
    objectives: Vector


@dataclass(frozen=True)
class Result:
    population: tuple[Individual, ...]
    archive: tuple[Individual, ...]
    ideal: Vector
    evaluations: int
    model_calls: int


def das_dennis(objectives: int, partitions: int) -> tuple[Vector, ...]:
    """Return all simplex lattice directions; count = C(H + M - 1, M - 1)."""
    total = partitions + objectives - 1
    return tuple(
        tuple((b - a - 1) / partitions for a, b in zip((-1, *cuts), (*cuts, total)))
        for cuts in combinations(range(total), objectives - 1)
    )


def rank_weights(count: int, normalization: Literal["official", "paper"] = "official") -> Vector:
    """Weights for worst-to-best parents, before adding unnormalized Gaussian noise.

    OperatorLO.m divides polynomial values by their sum. Equation (7) uses softmax.
    """
    ranks = ((count - i) / count for i in range(count))
    values = [(((-0.111 * r + 1.037) * r - 1.291) * r + 0.445) for r in ranks]
    if normalization == "paper":
        largest = max(values)
        values = [math.exp(v - largest) for v in values]
    total = sum(values)
    return tuple(v / total for v in values)


def _scalar(objectives: Vector, weights: Vector, ideal: Vector) -> float:
    return max(w * (f - z) for f, w, z in zip(objectives, weights, ideal))


def _dominates(a: Individual, b: Individual) -> bool:
    return all(x <= y for x, y in zip(a.objectives, b.objectives)) and any(
        x < y for x, y in zip(a.objectives, b.objectives)
    )


class MOEAD:
    """Own the search; callers supply bounds, minimization objectives, and evaluation.

    The evaluator receives an immutable vector in original units. It owns decoding,
    constraints, execution isolation, and deadlines. One active run per instance;
    each run resets state. LLM calls use a stateless provider without Session history.
    Configure slick.prompts.TEMPLATE_ROOT once at application startup for LLM use.
    """

    def __init__(
        self,
        task: str,
        bounds: Sequence[tuple[float, float]],
        objectives: int,
        evaluate: Callable[[Vector], Awaitable[Sequence[float]]],
        provider: Provider | None = None,
    ):
        self.task = task
        self.bounds = tuple(bounds)
        self.objectives = objectives
        self.evaluate = evaluate
        self.provider = provider
        self.population: list[Individual] = []
        self.archive: list[Individual] = []
        self.attempts: list[dict] = []
        self.measurements: list[dict] = []
        self.evaluations = 0

    @prompt(template="propose.j2")
    async def propose(self, samples: list[tuple[Vector, float]], count: int) -> str:
        """Request tagged vectors in normalized coordinates, preserving raw response text."""
        ...

    async def run(
        self,
        *,
        operator: Literal["lo", "llm"] = "lo",
        max_evaluations: int = 1000,
        population_size: int = 50,
        neighborhood_size: int = 10,
        parents: int = 10,
        offspring: int | None = None,
        neighbor_probability: float = 0.9,
        noise: float = 0.5,
        dimension_probability: float = 0.1,
        normalization: Literal["official", "paper"] = "official",
        mutation_probability: float = 1.0,
        mutation_eta: float = 20.0,
        max_attempts: int = 3,
        weights: Sequence[Sequence[float]] | None = None,
        seed: int | None = None,
    ) -> Result:
        """Run to an exact evaluator-call budget, including initialization and rejections.

        Defaults generate one LO or two LLM offspring per subproblem. Without supplied
        weights, use the largest complete Das–Dennis lattice within population_size.
        Mutation selects each coordinate with probability 1/d, then gates the whole
        offspring with mutation_probability. Invalid model responses retry up to
        max_attempts per subproblem; exhaustion raises with all records retained.
        """
        self.rng = random.Random(seed)
        self.population, self.archive, self.attempts, self.measurements = [], [], [], []
        self.evaluations = 0
        self.ideal = (math.inf,) * self.objectives
        count = (2 if operator == "llm" else 1) if offspring is None else offspring
        await self._initialize(population_size, neighborhood_size, weights, max_evaluations)
        while self.evaluations < max_evaluations:
            order = list(range(len(self.population)))
            if operator == "llm":
                self.rng.shuffle(order)
            for index in order:
                if self.evaluations >= max_evaluations:
                    break
                selected, pool = self._select(index, parents, neighbor_probability)
                remaining = min(count, max_evaluations - self.evaluations)
                if operator == "llm":
                    children = await self._generate(index, selected, remaining, max_attempts)
                    replacement = self.neighbors[index]
                else:
                    children = [
                        self._linear(
                            selected,
                            self.population[index],
                            noise,
                            dimension_probability,
                            normalization,
                        )
                        for _ in range(remaining)
                    ]
                    replacement = pool
                for x in children:
                    child = await self._measure(self._mutate(x, mutation_probability, mutation_eta))
                    if child is not None:
                        self._update(child, replacement, strict=operator == "llm")
        return Result(
            tuple(self.population),
            tuple(self.archive),
            self.ideal,
            self.evaluations,
            len(self.attempts),
        )

    async def _initialize(self, size, neighborhood_size, weights, budget):
        if weights is None:
            partitions = 1
            if self.objectives > 1:
                while math.comb(partitions + self.objectives, self.objectives - 1) <= size:
                    partitions += 1
            self.weights = das_dennis(self.objectives, partitions)
        else:
            self.weights = tuple(tuple(w) for w in weights)
        self.neighbors = [
            sorted(range(len(self.weights)), key=lambda j: math.dist(w, self.weights[j]))[
                :neighborhood_size
            ]
            for w in self.weights
        ]
        for _ in self.weights:
            if self.evaluations >= budget:
                raise RuntimeError("Evaluation budget exhausted before population initialization")
            individual = await self._measure(tuple(self.rng.uniform(a, b) for a, b in self.bounds))
            if individual is None:
                raise RuntimeError("Initial population evaluation failed; see measurements")
            self.population.append(individual)
            self._update(individual, [], strict=False)

    def _select(self, index, count, probability):
        pool = list(
            self.neighbors[index]
            if self.rng.random() < probability
            else range(len(self.population))
        )
        self.rng.shuffle(pool)
        # Indexing preserves sampling without replacement; callers provide a large enough pool.
        selected = [self.population[pool[j]] for j in range(count)]
        selected.sort(
            key=lambda p: _scalar(p.objectives, self.weights[index], self.ideal), reverse=True
        )
        return selected, pool

    async def _generate(self, index, selected, count, max_attempts):
        samples = [
            (
                tuple((v - a) / (b - a) if a != b else 0.0 for v, (a, b) in zip(p.x, self.bounds)),
                _scalar(p.objectives, self.weights[index], self.ideal),
            )
            for p in selected
        ]
        for _ in range(max_attempts):
            record = {"subproblem": index, "samples": samples, "status": "pending"}
            self.attempts.append(record)
            try:
                raw = await self.propose(samples, count, provider=self.provider)
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise
            record["response"] = raw
            children, rejected = [], []
            for block in re.findall(r"<start>(.*?)<end>", raw, flags=re.DOTALL):
                try:
                    values = tuple(float(v.strip()) for v in block.split(","))
                    if len(values) != len(self.bounds) or not all(map(math.isfinite, values)):
                        raise ValueError("expected one finite value per coordinate")
                except ValueError as exc:
                    rejected.append(str(exc))
                    continue
                children.append(
                    tuple(
                        a + min(1.0, max(0.0, v)) * (b - a)
                        for v, (a, b) in zip(values, self.bounds)
                    )
                )
                if len(children) == count:
                    break
            record.update(offspring=tuple(children), rejected=rejected)
            if children:
                record["status"] = "accepted"
                return children
            record.update(status="rejected", error="No usable tagged points")
        raise RuntimeError(f"No usable LLM offspring after {max_attempts} attempts")

    def _linear(self, selected, current, noise, probability, normalization):
        # OperatorLO.m: noise is per parent, shared across dimensions, not renormalized.
        weights = [
            w + noise * self.rng.gauss(0, 1) for w in rank_weights(len(selected), normalization)
        ]
        return tuple(
            sum(w * p.x[j] for w, p in zip(weights, selected))
            if self.rng.random() < probability
            else current.x[j]
            for j in range(len(self.bounds))
        )

    def _mutate(self, x, probability, eta):
        child = [min(b, max(a, v)) for v, (a, b) in zip(x, self.bounds)]
        if self.rng.random() >= probability:
            return tuple(child)
        for j, (a, b) in enumerate(self.bounds):
            if self.rng.random() >= 1 / len(child) or a == b:
                continue
            value, u = child[j], self.rng.random()
            if u <= 0.5:
                delta = 2 * u + (1 - 2 * u) * (1 - (value - a) / (b - a)) ** (eta + 1)
                delta = delta ** (1 / (eta + 1)) - 1
            else:
                delta = 2 * (1 - u) + 2 * (u - 0.5) * (1 - (b - value) / (b - a)) ** (eta + 1)
                delta = 1 - delta ** (1 / (eta + 1))
            child[j] = min(b, max(a, value + (b - a) * delta))
        return tuple(child)

    async def _measure(self, x):
        self.evaluations += 1
        record = {"evaluation": self.evaluations, "x": x, "status": "pending"}
        self.measurements.append(record)
        try:
            values = tuple(float(v) for v in await self.evaluate(x))
            record["objectives"] = values
            if len(values) != self.objectives or not all(map(math.isfinite, values)):
                raise CandidateRejected("expected one finite measurement per objective")
        except (CandidateRejected, TimeoutError) as exc:
            record.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
            return None
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        individual = Individual(x, values)
        record.update(status="evaluated", individual=individual)
        return individual

    def _update(self, child, pool, *, strict):
        self.ideal = tuple(min(z, f) for z, f in zip(self.ideal, child.objectives))
        replacements = 0
        for index in pool:
            old = _scalar(self.population[index].objectives, self.weights[index], self.ideal)
            new = _scalar(child.objectives, self.weights[index], self.ideal)
            if new < old or (not strict and new == old):
                self.population[index] = child
                replacements += 1
                if replacements == 2:
                    break
        # ponytail: linear archive scan; use indexed dominance if archive size dominates runtime.
        if child not in self.archive and not any(_dominates(p, child) for p in self.archive):
            self.archive = [p for p in self.archive if not _dominates(child, p)]
            self.archive.append(child)
