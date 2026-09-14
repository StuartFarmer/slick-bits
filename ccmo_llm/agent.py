"""Coevolve constrained and unconstrained populations using GA and LLM offspring."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from slick import prompt
from slick.providers import Provider

Vector = tuple[float, ...]


class CandidateRejected(ValueError):
    """A candidate cannot be measured; ordinary constraint violations are valid data."""


@dataclass(frozen=True)
class Evaluation:
    objectives: Vector
    inequalities: Vector = ()
    equalities: Vector = ()


@dataclass(frozen=True)
class Individual:
    x: Vector
    objectives: Vector
    cv: float


@dataclass(frozen=True)
class Result:
    population: tuple[Individual, ...]
    auxiliary: tuple[Individual, ...]
    front: tuple[Individual, ...]
    evaluations: int
    model_calls: int


def fitness(population: Sequence[Individual], constrained: bool) -> list[float]:
    """PlatEMO CalFitness: dominance strength, raw fitness, and kth-neighbor density."""
    size = len(population)
    dominates = [[False] * size for _ in population]
    for i, a in enumerate(population):
        for j in range(i + 1, size):
            b = population[j]
            if constrained and a.cv != b.cv:
                dominates[i][j], dominates[j][i] = a.cv < b.cv, b.cv < a.cv
            else:
                less = any(x < y for x, y in zip(a.objectives, b.objectives))
                more = any(x > y for x, y in zip(a.objectives, b.objectives))
                dominates[i][j], dominates[j][i] = less and not more, more and not less
    strength = [sum(row) for row in dominates]
    scores = []
    for i, a in enumerate(population):
        raw = sum(strength[j] for j in range(size) if dominates[j][i])
        distances = sorted(
            math.inf if i == j else math.dist(a.objectives, b.objectives)
            for j, b in enumerate(population)
        )
        scores.append(raw + 1 / (distances[math.isqrt(size) - 1] + 2))
    return scores


def environmental_selection(
    population: list[Individual], size: int, constrained: bool
) -> tuple[list[Individual], list[float]]:
    """SPEA2 survival; retain union fitness for the next mating tournament."""
    scores = fitness(population, constrained)
    selected = [i for i, score in enumerate(scores) if score < 1]
    if len(selected) < size:
        selected = sorted(range(len(population)), key=scores.__getitem__)[:size]
    elif len(selected) > size:
        distances = {
            (i, j): math.inf
            if i == j
            else math.dist(population[i].objectives, population[j].objectives)
            for i in selected
            for j in selected
        }
        # ponytail: repeated distance sorting suits N=100; index neighbors for large populations.
        while len(selected) > size:
            crowded = min(selected, key=lambda i: sorted(distances[i, j] for j in selected))
            selected.remove(crowded)
    selected.sort(key=lambda i: (scores[i], i))
    return [population[i] for i in selected], [scores[i] for i in selected]


class CCMOLLM:
    """Own CCMO search over real vectors; callers own evaluation and provider resources.

    All objectives are minimized, inequalities satisfy g(x)<=0, and equalities
    satisfy abs(h(x))<=equality_tolerance. The evaluator owns decoding, isolation,
    and deadlines. Infeasible points remain usable search data. Configure Slick's
    process-global template root once at application startup, outside this class.
    """

    def __init__(
        self,
        task: str,
        bounds: Sequence[tuple[float, float]],
        objectives: int,
        evaluate: Callable[[Vector], Awaitable[Evaluation]],
        provider: Provider | None = None,
    ):
        self.task = task
        self.bounds = tuple(bounds)
        self.objectives = objectives
        self.evaluate = evaluate
        self.provider = provider
        self.attempts: list[dict] = []
        self.measurements: list[dict] = []
        self.generations: list[dict] = []
        self.evaluations = 0

    @prompt(template="propose.j2")
    async def propose(self, feasible: str, infeasible: str, count: int) -> str:
        """Return the paper's tagged text unchanged so rejected responses can be logged."""
        ...

    async def run(
        self,
        *,
        population_size: int = 100,
        max_evaluations: int = 10_000,
        llm_fraction: float = 0.1,
        equality_tolerance: float = 1e-4,
        max_attempts: int = 3,
        seed: int | None = None,
    ) -> Result:
        """Reset and run to an exact evaluator-call budget, including both initial populations.

        One run at a time per instance. Each population contributes half the offspring;
        round its LLM allocation to the nearest integer (half up). Use N divisible by
        20 for the paper's exact split. Zero llm_fraction runs the CCMO baseline.
        Invalid model batches retry at most max_attempts, then raise without GA fallback.
        Provider and unexpected evaluator errors propagate with their records retained.
        """
        self.rng = random.Random(seed)
        self.attempts, self.measurements, self.generations = [], [], []
        self.evaluations = 0
        self.equality_tolerance = equality_tolerance
        await self._initialize(population_size, max_evaluations)
        while self.evaluations < max_evaluations:
            count = min(population_size, max_evaluations - self.evaluations)
            candidates = await self._reproduce(count, llm_fraction, max_attempts)
            offspring = []
            for x, source in candidates:
                child = await self._measure(x, source)
                if child is not None:
                    offspring.append(child)
            self._select_survivors(offspring, population_size)
        front_scores = fitness(self.population, True)
        front = tuple(
            dict.fromkeys(p for p, f in zip(self.population, front_scores) if p.cv == 0 and f < 1)
        )
        return Result(
            tuple(self.population),
            tuple(self.auxiliary),
            front,
            self.evaluations,
            len(self.attempts),
        )

    async def _initialize(self, size, budget):
        populations = [[], []]
        for population in populations:
            for _ in range(size):
                if self.evaluations >= budget:
                    raise RuntimeError("Evaluation budget exhausted during initialization")
                x = tuple(self.rng.uniform(a, b) for a, b in self.bounds)
                individual = await self._measure(x, "initial")
                if individual is None:
                    raise RuntimeError("Cannot measure an initial solution; see measurements")
                population.append(individual)
        self.population, self.auxiliary = populations
        self.scores = [fitness(self.population, True), fitness(self.auxiliary, False)]

    async def _reproduce(self, count, fraction, max_attempts):
        candidates = []
        known = {p.x for p in self.population + self.auxiliary}
        for index, (population, quota) in enumerate(
            zip((self.population, self.auxiliary), ((count + 1) // 2, count // 2))
        ):
            llm_count = int(quota * fraction + 0.5)
            children = self._ga_offspring(population, self.scores[index], quota - llm_count)
            candidates.extend((x, "ga") for x in children)
            known.update(children)
            if llm_count:
                samples = self.rng.sample(population, 2 * llm_count)
                children = await self._llm_offspring(samples, llm_count, index, known, max_attempts)
                candidates.extend((x, "llm") for x in children)
                known.update(children)
        return candidates

    async def _llm_offspring(self, samples, count, population_index, known, max_attempts):
        groups = [[], []]
        for p in samples:
            groups[p.cv > 0].append(
                f"solution: <start>{','.join(map(str, p.x))}<end> "
                f"function value: {p.objectives} constraint violation degree: {p.cv}"
            )
        feasible = "\n".join(groups[0]) or "no feasible solution"
        infeasible = "\n".join(groups[1]) or "no infeasible solution"
        for _ in range(max_attempts):
            record = {
                "population": population_index,
                "samples": tuple(samples),
                "status": "pending",
            }
            self.attempts.append(record)
            try:
                raw = await self.propose(feasible, infeasible, count, provider=self.provider)
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise
            record["response"] = raw
            try:
                blocks = re.findall(r"<start>(.*?)<end>", raw, flags=re.DOTALL)
                if len(blocks) != count:
                    raise ValueError(f"expected exactly {count} tagged solutions")
                children = [tuple(float(v.strip()) for v in block.split(",")) for block in blocks]
                for x in children:
                    if len(x) != len(self.bounds) or not all(map(math.isfinite, x)):
                        raise ValueError("expected one finite value per decision coordinate")
                    if any(v < a or v > b for v, (a, b) in zip(x, self.bounds)):
                        raise ValueError("generated solution is outside the decision bounds")
                if len(set(children)) != count or any(
                    x in known or x in {p.x for p in samples} for x in children
                ):
                    raise ValueError("generated solutions must be distinct from existing solutions")
            except ValueError as exc:
                record.update(status="rejected", error=str(exc))
                continue
            record.update(status="accepted", offspring=tuple(children))
            return children
        raise RuntimeError(f"No valid LLM batch after {max_attempts} attempts; see attempts")

    def _ga_offspring(self, population, scores, count):
        mating = [
            min((self.rng.randrange(len(population)) for _ in range(2)), key=scores.__getitem__)
            for _ in range(2 * count)
        ]
        children = []
        for first, second in zip(mating[:count], mating[count:]):
            child = []
            for a, b in zip(population[first].x, population[second].x):
                u = self.rng.random()
                beta = (2 * u) ** (1 / 21) if u <= 0.5 else (2 - 2 * u) ** (-1 / 21)
                beta *= (-1) ** self.rng.randrange(2)
                if self.rng.random() < 0.5:
                    beta = 1
                child.append((a + b) / 2 + beta * (a - b) / 2)
            children.append(self._mutate(tuple(child)))
        return children

    def _mutate(self, x):
        child = [min(b, max(a, v)) for v, (a, b) in zip(x, self.bounds)]
        for j, (a, b) in enumerate(self.bounds):
            if self.rng.random() >= 1 / len(child) or a == b:
                continue
            u = self.rng.random()
            value = child[j]
            # OperatorGAhalf.m: polynomial mutation, distribution index 20.
            if u <= 0.5:
                base = 2 * u + (1 - 2 * u) * (1 - (value - a) / (b - a)) ** 21
                delta = base ** (1 / 21) - 1
            else:
                base = 2 * (1 - u) + 2 * (u - 0.5) * (1 - (b - value) / (b - a)) ** 21
                delta = 1 - base ** (1 / 21)
            child[j] = min(b, max(a, value + (b - a) * delta))
        return tuple(child)

    async def _measure(self, x, source):
        self.evaluations += 1
        record = {"evaluation": self.evaluations, "x": x, "source": source, "status": "pending"}
        self.measurements.append(record)
        try:
            measurement = await self.evaluate(x)
            record["measurement"] = measurement
            objectives = tuple(measurement.objectives)
            residuals = (*measurement.inequalities, *measurement.equalities)
            if len(objectives) != self.objectives or not all(
                map(math.isfinite, (*objectives, *residuals))
            ):
                raise CandidateRejected("invalid measured objectives or constraint residuals")
            cv = sum(max(0, g) for g in measurement.inequalities) + sum(
                max(0, abs(h) - self.equality_tolerance) for h in measurement.equalities
            )
            if not math.isfinite(cv):
                raise CandidateRejected("nonfinite total constraint violation")
        except (CandidateRejected, TimeoutError) as exc:
            record.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
            return None
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        individual = Individual(x, objectives, cv)
        record.update(status="evaluated", individual=individual)
        return individual

    def _select_survivors(self, offspring, size):
        self.population, primary_scores = environmental_selection(
            self.population + offspring, size, True
        )
        self.auxiliary, auxiliary_scores = environmental_selection(
            self.auxiliary + offspring, size, False
        )
        self.scores = [primary_scores, auxiliary_scores]
        self.generations.append(
            {
                "evaluations": self.evaluations,
                "offspring": tuple(offspring),
                "population": tuple(self.population),
                "auxiliary": tuple(self.auxiliary),
            }
        )
