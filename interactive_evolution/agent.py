"""Evolve free-form text through human feedback and LLM genetic operators."""

import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Literal

from slick import prompt
from slick.providers import Provider

Rating = Literal[-1, 0, 1]


@dataclass(frozen=True)
class Evaluation:
    """One new anonymous vote; the collector owns participant deduplication."""

    individual_id: int
    rating: Rating


@dataclass(frozen=True)
class Individual:
    id: int
    content: str
    parents: tuple[int, ...] = ()
    mutation_topic: str | None = None
    ratings: tuple[Rating, ...] = ()

    @property
    def fitness(self) -> float:
        """Mean vote, maximized; an unrated individual has no selectable fitness."""
        return sum(self.ratings) / len(self.ratings) if self.ratings else -float("inf")


def _nonblank(text: str) -> str:
    if not text.strip():
        raise ValueError("generated candidate must not be blank")
    return text


class InteractiveEvolution:
    """Steady-state interactive evolution with independent, text-only model calls.

    The async evaluator receives the active population, publishes unseen IDs, and
    waits for a batch of new votes. Use one active run per instance. Each run
    resets the archive and records; generation and feedback errors propagate.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[tuple[Individual, ...]], Awaitable[Sequence[Evaluation]]],
        *,
        mutation_topics: Sequence[str],
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.mutation_topics = tuple(mutation_topics)
        self.archive: dict[int, Individual] = {}
        self.active_ids: list[int] = []
        self.published: set[int] = set()
        self.evaluations: list[Evaluation] = []
        self.attempts: list[dict] = []
        self.history: list[tuple[Individual, ...]] = []
        self.completed_iterations = 0

    @property
    def population(self) -> tuple[Individual, ...]:
        return tuple(self.archive[id_] for id_ in self.active_ids)

    @prompt(template="initialize.j2")
    async def initialize(self, *, generated: str) -> str:
        """Generate one candidate from the brief."""
        return _nonblank(generated)

    @prompt(template="crossover.j2")
    async def crossover(self, first: str, second: str, *, generated: str) -> str:
        """Recombine two candidates into one offspring."""
        return _nonblank(generated)

    @prompt(template="mutate.j2")
    async def mutate(self, candidate: str, topic: str, *, generated: str) -> str:
        """Vary one high-level characteristic of a candidate."""
        return _nonblank(generated)

    async def run(
        self,
        *,
        initial: Sequence[str] = (),
        population_size: int = 10,
        iterations: int = 30,
        new_evaluations: int = 25,
        min_evaluations: int = 1,
        crossover_probability: float = 0.7,
        seed: int | None = None,
    ) -> tuple[Individual, ...]:
        """Return the final evaluated population, ordered by descending mean vote.

        Supply at most population_size seeds, a population of at least two,
        positive feedback thresholds, and at least one mutation topic. The final
        evaluation needs per-individual coverage but no new activation quota.
        """
        self.archive.clear()
        self.active_ids.clear()
        self.published.clear()
        self.evaluations.clear()
        self.attempts.clear()
        self.history.clear()
        self.completed_iterations = 0
        self.rng = random.Random(seed)
        await self._initialize_population(initial, population_size)
        for _ in range(iterations):
            await self._collect_evaluations(new_evaluations, min_evaluations)
            self.history.append(self.population)
            parents = self._select_parents()
            offspring = await self._create_offspring(parents, crossover_probability)
            self._replace_worst(offspring)
            self.completed_iterations += 1
        await self._collect_evaluations(0, min_evaluations)
        self.history.append(self.population)
        return tuple(sorted(self.population, key=lambda p: p.fitness, reverse=True))

    async def _initialize_population(self, initial: Sequence[str], size: int) -> None:
        for content in initial:
            individual = Individual(len(self.archive), content)
            self.archive[individual.id] = individual
            self.active_ids.append(individual.id)
        while len(self.active_ids) < size:
            content = await self._generate("initialize")
            individual = Individual(len(self.archive), content)
            self.archive[individual.id] = individual
            self.active_ids.append(individual.id)

    async def _collect_evaluations(self, minimum_new: int, minimum_each: int) -> None:
        received = 0
        while received < minimum_new or any(len(p.ratings) < minimum_each for p in self.population):
            snapshot = self.population
            self.published.update(p.id for p in snapshot)
            batch = tuple(await self.evaluate(snapshot))
            if not batch:
                raise RuntimeError(
                    "feedback collector returned no votes; await new feedback or cancel"
                )
            self._record_evaluations(batch)
            received += len(batch)

    def _record_evaluations(self, batch: tuple[Evaluation, ...]) -> None:
        # Stage a complete batch so an unknown ID or rating cannot partially apply it.
        updated = {}
        for evaluation in batch:
            id_ = evaluation.individual_id
            if id_ not in self.published:
                raise KeyError(f"individual {id_} has not been offered for evaluation")
            rating = {-1: -1, 0: 0, 1: 1}[evaluation.rating]
            individual = updated.get(id_, self.archive[id_])
            updated[id_] = replace(individual, ratings=individual.ratings + (rating,))
        self.archive.update(updated)
        self.evaluations.extend(batch)

    def _select_parents(self) -> tuple[Individual, Individual]:
        # Independent size-two tournaments; the same winner may be selected twice.
        first = max(self.rng.sample(self.population, 2), key=lambda p: p.fitness)
        second = max(self.rng.sample(self.population, 2), key=lambda p: p.fitness)
        return first, second

    async def _create_offspring(
        self, parents: tuple[Individual, Individual], probability: float
    ) -> Individual:
        first, second = parents
        content = first.content
        parent_ids = (first.id,)
        if self.rng.random() < probability:
            content = await self._generate("crossover", first.content, second.content)
            parent_ids = (first.id, second.id)
        topic = self.rng.choice(self.mutation_topics)
        content = await self._generate("mutate", content, topic)
        return Individual(len(self.archive), content, parent_ids, topic)

    def _replace_worst(self, offspring: Individual) -> None:
        # Protect the unrated child until feedback arrives; ties evict the oldest incumbent.
        worst = min(self.population, key=lambda p: p.fitness)
        self.archive[offspring.id] = offspring
        self.active_ids.remove(worst.id)
        self.active_ids.append(offspring.id)

    async def _generate(self, operation: str, *args: str) -> str:
        record = {"operation": operation, "iteration": self.completed_iterations}
        self.attempts.append(record)

        async def recorded_call(context, **kwargs):
            response, requests = await self.provider.acall(context, **kwargs)
            record["raw_response"] = response
            return response, requests

        try:
            content = await getattr(self, operation)(
                *args, provider=SimpleNamespace(acall=recorded_call)
            )
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        record["status"] = "generated"
        return content
