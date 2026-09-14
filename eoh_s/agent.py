"""Evolve complementary heuristic sets using per-instance costs and Slick prompts."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from itertools import combinations
from statistics import fmean
from types import SimpleNamespace
from typing import Annotated

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider, ProviderError

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Proposal(BaseModel, extra="forbid", frozen=True):
    description: Text
    content: str


class Individual(Proposal):
    id: int
    scores: tuple[float, ...]

    @property
    def fitness(self) -> float:
        return fmean(self.scores)


class EvaluationError(ValueError):
    """An evaluator's explicit rejection of a candidate; consumes one evaluation."""


def cpi(population: Sequence[Individual]) -> float:
    """Mean of the best cost per instance; requires a nonempty evaluated set."""
    return fmean(min(scores) for scores in zip(*(p.scores for p in population), strict=True))


def complementary_select(candidates: Sequence[Individual], size: int) -> list[Individual]:
    """Greedy CPM: best mean first, then greatest reduction in per-instance cost."""
    remaining = sorted(candidates, key=lambda p: p.fitness)
    selected = [remaining.pop(0)]
    reference = selected[0].scores
    while len(selected) < size and remaining:
        index = min(
            range(len(remaining)),
            key=lambda i: sum(
                min(score - best, 0)
                for score, best in zip(remaining[i].scores, reference, strict=True)
            ),
        )
        chosen = remaining.pop(index)
        selected.append(chosen)
        reference = tuple(min(a, b) for a, b in zip(reference, chosen.scores, strict=True))
    return selected


class EoHS:
    """Own one search over arbitrary content; the caller evaluates and isolates it.

    Scores are lower-is-better costs on the same ordered instances on every call.
    Configure Slick's template root before running. Use a fresh instance per run.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Sequence[float]]],
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.attempts: list[dict] = []
        self.history: list[tuple[Individual, ...]] = []
        self.evaluations = 0
        self.instance_count: int | None = None

    @prompt(template="initialize.j2", output_type=Proposal)
    async def initialize(self, *, generated: Proposal) -> Proposal:
        """Generate an initial thought and artifact without parents."""
        return self._check_proposal(generated)

    @prompt(template="complementary_search.j2", output_type=Proposal)
    async def complementary_search(
        self, parents: tuple[Individual, Individual], *, generated: Proposal
    ) -> Proposal:
        """Explore a different design from the most distant pair."""
        return self._check_proposal(generated)

    @prompt(template="local_search.j2", output_type=Proposal)
    async def local_search(self, parent: Individual, *, generated: Proposal) -> Proposal:
        """Refine one rank-selected parent while preserving its strengths."""
        return self._check_proposal(generated)

    @staticmethod
    def _check_proposal(proposal: Proposal) -> Proposal:
        if not proposal.content.strip():
            raise ValueError("candidate content must not be blank")
        return proposal

    async def run(
        self,
        population_size: int = 10,
        max_evaluations: int = 2000,
        *,
        max_attempts: int | None = None,
        seed: int = 0,
    ) -> list[Individual]:
        """Initialize, sample N offspring from a snapshot, then apply greedy CPM.

        Attempt and evaluation caps include initialization. A partial final batch
        still competes. Failed offspring consume their slot; incumbents survive.
        """
        self.rng = random.Random(seed)
        self.max_evaluations = max_evaluations
        self.max_attempts = 3 * max_evaluations if max_attempts is None else max_attempts
        population = await self._initialize_population(population_size)
        self.history.append(tuple(population))
        while self._within_budget():
            offspring = await self._generate_offspring(population, len(self.history))
            population = complementary_select(population + offspring, population_size)
            self.history.append(tuple(population))
        return population

    def _within_budget(self) -> bool:
        return self.evaluations < self.max_evaluations and len(self.attempts) < self.max_attempts

    async def _initialize_population(self, size: int) -> list[Individual]:
        population = []
        for _ in range(3 * size):
            if not self._within_budget():
                break
            candidate = await self._attempt("INIT", (), 0)
            if candidate is not None:
                population.append(candidate)
            if len(population) == size:
                return complementary_select(population, size)
        raise RuntimeError("could not initialize a full population; inspect attempts")

    async def _generate_offspring(
        self, population: list[Individual], generation: int
    ) -> list[Individual]:
        offspring = []
        for _ in range(len(population)):
            if not self._within_budget():
                break
            if len(population) > 1 and self.rng.random() < 0.5:
                operation = "CS"
                parents = max(
                    combinations(population, 2),
                    key=lambda pair: sum(
                        abs(a - b) for a, b in zip(pair[0].scores, pair[1].scores, strict=True)
                    ),
                )
            else:
                operation = "LS"
                parents = (self._select_local_parent(population),)
            candidate = await self._attempt(operation, parents, generation)
            if candidate is not None:
                offspring.append(candidate)
        return offspring

    def _select_local_parent(self, population: list[Individual]) -> Individual:
        ranked = sorted(population, key=lambda p: p.fitness)
        # Official EoH-S uses zero-based rank weights; sort by mean as the paper specifies.
        return self.rng.choices(
            ranked, weights=[1 / (rank + len(ranked)) for rank in range(len(ranked))], k=1
        )[0]

    async def _attempt(
        self, operation: str, parents: tuple[Individual, ...], generation: int
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

        # Capture raw output before Slick parsing, including rejected responses.
        execution = SimpleNamespace(acall=recorded_call)
        try:
            try:
                if operation == "INIT":
                    proposal = await self.initialize(provider=execution)
                elif operation == "CS":
                    proposal = await self.complementary_search(parents, provider=execution)
                else:
                    proposal = await self.local_search(parents[0], provider=execution)
            except (ValueError, ProviderError, TimeoutError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                return None
            record["proposal"] = proposal
            self.evaluations += 1
            scores = tuple(await self.evaluate(proposal.content))
            record["scores"] = scores
            if not scores or not all(math.isfinite(score) for score in scores):
                raise EvaluationError("scores must be a nonempty finite vector")
            if self.instance_count is not None and len(scores) != self.instance_count:
                raise EvaluationError("score vector must retain the same instance count")
            self.instance_count = len(scores)
            return Individual(**proposal.model_dump(), id=record["id"], scores=scores)
        except (EvaluationError, TimeoutError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            return None
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
