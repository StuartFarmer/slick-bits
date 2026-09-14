"""Construct, merge, solve and age component subsets using the official CMSA rules."""

# SPDX-License-Identifier: GPL-2.0-or-later
# Algorithm adaptation of Christian Blum's (C) 2024 CMSA implementation.
# See SOURCES.md for the official archive, attribution and deliberate changes.

import math
import random
import time
from collections.abc import Awaitable, Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

C = TypeVar("C", bound=Hashable)


def selection_probabilities(
    costs: Sequence[float], ages: Sequence[int], *, entropy: bool = False
) -> list[float]:
    """Compute official V1/V2 weights; cost generalizes the MIS vertex degree."""
    weights = [1 / (2 + age) + 1 / (1 + cost) for cost, age in zip(costs, ages)]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    if entropy:
        h = -sum(p * math.log(p) for p in probabilities if p > 0)
        probabilities = [(p + h) / (1 + len(probabilities) * h) for p in probabilities]
    return probabilities


@dataclass(frozen=True)
class Solution(Generic[C]):
    components: frozenset[C]
    score: float


class CMSA(Generic[C]):
    """Optimize finite component subsets through caller-owned domain operations.

    can_add(partial, component) defines feasible construction steps and stopping.
    solve(pool, seconds) returns a feasible subset or None if no incumbent exists.
    evaluate validates complete solutions and returns a finite objective score.
    Costs are static nonnegative desirability costs (smaller is preferred).
    """

    def __init__(
        self,
        components: Sequence[C],
        costs: Mapping[C, float],
        can_add: Callable[[frozenset[C], C], bool],
        solve: Callable[[frozenset[C], float], Awaitable[frozenset[C] | None]],
        evaluate: Callable[[frozenset[C]], Awaitable[float]],
        *,
        variant: Literal["baseline", "v1", "v2"] = "v1",
        maximize: bool = True,
    ):
        self.components = tuple(sorted(components, key=costs.__getitem__))
        self.costs, self.can_add, self.solve, self.evaluate = costs, can_add, solve, evaluate
        self.variant, self.maximize = variant, maximize
        self.age: dict[C, int] = {}
        self.history: list[dict] = []
        self.best: Solution[C] | None = None

    async def run(
        self,
        *,
        iterations: int = 100,
        constructions: int = 10,
        age_max: int = 10,
        determinism_rate: float = 0.8,
        candidate_list_size: int = 5,
        solve_time_limit: float = 10.0,
        time_limit: float | None = None,
        seed: int = 0,
    ) -> Solution[C] | None:
        """Run bounded iterations, optionally with a cooperative wall-clock limit.

        Callbacks must honor their budgets; running work is not forcibly killed.
        None means the budget ended before any complete solution was evaluated.
        A missing solver incumbent skips adaptation, matching the official code.
        """
        self.age = dict.fromkeys(self.components, -1)
        self.history, self.best = [], None
        self.rng = random.Random(seed)
        deadline = math.inf if time_limit is None else time.monotonic() + time_limit
        for _ in range(iterations):
            built = await self._construct_and_merge(
                constructions, determinism_rate, candidate_list_size, deadline
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            pool = frozenset(c for c in self.components if self.age[c] >= 0)
            solved = await self.solve(pool, min(solve_time_limit, remaining))
            if solved is not None:
                if not solved <= pool:
                    raise ValueError("solver returned components outside the merged pool")
                await self._retain(solved)
                self._adapt(solved, age_max)
            self.history.append(
                {
                    "pool": pool,
                    "solved": solved,
                    "ages": self.age.copy(),
                    "best": self.best,
                    "constructions": built,
                }
            )
        return self.best

    async def _construct_and_merge(self, count, determinism_rate, candidate_list_size, deadline):
        built = 0
        for _ in range(count):
            if time.monotonic() >= deadline:
                break
            chosen = frozenset()
            while True:
                # ponytail: rescan components; use a domain-specific incremental constructor
                # if feasibility checks dominate runtime on large instances.
                feasible = [
                    c for c in self.components if c not in chosen and self.can_add(chosen, c)
                ]
                if not feasible:
                    break
                component = self._select(feasible, determinism_rate, candidate_list_size)
                chosen = chosen | {component}
                # Reconstructing an existing component does not rejuvenate it.
                if self.age[component] == -1:
                    self.age[component] = 0
            await self._retain(chosen)
            built += 1
        return built

    def _select(self, feasible, determinism_rate, candidate_list_size):
        if self.rng.random() <= determinism_rate:
            # The official code takes the first in static cost order, not argmin(weight).
            return feasible[0]
        if self.variant == "baseline":
            return self.rng.choice(feasible[:candidate_list_size])
        weights = selection_probabilities(
            [self.costs[c] for c in feasible],
            [self.age[c] for c in feasible],
            entropy=self.variant == "v2",
        )
        return self.rng.choices(feasible, weights=weights, k=1)[0]

    async def _retain(self, components: frozenset[C]):
        score = await self.evaluate(components)
        if not math.isfinite(score):
            raise ValueError("measured objective must be finite")
        if self.best is None or (
            score > self.best.score if self.maximize else score < self.best.score
        ):
            self.best = Solution(components, score)

    def _adapt(self, solved: frozenset[C], age_max: int):
        for component, age in self.age.items():
            if age >= 0:
                age = 0 if component in solved else age + 1
                self.age[component] = -1 if age >= age_max else age
