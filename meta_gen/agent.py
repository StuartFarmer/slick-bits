"""Optimize caller-defined MetaGen domains with population mutation or annealing.

Adapted from the paper's sections 4.1–4.2 and the official MetaGen implementation.
SPDX-License-Identifier: GPL-3.0-or-later
Upstream copyright (C) 2023 David Gutierrez Avilés and Manuel Jesús Jiménez Navarro.
"""

import math
import random
from collections.abc import Awaitable, Callable
from copy import deepcopy

from metagen.framework import Domain, Solution

Evaluator = Callable[[Solution], Awaitable[float]]


def new_solution(domain: Domain) -> Solution:
    connector = domain.get_connector()
    solution_type = connector.get_type(domain.get_core())
    return solution_type(domain.get_core(), connector=connector)


async def evaluate_solution(solution: Solution, evaluate: Evaluator) -> None:
    # Protect the candidate from evaluator edits and retained mutable references.
    score = float(await evaluate(deepcopy(solution)))
    if not math.isfinite(score):
        raise ValueError("Evaluation must return a finite fitness")
    solution.set_fitness(score)


class RandomSearch:
    """Mutate every individual each round and retain a separate best snapshot.

    Fitness is minimized. The evaluator owns execution isolation and retries;
    errors propagate. Counts include attempted evaluations, including failures.
    Uses the official library's process-global Python random generator.
    """

    def __init__(self, task: str, domain: Domain, evaluate: Evaluator):
        self.task, self.domain, self.evaluate = task, domain, evaluate

    async def initialize(self, population_size: int) -> None:
        self.population = []
        for _ in range(population_size):
            solution = new_solution(self.domain)
            self.evaluations += 1
            await evaluate_solution(solution, self.evaluate)
            self.population.append(solution)
        self.best = deepcopy(min(self.population, key=lambda solution: solution.fitness))

    async def mutate_population(self) -> None:
        for solution in self.population:
            solution.mutate()
            self.evaluations += 1
            await evaluate_solution(solution, self.evaluate)
            if solution.fitness < self.best.fitness:
                self.best = deepcopy(solution)

    async def run(self, *, population_size: int = 30, iterations: int = 20) -> Solution:
        """Use population_size * (iterations + 1) evaluations on a successful run."""
        self.evaluations = 0
        self.history = []
        await self.initialize(population_size)
        self.history.append(self.best.fitness)
        for _ in range(iterations):
            await self.mutate_population()
            self.history.append(self.best.fitness)
        return deepcopy(self.best)


class SimulatedAnnealing:
    """Accept improvements and temperature-weighted worse neighbors; minimize fitness.

    Returns the best observed solution; ``current`` retains the final accepted
    state. The evaluator owns execution isolation and retries; errors propagate.
    Uses the official library's process-global Python random generator.
    """

    def __init__(self, task: str, domain: Domain, evaluate: Evaluator):
        self.task, self.domain, self.evaluate = task, domain, evaluate

    async def initialize(self) -> None:
        self.current = new_solution(self.domain)
        self.evaluations += 1
        await evaluate_solution(self.current, self.evaluate)
        self.best = deepcopy(self.current)

    async def propose_neighbor(self, alteration_limit: float | None) -> Solution:
        neighbor = deepcopy(self.current)
        neighbor.mutate(alteration_limit=alteration_limit)
        self.evaluations += 1
        await evaluate_solution(neighbor, self.evaluate)
        return neighbor

    def accept_neighbor(self, neighbor: Solution) -> None:
        delta = neighbor.fitness - self.current.fitness
        # Evaluate exp only for worse moves; cooling can underflow to zero.
        accept = delta <= 0 or (
            self.temperature > 0 and random.random() < math.exp(-delta / self.temperature)
        )
        self.accepted.append(accept)
        if accept:
            self.current = neighbor
        if neighbor.fitness < self.best.fitness:
            self.best = deepcopy(neighbor)

    async def run(
        self,
        *,
        iterations: int = 50,
        alteration_limit: float | None = 0.1,
        initial_temp: float = 50.0,
        cooling_rate: float = 0.99,
    ) -> Solution:
        """Evaluate an initial solution and exactly ``iterations`` neighbors.

        ``alteration_limit`` is passed to official mutation unchanged: it is an
        absolute numeric radius, not a percentage; None permits full-range moves.
        """
        self.evaluations = 0
        self.history, self.accepted = [], []
        self.temperature = initial_temp
        await self.initialize()
        self.history.append(self.best.fitness)
        for _ in range(iterations):
            neighbor = await self.propose_neighbor(alteration_limit)
            self.accept_neighbor(neighbor)
            self.temperature *= cooling_rate
            self.history.append(self.best.fitness)
        return deepcopy(self.best)
