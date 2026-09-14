"""Evolve task instructions using GA or DE and caller-owned fitness evaluation."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from slick import Session, prompt
from slick.providers import Provider


def extract_prompt(response: str) -> str:
    """Check the tagged text contract; include rejected raw output in the error."""
    match = re.search(r"<prompt>(.*?)</prompt>", response, flags=re.DOTALL)
    if (
        response.count("<prompt>") != 1
        or response.count("</prompt>") != 1
        or match is None
        or not match.group(1).strip()
    ):
        raise ValueError(
            f"expected exactly one nonempty <prompt>...</prompt>; response={response!r}"
        )
    return match.group(1).strip()


class EvoPrompt:
    """Own the task context, generation methods, and explicit population search.

    Fitness must be finite and nonnegative, with higher scores preferred.
    Cached fitness assumes repeatable evaluation; disable caching for noisy
    scores. Provider, parsing, and evaluator failures propagate without retries.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate

    @prompt(template="ga_offspring.j2")
    async def ga_offspring(self, parent1: str, parent2: str, *, generated: str) -> str:
        """Cross over and mutate two parents, then check the final instruction."""
        return extract_prompt(generated)

    @prompt(template="de_offspring.j2")
    async def de_offspring(
        self, donor1: str, donor2: str, best: str, target: str, *, generated: str
    ) -> str:
        """Generate a checked DE trial from one fixed generation's parents."""
        return extract_prompt(generated)

    @prompt(template="variation.j2")
    async def variation(self, instruction: str, *, generated: str) -> str:
        """Fill the population with a checked variation of an initial instruction."""
        return extract_prompt(generated)

    async def _score(self, instruction: str) -> dict:
        if self.use_cache and instruction in self.fitness:
            self.cache_hits += 1
            value = self.fitness[instruction]
        else:
            value = float(await self.evaluate(instruction))
            if not math.isfinite(value) or value < 0:
                raise ValueError("fitness must be finite and nonnegative (higher is better)")
            self.evaluations += 1
            if self.use_cache:
                self.fitness[instruction] = value
        return {"prompt": instruction, "score": value}

    async def _initialize(self, prompts, size, execution):
        manual = prompts.copy()
        for _ in range(size - len(prompts)):
            prompts.append(await self.variation(self.rng.choice(manual), **execution))
            self.optimizer_calls += 1
        return [await self._score(p) for p in prompts]

    async def _ga_generation(self, population, execution):
        # Scaling preserves roulette probabilities without overflowing their sum.
        maximum = max(p["score"] for p in population)
        weights = [p["score"] / maximum for p in population] if maximum else None
        offspring = []
        for _ in range(len(population)):
            parents = [p["prompt"] for p in self.rng.choices(population, weights=weights, k=2)]
            child = await self.ga_offspring(*parents, **execution)
            self.optimizer_calls += 1
            offspring.append(await self._score(child))
        # Stable ties prefer incumbents, and every parent came from the old population.
        return sorted(population + offspring, key=lambda p: p["score"], reverse=True)[
            : len(population)
        ]

    async def _de_generation(self, population, execution):
        best = max(population, key=lambda p: p["score"])
        offspring = []
        for slot, target in enumerate(population):
            donors = self.rng.sample([j for j in range(len(population)) if j != slot], 2)
            child = await self.de_offspring(
                population[donors[0]]["prompt"],
                population[donors[1]]["prompt"],
                best["prompt"],
                target["prompt"],
                **execution,
            )
            self.optimizer_calls += 1
            trial = await self._score(child)
            offspring.append(trial if trial["score"] > target["score"] else target)
        return offspring

    @staticmethod
    def _snapshot(population, iteration):
        return {
            "iteration": iteration,
            "best_score": max(p["score"] for p in population),
            "mean_score": sum(p["score"] / len(population) for p in population),
        }

    async def run(
        self,
        initial_prompts: Sequence[str],
        *,
        algorithm: Literal["ga", "de"] = "de",
        population_size: int | None = None,
        iterations: int = 10,
        seed: int = 0,
        cache: bool = True,
        session: Session | None = None,
    ) -> dict:
        """Initialize, generate, evaluate, and select each frozen generation.

        GA uses roulette sampling and global elitism. DE uses distinct donors
        excluding the target and requires strict improvement for replacement.
        Run one search at a time per agent; all generation calls are sequential.
        """
        prompts = [p.strip() for p in initial_prompts]
        size = len(prompts) if population_size is None else population_size
        generation = {"ga": self._ga_generation, "de": self._de_generation}[algorithm]
        execution = {"session": session} if session is not None else {"provider": self.provider}
        self.rng = random.Random(seed)
        self.use_cache = cache
        self.fitness = {}
        self.evaluations = self.cache_hits = self.optimizer_calls = 0
        population = await self._initialize(prompts, size, execution)
        initial_best = max(population, key=lambda p: p["score"])
        history = [self._snapshot(population, 0)]
        for iteration in range(1, iterations + 1):
            population = await generation(population, execution)
            history.append(self._snapshot(population, iteration))
        return {
            "algorithm": algorithm,
            "seed": seed,
            "iterations": iterations,
            "population_size": size,
            "cache": cache,
            "initial_best": initial_best,
            "best": max(population, key=lambda p: p["score"]),
            "population": population,
            "history": history,
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "optimizer_calls": self.optimizer_calls,
        }
