"""Evolve complete prompts with scored parents and exponential roulette selection."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Individual:
    prompt: str
    score: float


class SPELL:
    """Own semantic reproduction; failures propagate and higher finite scores win."""

    def __init__(self, task: str, provider: Provider, evaluate: Callable[[str], Awaitable[float]]):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.population: list[Individual] = []
        self.attempts: list[dict] = []
        self.evaluations = self.optimizer_calls = 0
        self.last_response: str | None = None

    @prompt(template="reproduce.j2")
    async def reproduce(self, parents: Sequence[Individual], *, generated: str) -> str:
        self.last_response = generated  # Retain raw text even when extraction fails.
        matches, depth, start = [], 0, 0
        for index, character in enumerate(generated):
            if character == "{":
                if depth == 0:
                    start = index + 1
                depth += 1
            elif character == "}":
                depth -= 1
                if depth < 0:
                    raise ValueError("unmatched closing brace")
                if depth == 0:
                    matches.append(generated[start:index])
        if depth:
            raise ValueError("unmatched opening brace")
        if len(matches) != 1 or not matches[0].strip():
            raise ValueError(f"expected one nonblank brace-delimited prompt: {generated!r}")
        return matches[0].strip()

    async def _assess(self, text):
        self.evaluations += 1
        score = float(await self.evaluate(text))
        if not math.isfinite(score):
            raise ValueError("fitness must be finite")
        return Individual(text, score)

    def _select(self, population, count):
        maximum = max(p.score for p in population)
        weights = [math.exp(p.score - maximum) for p in population]
        return self.rng.choices(population, weights=weights, k=count)

    async def _reproduce(self, population, parent_counts, generation):
        children = []
        for count in parent_counts:
            parents = self._select(population, count)
            record = {"generation": generation, "parents": parents, "status": "generation_failed"}
            self.attempts.append(record)
            self.last_response = None
            self.optimizer_calls += 1
            try:
                text = await self.reproduce(parents, provider=self.provider)
                record.update(prompt=text, status="evaluation_failed")
                child = await self._assess(text)
                children.append(child)
                record.update(score=child.score, status="evaluated")
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["raw_response"] = self.last_response
        return children

    def _survive(self, population, children):
        pool = population + children
        elite = max(pool, key=lambda p: p.score)
        return [elite] + self._select(pool, len(population) - 1)

    async def run(
        self,
        initial_prompts: Sequence[str],
        *,
        iterations: int = 500,
        parent_counts: Sequence[int] = (1, 1, 1, 1, 1, 2, 2, 2, 2, 2),
        seed: int = 42,
    ) -> dict:
        """Evolve a caller-supplied population with one elite and roulette survivors.

        Population size is len(initial_prompts); use [initial_prompt] * 20 for
        the paper's initialization. The default schedule makes ten calls per
        round. Failures abort without replacing the last completed population.
        """
        self.rng = random.Random(seed)
        self.evaluations = self.optimizer_calls = 0
        self.population, self.attempts, self.last_response = [], [], None
        self.population = [await self._assess(text) for text in initial_prompts]
        history = [max(p.score for p in self.population)]
        for generation in range(1, iterations + 1):
            children = await self._reproduce(self.population, parent_counts, generation)
            self.population = self._survive(self.population, children)
            history.append(max(p.score for p in self.population))
        return {
            "best": max(self.population, key=lambda p: p.score),
            "population": self.population.copy(),
            "history": history,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "attempts": self.attempts.copy(),
        }
