"""Evolve prompts through generated text phenotypes with NSGA-II or S-metric selection."""

import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

from prompt_optimization.pareto import checked_scores, dominates, fronts, nsga_select

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class GeneratedText(BaseModel, extra="forbid"):
    text: Text


@dataclass(frozen=True)
class Individual:
    prompt: str
    text: str
    scores: tuple[float, ...]


def hypervolume_2d(scores, reference):
    """Exact union area of maximizing rectangles above a two-dimensional reference."""
    rx, ry = reference
    points = [(x, y) for x, y in scores if x > rx and y > ry]
    area, previous = 0.0, rx
    for x in sorted({x for x, _ in points}):
        area += (x - previous) * (max(y for px, y in points if px >= x) - ry)
        previous = x
    return area


class EMO:
    """Maximize objective vectors measured on generated text, with no retries.

    NSGA-II supports any objective count; SMS uses exact two-objective hypervolume.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Sequence[float]]],
        objectives: Sequence[str],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.objectives = objectives

    @prompt(template="crossover.j2", output_type=GeneratedText)
    async def crossover(self, first: str, second: str, *, generated: GeneratedText) -> str:
        return generated.text

    @prompt(template="change.j2", output_type=GeneratedText)
    async def change(self, instruction: str, *, generated: GeneratedText) -> str:
        return generated.text

    @prompt(template="modify.j2", output_type=GeneratedText)
    async def modify(self, instruction: str, *, generated: GeneratedText) -> str:
        return generated.text

    @prompt(template="paraphrase.j2", output_type=GeneratedText)
    async def paraphrase(self, instruction: str, *, generated: GeneratedText) -> str:
        return generated.text

    @prompt(template="generate.j2", output_type=GeneratedText)
    async def generate(self, instruction: str, *, generated: GeneratedText) -> str:
        return generated.text

    async def _assess(self, instruction):
        text = await self.generate(instruction, provider=self.provider)
        scores = checked_scores(await self.evaluate(text), len(self.objectives))
        self.evaluations += 1
        return Individual(instruction, text, scores)

    async def _offspring(self, population, count):
        children = []
        for _ in range(count):
            first, second = self.rng.sample(population, 2)
            crossed = await self.crossover(first.prompt, second.prompt, provider=self.provider)
            operation = self.rng.choice((self.change, self.modify, self.paraphrase))
            instruction = await operation(crossed, provider=self.provider)
            children.append(await self._assess(instruction))
        return children

    def _select_sms(self, combined, size, reference):
        selected = list(combined)
        while len(selected) > size:
            values = [p.scores for p in selected]
            ranked = fronts(values)
            worst = ranked[-1]
            if len(ranked) > 1:
                # Remove the most dominated member of the worst front first.
                remove = max(worst, key=lambda i: sum(dominates(v, values[i]) for v in values))
            else:
                total = hypervolume_2d(values, reference)
                remove = min(
                    worst,
                    key=lambda i: total
                    - hypervolume_2d([v for j, v in enumerate(values) if j != i], reference),
                )
            selected.pop(remove)
        return selected

    async def run(
        self,
        initial: Sequence[str],
        *,
        generations: int = 30,
        offspring_size: int = 20,
        selection: Literal["nsga2", "sms"] = "nsga2",
        reference: tuple[float, float] = (0.0, 0.0),
        seed: int = 0,
    ) -> dict:
        self.rng = random.Random(seed)
        self.evaluations = 0
        population = [await self._assess(instruction) for instruction in initial]
        for _ in range(generations):
            combined = population + await self._offspring(population, offspring_size)
            if selection == "sms":
                population = self._select_sms(combined, len(population), reference)
            else:
                population = [
                    combined[i] for i in nsga_select([p.scores for p in combined], len(population))
                ]
        return {
            "population": population,
            "pareto": [population[i] for i in fronts([p.scores for p in population])[0]],
            "evaluations": self.evaluations,
        }
