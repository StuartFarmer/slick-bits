"""Use an LLM for parent selection, crossover-operator choice, and mutation choice."""

import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Parents(BaseModel, extra="forbid"):
    indices: list[Annotated[int, Field(strict=True)]]


class Offspring(BaseModel, extra="forbid"):
    operator: Text
    solution: Text


class LLMEvolution:
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        *,
        crossover_knowledge: str,
        mutation_knowledge: str,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.crossover_knowledge, self.mutation_knowledge = crossover_knowledge, mutation_knowledge

    @prompt(template="select.j2", output_type=Parents)
    async def select(self, population: list[dict], *, generated: Parents) -> list[int]:
        ids = generated.indices
        if len(ids) != 2 or len(set(ids)) != 2 or any(i < 0 or i >= len(population) for i in ids):
            raise ValueError("select two distinct current population indices")
        return ids

    @prompt(template="crossover.j2", output_type=Offspring)
    async def crossover(self, parents: list[dict], *, generated: Offspring) -> Offspring:
        return generated

    @prompt(template="mutate.j2", output_type=Offspring)
    async def mutate(self, solution: str, *, generated: Offspring) -> Offspring:
        return generated

    async def assess(self, text):
        if text not in self.archive:
            score = await self.evaluate(text)
            if not math.isfinite(score):
                raise ValueError("fitness must be finite")
            self.archive[text] = score
        return {"solution": text, "score": self.archive[text]}

    async def offspring(self, population, count):
        children, records = [], []
        for _ in range(count):
            ids = await self.select(population, provider=self.provider)
            crossed = await self.crossover([population[i] for i in ids], provider=self.provider)
            mutated = await self.mutate(crossed.solution, provider=self.provider)
            children.append(await self.assess(mutated.solution))
            records.append({"parents": ids, "crossover": crossed, "mutation": mutated})
        return children, records

    async def run(
        self,
        initial: Sequence[str],
        *,
        rounds: int = 10,
        population_size: int = 5,
        candidates: int = 16,
    ) -> dict:
        self.archive = {}
        population = [await self.assess(text) for text in dict.fromkeys(initial)]
        population = sorted(population, key=lambda p: p["score"], reverse=True)[:population_size]
        history = []
        for _ in range(rounds):
            children, records = await self.offspring(population, candidates)
            unique = {p["solution"]: p for p in population + children}
            population = sorted(unique.values(), key=lambda p: p["score"], reverse=True)[
                :population_size
            ]
            history.append(records)
        return {
            "best": population[0],
            "population": population,
            "history": history,
            "evaluations": len(self.archive),
        }
