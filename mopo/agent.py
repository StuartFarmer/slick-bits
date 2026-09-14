"""Co-evolve task prompts and their operators with Pareto and objective-specific survival."""

import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

from prompt_optimization.pareto import checked_scores, fronts, nsga_select

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class GeneratedText(BaseModel, extra="forbid"):
    text: Text


@dataclass(frozen=True)
class Individual:
    prompt: str
    scores: tuple[float, ...]
    operator: tuple[str, str] | None = None


class MOPO:
    """Optimize three prompt layers, preserving attribution to learned operators.

    Evaluator owns conditional text generation, filtering, and objective aggregation.
    fill_mask(prefix, suffix) returns a predicted whitespace token. Errors propagate.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Sequence[float]]],
        objectives: Sequence[str],
        fill_mask: Callable[[str, str], Awaitable[str]],
        required_tokens: Sequence[str] = (),
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.objectives, self.fill_mask = objectives, fill_mask
        self.required_tokens = required_tokens

    def _checked(self, text):
        if not text.strip() or any(token not in text for token in self.required_tokens):
            raise ValueError("generated prompt is blank or lost a required token")
        return text.strip()

    @prompt(template="evolve_combine.j2", output_type=GeneratedText)
    async def evolve_combine(self, instruction: str, *, generated: GeneratedText) -> str:
        return generated.text

    @prompt(template="evolve_paraphrase.j2", output_type=GeneratedText)
    async def evolve_paraphrase(self, instruction: str, *, generated: GeneratedText) -> str:
        return generated.text

    @prompt(template="combine.j2", output_type=GeneratedText)
    async def combine(
        self, first: str, second: str, instruction: str, *, generated: GeneratedText
    ) -> str:
        return self._checked(generated.text)

    @prompt(template="paraphrase.j2", output_type=GeneratedText)
    async def paraphrase(self, parent: str, instruction: str, *, generated: GeneratedText) -> str:
        return self._checked(generated.text)

    async def _assess(self, text, operator=None):
        scores = checked_scores(await self.evaluate(text), len(self.objectives))
        self.evaluations += 1
        return Individual(text, scores, operator)

    async def _evolve_operators(self, combine, paraphrase):
        combined = list(combine)
        for instruction in combine:
            combined.append(await self.evolve_combine(instruction, provider=self.provider))
        paraphrased = list(paraphrase)
        for instruction in paraphrase:
            paraphrased.append(await self.evolve_paraphrase(instruction, provider=self.provider))
        return list(dict.fromkeys(combined)), list(dict.fromkeys(paraphrased))

    def _pairs(self, population):
        champions = []
        for objective in range(len(self.objectives)):
            remaining = [p for p in population if p not in champions]
            if remaining:
                champions.append(max(remaining, key=lambda p: p.scores[objective]))
        return list(combinations(champions, 2))

    async def _word_variants(self, text):
        words = text.split()
        position = self.rng.randrange(len(words) + 1)
        token = await self.fill_mask(" ".join(words[:position]), " ".join(words[position:]))
        if len(token.split()) != 1:
            raise ValueError("mask filler must generate one nonblank whitespace token")
        variants = [" ".join(words[:position] + [token.strip()] + words[position:])]
        editable = [
            i
            for i, word in enumerate(words)
            if not any(required in word for required in self.required_tokens)
        ]
        if editable:
            position = self.rng.choice(editable)
            if len(words) > 1:
                variants.append(" ".join(words[:position] + words[position + 1 :]))
            token = await self.fill_mask(
                " ".join(words[:position]), " ".join(words[position + 1 :])
            )
            if len(token.split()) != 1:
                raise ValueError("mask filler must generate one nonblank whitespace token")
            variants.append(" ".join(words[:position] + [token.strip()] + words[position + 1 :]))
        return [self._checked(value) for value in variants]

    async def _breed(self, population, combine, paraphrase, count):
        children = []
        for first, second in self._pairs(population):
            for operator in combine:
                for _ in range(count):
                    text = await self.combine(
                        first.prompt, second.prompt, operator, provider=self.provider
                    )
                    children.append(await self._assess(text, ("combine", operator)))
        # Match the official main script: both branches read the seed population.
        for parent in population:
            for operator in paraphrase:
                for _ in range(count):
                    text = await self.paraphrase(parent.prompt, operator, provider=self.provider)
                    children.append(await self._assess(text, ("paraphrase", operator)))
            for _ in range(count):
                for text in await self._word_variants(parent.prompt):
                    children.append(await self._assess(text))
        return children

    def _survive(self, population, size, specialists):
        # Keep first occurrence and its measurement; no cross-generation cache.
        unique = {}
        for individual in population:
            unique.setdefault(individual.prompt, individual)
        population = list(unique.values())
        chosen = nsga_select([p.scores for p in population], min(size, len(population)))
        for objective in range(len(self.objectives)):
            remaining = [i for i in range(len(population)) if i not in chosen]
            chosen.extend(
                sorted(remaining, key=lambda i: population[i].scores[objective], reverse=True)[
                    :specialists
                ]
            )
        return [population[i] for i in chosen]

    def _credit(self, operators, population, kind, size):
        contributors = {p.operator[1] for p in population if p.operator and p.operator[0] == kind}
        selected = [operator for operator in operators if operator in contributors][:size]
        remaining = [operator for operator in operators if operator not in selected]
        selected.extend(self.rng.sample(remaining, min(size - len(selected), len(remaining))))
        return selected

    async def run(
        self,
        initial: Sequence[str],
        combine: Sequence[str],
        paraphrase: Sequence[str],
        *,
        generations: int = 5,
        population_size: int = 10,
        specialists: int = 3,
        operator_size: int = 4,
        offspring_per_operator: int = 1,
        seed: int = 0,
    ) -> dict:
        self.rng = random.Random(seed)
        self.evaluations = 0
        population = [await self._assess(text) for text in initial]
        archive = list(population)
        for _ in range(generations):
            combine, paraphrase = await self._evolve_operators(combine, paraphrase)
            children = await self._breed(population, combine, paraphrase, offspring_per_operator)
            population = self._survive(population + children, population_size, specialists)
            archive.extend(population)
            combine = self._credit(combine, population, "combine", operator_size)
            paraphrase = self._credit(paraphrase, population, "paraphrase", operator_size)
        population = self._survive(archive, population_size, specialists)
        return {
            "population": population,
            "pareto": [population[i] for i in fronts([p.scores for p in population])[0]],
            "combine": list(combine),
            "paraphrase": list(paraphrase),
            "evaluations": self.evaluations,
        }
