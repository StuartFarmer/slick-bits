"""Evolve instruction/example pairs and allocate evaluations by statistical racing."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t
from slick import prompt
from slick.providers import Provider


def unpack(text):
    match = re.search(r"<prompt>(.*?)</prompt>", text, re.DOTALL)
    if not match or not match[1].strip():
        raise ValueError(f"missing nonempty prompt tags; response={text!r}")
    return match[1].strip()


def significantly_better(left, right, alpha):
    differences = np.asarray(left) - right
    if len(differences) < 2:
        return False
    standard_error = differences.std(ddof=1) / math.sqrt(len(differences))
    if standard_error == 0:
        return bool(differences.mean() > 0)
    return bool(t.sf(differences.mean() / standard_error, len(differences) - 1) < alpha)


@dataclass
class Candidate:
    instruction: str
    examples: list[str] = field(default_factory=list)
    score: float | None = None

    def render(self):
        return "\n\n".join([self.instruction, *self.examples])


class CAPO:
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str, str], Awaitable[np.ndarray]],
        demonstrate: Callable[[str, dict], Awaitable[tuple[str, str]]],
        token_count: Callable[[str], int] = lambda text: len(text.split()),
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.demonstrate, self.token_count = demonstrate, token_count

    @prompt(template="crossover.j2")
    async def crossover(self, mother: str, father: str, *, generated: str) -> str:
        return unpack(generated)

    @prompt(template="mutate.j2")
    async def mutate(self, instruction: str, *, generated: str) -> str:
        return unpack(generated)

    async def _examples(self, instruction, count):
        examples = []
        for row in self.rng.sample(self.demonstrations, count):
            self.demonstration_calls += 1
            prediction, reasoning = await self.demonstrate(instruction, row)
            output = reasoning if prediction == row["target"] else row["target"]
            examples.append(f"Input: {row['input']}\nOutput: {output}")
        return examples

    async def _offspring(self, population, count, upper_shots):
        children = []
        for _ in range(count):
            mother, father = self.rng.sample(population, 2)
            self.optimizer_calls += 1
            instruction = await self.crossover(
                mother.instruction, father.instruction, provider=self.provider
            )
            combined = mother.examples + father.examples
            examples = self.rng.sample(combined, len(combined) // 2)
            self.optimizer_calls += 1
            instruction = await self.mutate(instruction, provider=self.provider)
            decision = self.rng.random()
            if decision < 1 / 3 and len(examples) < upper_shots:
                examples += await self._examples(instruction, 1)
            elif decision < 2 / 3 and decision >= 1 / 3 and examples:
                examples = self.rng.sample(examples, len(examples) - 1)
            self.rng.shuffle(examples)
            children.append(Candidate(instruction, examples))
        return children

    async def _race(self, candidates, blocks, survivors, max_blocks, penalty, alpha):
        accumulated = [[] for _ in candidates]
        for block in blocks[:max_blocks]:
            for i, candidate in enumerate(candidates):
                key = (candidate.render(), block)
                if key not in self.cache:
                    self.evaluations += 1
                    scores = np.array(await self.evaluate(*key), dtype=float, copy=True)
                    if not np.isfinite(scores).all():
                        raise ValueError("measured block scores must be finite")
                    self.cache[key] = scores
                    self.example_evaluations += scores.size
                adjusted = (
                    self.cache[key]
                    - penalty * self.token_count(candidate.render()) / self.max_length
                )
                accumulated[i].extend(adjusted.tolist())
            beaten = [
                sum(significantly_better(other, scores, alpha) for other in accumulated)
                for scores in accumulated
            ]
            keep = [i for i, value in enumerate(beaten) if value < survivors]
            self.history.append(
                {
                    "block": block,
                    "prompts": [item.render() for item in candidates],
                    "beaten": beaten,
                    "retained": keep,
                }
            )
            candidates = [candidates[i] for i in keep]
            accumulated = [accumulated[i] for i in keep]
            if len(candidates) <= survivors:
                break
        for candidate in candidates:
            values = [
                scores for (text, _), scores in self.cache.items() if text == candidate.render()
            ]
            candidate.score = float(np.concatenate(values).mean())
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:survivors]

    async def run(
        self,
        initial_prompts: Sequence[str],
        blocks: Sequence[str],
        demonstrations: Sequence[dict] = (),
        *,
        iterations: int = 5,
        crossovers: int = 4,
        upper_shots: int = 0,
        max_blocks: int = 10,
        length_penalty: float = 0.1,
        significance: float = 0.05,
        shuffle_blocks: bool = True,
        seed: int = 0,
    ) -> dict:
        self.rng, self.demonstrations = random.Random(seed), list(demonstrations)
        self.optimizer_calls = self.evaluations = self.example_evaluations = (
            self.demonstration_calls
        ) = 0
        self.cache, self.history = {}, []
        population = []
        for instruction in initial_prompts:
            examples = await self._examples(instruction, self.rng.randint(0, upper_shots))
            population.append(Candidate(instruction, examples))
        self.max_length = max(self.token_count(item.render()) for item in population)
        for _ in range(iterations):
            children = await self._offspring(population, crossovers, upper_shots)
            order = list(blocks)
            if shuffle_blocks:
                self.rng.shuffle(order)
            population = await self._race(
                population + children,
                order,
                len(initial_prompts),
                max_blocks,
                length_penalty,
                significance,
            )
        return {
            "population": population,
            "history": self.history,
            "cache": self.cache,
            "optimizer_calls": self.optimizer_calls,
            "evaluations": self.evaluations,
            "example_evaluations": self.example_evaluations,
            "demonstration_calls": self.demonstration_calls,
        }
