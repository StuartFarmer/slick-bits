"""Optimize objectives exhaustively in sequence using feedback and local neighborhoods."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

from prompt_optimization.pareto import checked_scores

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class GeneratedText(BaseModel, extra="forbid"):
    text: Text


@dataclass(frozen=True)
class Evaluation:
    scores: tuple[float, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class Individual:
    prompt: str
    evaluation: Evaluation


class SoS:
    """Preserve vector neighborhoods; weights control stopping and final ranking only.

    Evaluator returns objective-aligned scores and observed-error descriptions.
    All failures propagate. A per-objective cap bounds exhaustive improvement.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        objectives: Sequence[str],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.objectives = objectives

    @prompt(template="semantic.j2", output_type=GeneratedText)
    async def semantic(self, instruction: str, *, generated: GeneratedText) -> str:
        return generated.text

    @prompt(template="feedback.j2", output_type=GeneratedText)
    async def feedback(
        self, instruction: str, objective: str, errors: str, *, generated: GeneratedText
    ) -> str:
        return generated.text

    @prompt(template="improve.j2", output_type=GeneratedText)
    async def improve(
        self, instruction: str, objective: str, feedback: str, *, generated: GeneratedText
    ) -> str:
        return generated.text

    @prompt(template="crossover.j2", output_type=GeneratedText)
    async def crossover(self, first: str, second: str, *, generated: GeneratedText) -> str:
        return generated.text

    async def _assess(self, instruction):
        evaluation = await self.evaluate(instruction)
        checked_scores(evaluation.scores, len(self.objectives))
        self.evaluations += 1
        return Individual(instruction, evaluation)

    def _local_optima(self, population, delta):
        unique = {p.prompt: p for p in population}
        population = list(unique.values())
        selected = []
        for p in population:
            for objective in range(len(self.objectives)):
                if all(
                    q.evaluation.scores[objective] <= p.evaluation.scores[objective]
                    for q in population
                    if sum(
                        abs(a - b)
                        for j, (a, b) in enumerate(
                            zip(p.evaluation.scores, q.evaluation.scores, strict=True)
                        )
                        if j != objective
                    )
                    < delta
                ):
                    selected.append(p)
                    break
        return selected

    def _weighted(self, individual, weights):
        value = math.fsum(
            score * weight
            for score, weight in zip(individual.evaluation.scores, weights, strict=True)
        )
        if not math.isfinite(value):
            raise ValueError("weighted measured score must be finite")
        return value

    async def _optimize_objective(self, population, objective, weights, delta, gain, cap):
        for _ in range(cap):
            previous = max(self._weighted(p, weights) for p in population)
            children = []
            for parent in population:
                feedback = await self.feedback(
                    parent.prompt,
                    self.objectives[objective],
                    parent.evaluation.errors[objective],
                    provider=self.provider,
                )
                instruction = await self.improve(
                    parent.prompt, self.objectives[objective], feedback, provider=self.provider
                )
                children.append(await self._assess(instruction))
            improvement = max(self._weighted(p, weights) for p in children) - previous
            population = self._local_optima(population + children, delta)
            self.objective_history.append(objective)
            if improvement <= gain:
                break
        return population

    async def _cross(self, population, count):
        pairs = list(combinations(population, 2))
        self.rng.shuffle(pairs)
        children = []
        for first, second in pairs[:count]:
            instruction = await self.crossover(first.prompt, second.prompt, provider=self.provider)
            children.append(await self._assess(instruction))
        return children

    async def run(
        self,
        initial: str,
        *,
        weights: Sequence[float],
        variants: int = 4,
        delta: float = 0.05,
        minimum_gain: float = 0.0,
        max_rounds: int = 10,
        crossovers: int = 4,
        seed: int = 0,
    ) -> dict:
        self.rng = random.Random(seed)
        self.evaluations = 0
        self.objective_history = []
        population = [await self._assess(initial)]
        for _ in range(variants):
            population.append(
                await self._assess(await self.semantic(initial, provider=self.provider))
            )
        population = self._local_optima(population, delta)
        for objective in range(len(self.objectives)):
            population = await self._optimize_objective(
                population, objective, weights, delta, minimum_gain, max_rounds
            )
        population = self._local_optima(
            population + await self._cross(population, crossovers), delta
        )
        population.sort(key=lambda p: self._weighted(p, weights), reverse=True)
        return {
            "population": population,
            "best": population[0],
            "objectives": self.objective_history,
            "evaluations": self.evaluations,
        }
