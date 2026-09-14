"""Optimize definition/example instructions with objective-guided NSGA-II variation."""

import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

from prompt_optimization.pareto import checked_scores, fronts, nsga_select

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Instruction(BaseModel, extra="forbid"):
    definition: Text
    examples: list[Text]


class Definition(BaseModel, extra="forbid"):
    text: Text


class Examples(BaseModel, extra="forbid"):
    examples: Annotated[list[Text], Field(max_length=2)]


@dataclass(frozen=True)
class Individual:
    instruction: Instruction
    scores: tuple[float, ...]


class InstOptima:
    """Follow the paper's four operators; caller maps objectives to maximization.

    Provider/evaluator errors propagate with no retries or implicit fine-tuning.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Instruction], Awaitable[Sequence[float]]],
        objectives: Sequence[str],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.objectives = objectives

    @prompt(template="initialize.j2", output_type=Instruction)
    async def initialize(self, *, generated: Instruction) -> Instruction:
        return generated

    @prompt(template="mutate_definition.j2", output_type=Definition)
    async def mutate_definition(self, parent: Individual, *, generated: Definition) -> Instruction:
        return Instruction(definition=generated.text, examples=parent.instruction.examples)

    @prompt(template="cross_definition.j2", output_type=Definition)
    async def cross_definition(
        self, first: Individual, second: Individual, *, generated: Definition
    ) -> Instruction:
        return Instruction(definition=generated.text, examples=first.instruction.examples)

    @prompt(template="mutate_examples.j2", output_type=Examples)
    async def mutate_examples(self, parent: Individual, *, generated: Examples) -> Instruction:
        return Instruction(definition=parent.instruction.definition, examples=generated.examples)

    @prompt(template="cross_examples.j2", output_type=Examples)
    async def cross_examples(
        self, first: Individual, second: Individual, *, generated: Examples
    ) -> Instruction:
        if not set(generated.examples) <= set(
            first.instruction.examples + second.instruction.examples
        ):
            raise ValueError("example crossover must select examples from its parents")
        return Instruction(definition=first.instruction.definition, examples=generated.examples)

    async def _assess(self, instruction):
        scores = checked_scores(await self.evaluate(instruction), len(self.objectives))
        self.evaluations += 1
        return Individual(instruction, scores)

    async def _offspring(self, population):
        children = []
        for parent in population:
            second = self.rng.choice(population)
            operator = self.rng.choice(
                (
                    self.mutate_definition,
                    self.cross_definition,
                    self.mutate_examples,
                    self.cross_examples,
                )
            )
            arguments = (
                (parent, second)
                if operator in (self.cross_definition, self.cross_examples)
                else (parent,)
            )
            instruction = await operator(*arguments, provider=self.provider)
            children.append(await self._assess(instruction))
        return children

    async def _refresh(self, population, probability):
        for index in fronts([p.scores for p in population])[0]:
            if self.rng.random() < probability:
                population[index] = await self._assess(
                    await self.initialize(provider=self.provider)
                )
        return population

    async def run(
        self,
        initial: Sequence[Instruction] = (),
        *,
        population_size: int = 10,
        generations: int = 10,
        refresh_probability: float = 0.1,
        seed: int = 0,
    ) -> dict:
        self.rng = random.Random(seed)
        self.evaluations = 0
        instructions = list(initial)
        for _ in range(population_size - len(instructions)):
            instructions.append(await self.initialize(provider=self.provider))
        population = [await self._assess(value) for value in instructions]
        for _ in range(generations):
            combined = population + await self._offspring(population)
            population = [
                combined[i] for i in nsga_select([p.scores for p in combined], len(population))
            ]
            population = await self._refresh(population, refresh_probability)
        return {
            "population": population,
            "pareto": [population[i] for i in fronts([p.scores for p in population])[0]],
            "evaluations": self.evaluations,
        }
