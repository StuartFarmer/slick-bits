"""Co-optimize generator and discriminator instructions and demonstrations by minimax loss."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Instruction(BaseModel, extra="forbid"):
    text: Text


class Example(BaseModel, extra="forbid", frozen=True):
    input: Text
    output: Text


class DiscriminatorExample(BaseModel, extra="forbid", frozen=True):
    input: Text
    output: Text
    label: Literal["real", "generated"]


@dataclass(frozen=True)
class GeneratorPrompt:
    instruction: str
    examples: tuple[Example, ...]


@dataclass(frozen=True)
class DiscriminatorPrompt:
    instruction: str
    examples: tuple[DiscriminatorExample, ...]


class AdvICL:
    """Generator callback returns text; discriminator returns log P(real).

    The evaluator requires actual model likelihoods, not self-rated confidence.
    Exact probability endpoints causing infinite loss are rejected. Each phase
    compares best-of-r replacements on the same sampled batch. Failures propagate.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        generate: Callable[[GeneratorPrompt, str], Awaitable[str]],
        evaluate: Callable[[DiscriminatorPrompt, str, str], Awaitable[float]],
    ):
        self.task, self.provider = task, provider
        self.generate, self.evaluate = generate, evaluate

    @prompt(template="generator_instruction.j2", output_type=Instruction)
    async def generator_instruction(self, instruction: str, *, generated: Instruction) -> str:
        return generated.text

    @prompt(template="discriminator_instruction.j2", output_type=Instruction)
    async def discriminator_instruction(self, instruction: str, *, generated: Instruction) -> str:
        return generated.text

    @prompt(template="generator_example.j2", output_type=Example)
    async def generator_example(self, example: Example, *, generated: Example) -> Example:
        return generated

    @prompt(template="discriminator_example.j2", output_type=DiscriminatorExample)
    async def discriminator_example(
        self, example: DiscriminatorExample, *, generated: DiscriminatorExample
    ) -> DiscriminatorExample:
        return generated

    async def _loss(self, generator, discriminator, batch):
        terms = []
        for example in batch:
            output = await self.generate(generator, example.input)
            if not output.strip():
                raise ValueError("generator returned empty text")
            real = await self.evaluate(discriminator, example.input, example.output)
            fake = await self.evaluate(discriminator, example.input, output)
            if not math.isfinite(real) or real > 0 or not math.isfinite(fake) or fake >= 0:
                raise ValueError(
                    "discriminator must return finite log probabilities with finite GAN loss"
                )
            terms.append(real + math.log(-math.expm1(fake)))
        loss = math.fsum(terms) / len(batch)
        if not math.isfinite(loss):
            raise ValueError("adversarial loss must be finite")
        self.evaluations += 1
        return loss

    async def _coordinate(self, generator, discriminator, batch, candidates, player, position):
        current = discriminator if player == "discriminator" else generator
        loss = await self._loss(generator, discriminator, batch)
        best, best_loss = current, loss
        for _ in range(candidates):
            if position is None:
                operation = (
                    self.discriminator_instruction
                    if player == "discriminator"
                    else self.generator_instruction
                )
                variant = replace(
                    current,
                    instruction=await operation(current.instruction, provider=self.provider),
                )
            else:
                operation = (
                    self.discriminator_example
                    if player == "discriminator"
                    else self.generator_example
                )
                example = await operation(current.examples[position], provider=self.provider)
                variant = replace(
                    current,
                    examples=current.examples[:position]
                    + (example,)
                    + current.examples[position + 1 :],
                )
            new_generator = generator if player == "discriminator" else variant
            new_discriminator = variant if player == "discriminator" else discriminator
            score = await self._loss(new_generator, new_discriminator, batch)
            improved = score > best_loss if player == "discriminator" else score < best_loss
            if improved:
                best, best_loss = variant, score
        self.history.append(
            {"player": player, "position": position, "before": loss, "after": best_loss}
        )
        return best

    async def _discriminator_phase(self, generator, discriminator, batch, candidates):
        for position in [None, *range(len(discriminator.examples))]:
            discriminator = await self._coordinate(
                generator, discriminator, batch, candidates, "discriminator", position
            )
        return discriminator

    async def _generator_phase(self, generator, discriminator, batch, candidates):
        for position in [None, *range(len(generator.examples))]:
            generator = await self._coordinate(
                generator, discriminator, batch, candidates, "generator", position
            )
        return generator

    async def run(
        self,
        generator: GeneratorPrompt,
        discriminator: DiscriminatorPrompt,
        training: Sequence[Example],
        *,
        rounds: int = 5,
        batch_size: int = 3,
        candidates: int = 5,
        seed: int = 0,
    ) -> dict:
        rng = random.Random(seed)
        self.history, self.evaluations = [], 0
        for _ in range(rounds):
            batch = rng.sample(list(training), batch_size)
            discriminator = await self._discriminator_phase(
                generator, discriminator, batch, candidates
            )
            generator = await self._generator_phase(generator, discriminator, batch, candidates)
        return {
            "generator": generator,
            "discriminator": discriminator,
            "history": self.history,
            "evaluations": self.evaluations,
        }
