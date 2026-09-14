"""Learn a SAMME prompt ensemble using reflection and bilateral confidence bagging."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Direction = Literal["forward", "backward"]


class GeneratedText(BaseModel, extra="forbid"):
    text: Text


@dataclass(frozen=True)
class Example:
    input: str
    label: str


@dataclass(frozen=True)
class Learner:
    instruction: str
    weight: float


class PREFER:
    """Confidence callback returns label-aligned support or elimination scores.

    Bilateral predictions feed both boosting and inference. Templates outside the
    optimized instruction belong to the callback. Generation/measurement failures
    propagate; chance-level learners are rejected and refined within the budget.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str, str, Direction], Awaitable[Sequence[float]]],
        labels: Sequence[str],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.labels = tuple(labels)

    @prompt(template="reflect.j2", output_type=GeneratedText)
    async def reflect(
        self, instruction: str, errors: list[dict], *, generated: GeneratedText
    ) -> str:
        return generated.text

    @prompt(template="refine.j2", output_type=GeneratedText)
    async def refine(self, instruction: str, reflection: str, *, generated: GeneratedText) -> str:
        return generated.text

    async def _bilateral(self, instruction, input):
        forward = tuple(await self.evaluate(instruction, input, "forward"))
        backward = tuple(await self.evaluate(instruction, input, "backward"))
        if len(forward) != len(self.labels) or len(backward) != len(self.labels):
            raise ValueError("confidence scores must align with labels")
        scores = [a - b for a, b in zip(forward, backward, strict=True)]
        if any(not math.isfinite(s) for s in (*forward, *backward, *scores)):
            raise ValueError("bilateral confidence scores must be finite")
        return self.labels[max(range(len(scores)), key=scores.__getitem__)]

    async def _assess(self, instruction, examples, weights):
        predictions = [await self._bilateral(instruction, e.input) for e in examples]
        wrong = [prediction != e.label for prediction, e in zip(predictions, examples, strict=True)]
        error = math.fsum(w for w, miss in zip(weights, wrong, strict=True) if miss)
        errors = [
            {"input": e.input, "predicted": prediction, "correct": e.label, "weight": weight}
            for e, prediction, miss, weight in zip(
                examples, predictions, wrong, weights, strict=True
            )
            if miss
        ]
        return error, wrong, errors

    def _boost(self, instruction, error, wrong, weights):
        if error >= 1 - 1 / len(self.labels):
            return weights
        alpha = math.log1p(-error) - math.log(error) + math.log(len(self.labels) - 1)
        self.ensemble.append(Learner(instruction, alpha))
        log_weights = [
            math.log(weight) + alpha * miss for weight, miss in zip(weights, wrong, strict=True)
        ]
        maximum = max(log_weights)
        updated = [math.exp(value - maximum) for value in log_weights]
        total = math.fsum(updated)
        return [value / total for value in updated]

    async def predict(self, ensemble: Sequence[Learner], input: str) -> str:
        """Combine each prompt's bilateral label with its learned ensemble weight."""
        votes = {label: 0.0 for label in self.labels}
        for learner in ensemble:
            votes[await self._bilateral(learner.instruction, input)] += learner.weight
        if not ensemble:
            raise ValueError("no informative prompt was learned")
        return max(votes, key=votes.__getitem__)

    async def run(self, initial: str, examples: Sequence[Example], *, rounds: int = 10) -> dict:
        self.ensemble = []
        reflections, errors = [], []
        weights = [1 / len(examples)] * len(examples)
        instruction = initial
        for iteration in range(rounds):
            error, wrong, evidence = await self._assess(instruction, examples, weights)
            errors.append(error)
            if error == 0:
                self.ensemble = [Learner(instruction, 1.0)]
                break
            weights = self._boost(instruction, error, wrong, weights)
            if iteration + 1 < rounds:
                reflection = await self.reflect(instruction, evidence, provider=self.provider)
                reflections.append(reflection)
                instruction = await self.refine(instruction, reflection, provider=self.provider)
        return {
            "ensemble": self.ensemble,
            "reflections": reflections,
            "instance_weights": weights,
            "errors": errors,
        }
