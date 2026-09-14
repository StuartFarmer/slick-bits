"""Rank demonstration permutations by prediction entropy on synthetic inputs."""

import itertools
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Synthetic(BaseModel, extra="forbid"):
    inputs: list[Text] = Field(min_length=1)


@dataclass(frozen=True)
class Example:
    input: str
    label: str


class OrderedPrompts:
    """Own synthetic development data, permutation enumeration and global entropy ranking.

    classify receives an ordered demonstration tuple and synthetic input strings,
    returning finite class scores. Gold validation labels never influence ranking.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Sequence[Example]], Awaitable[float]],
        classify: Callable[[Sequence[Example], Sequence[str]], Awaitable[np.ndarray]],
    ):
        self.task, self.provider, self.evaluate, self.classify = task, provider, evaluate, classify

    @prompt(template="generate.j2", output_type=Synthetic)
    async def generate(
        self, demonstrations: Sequence[Example], *, generated: Synthetic
    ) -> list[str]:
        return generated.inputs

    @prompt(template="answer.j2")
    async def answer(self, input: str, *, generated: str) -> str:
        return generated

    async def run(
        self,
        examples: Sequence[Example],
        *,
        classes: int,
        max_permutations: int = 24,
        keep: int = 4,
    ) -> dict:
        orders = list(
            itertools.islice(itertools.permutations(range(len(examples))), max_permutations)
        )
        self.generations = self.classifications = self.evaluations = 0
        synthetic = []
        for order in orders:
            self.generations += 1
            synthetic.extend(
                await self.generate(tuple(examples[i] for i in order), provider=self.provider)
            )
        synthetic = list(dict.fromkeys(synthetic))
        ranked = []
        for order in orders:
            self.classifications += 1
            scores = np.asarray(await self.classify(tuple(examples[i] for i in order), synthetic))
            if not np.isfinite(scores).all():
                raise ValueError("classification scores must be finite")
            frequencies = np.bincount(np.argmax(scores, axis=1), minlength=classes).astype(float)
            frequencies[frequencies == 0] = 1e-5
            distribution = frequencies / frequencies.sum()
            entropy = -float(distribution @ np.log(distribution))
            ranked.append({"order": order, "entropy": entropy, "distribution": distribution})
        ranked.sort(key=lambda row: row["entropy"], reverse=True)
        self.demonstrations = tuple(examples[i] for i in ranked[0]["order"])
        self.evaluations += 1
        score = float(await self.evaluate(self.demonstrations))
        if not math.isfinite(score):
            raise ValueError("selected prompt score must be finite")
        return {
            "demonstrations": self.demonstrations,
            "selected": ranked[:keep],
            "ranked": ranked,
            "synthetic": synthetic,
            "score": score,
            "generations": self.generations,
            "classifications": self.classifications,
            "evaluations": self.evaluations,
        }
