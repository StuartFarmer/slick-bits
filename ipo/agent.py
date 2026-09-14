"""Optimize interpretable class templates with visual descriptions and episodic memory."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Proposals(BaseModel, extra="forbid"):
    prompts: list[Text]


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    loss: float


@dataclass(frozen=True)
class Individual:
    prompt: str
    evaluation: Evaluation


class IPO:
    """Evaluator owns the frozen vision-language model and training-image split.

    Descriptions are caller-supplied outputs of a multimodal model. Selection
    maximizes accuracy, breaking ties by lower loss. All failures propagate.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        descriptions: Sequence[str] = (),
        class_token: str = "<CLASS>",
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.descriptions, self.class_token = descriptions, class_token

    @prompt(template="propose.j2", output_type=Proposals)
    async def propose(self, memory: list[dict], count: int, *, generated: Proposals) -> list[str]:
        if len(generated.prompts) != count:
            raise ValueError("optimizer returned the wrong number of prompts")
        if any(self.class_token not in text for text in generated.prompts):
            raise ValueError("generated template lost its class placeholder")
        return generated.prompts

    async def _assess(self, text):
        if text not in self.archive:
            measured = await self.evaluate(text)
            if not math.isfinite(measured.accuracy) or not math.isfinite(measured.loss):
                raise ValueError("accuracy and loss must be finite")
            self.archive[text] = Individual(text, measured)
        return self.archive[text]

    def _retrieve(self, size):
        memory = sorted(
            self.archive.values(),
            key=lambda p: (p.evaluation.accuracy, -p.evaluation.loss),
            reverse=True,
        )[:size]
        if self.baseline not in memory:
            memory.append(self.baseline)
        memory.sort(key=lambda p: (p.evaluation.accuracy, -p.evaluation.loss))
        return [
            {"prompt": p.prompt, "accuracy": p.evaluation.accuracy, "loss": p.evaluation.loss}
            for p in memory
        ]

    async def run(
        self,
        baseline: str = "a photo of <CLASS>",
        *,
        rounds: int = 100,
        candidates: int = 5,
        memory_size: int = 20,
    ) -> dict:
        self.archive = {}
        self.baseline = await self._assess(baseline)
        for _ in range(rounds):
            memory = self._retrieve(memory_size)
            proposals = await self.propose(memory, candidates, provider=self.provider)
            for text in proposals:
                await self._assess(text)
        best = max(self.archive.values(), key=lambda p: (p.evaluation.accuracy, -p.evaluation.loss))
        return {
            "best": best,
            "archive": list(self.archive.values()),
            "evaluations": len(self.archive),
        }
