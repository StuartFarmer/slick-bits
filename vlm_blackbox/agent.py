"""Search text templates using contrasting best and worst evaluated prompts."""

import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Templates(BaseModel, extra="forbid"):
    analysis: Text
    templates: list[Text]


class VLMBlackBox:
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        *,
        constraints: str = "",
        required_tokens: Sequence[str] = (),
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.constraints, self.required_tokens = constraints, tuple(required_tokens)

    @prompt(template="contrast.j2", output_type=Templates)
    async def contrast(
        self, good: list[str], bad: list[str], count: int, *, generated: Templates
    ) -> list[str]:
        if len(generated.templates) != count:
            raise ValueError("wrong number of generated templates")
        if any(token not in text for text in generated.templates for token in self.required_tokens):
            raise ValueError("generated template lost a required token")
        return generated.templates

    async def assess(self, templates):
        for text in templates:
            if text not in self.archive:
                score = await self.evaluate(text)
                if not math.isfinite(score):
                    raise ValueError("evaluation must be finite")
                self.archive[text] = score

    async def run(
        self, initial: Sequence[str], *, rounds: int = 10, pool_size: int = 3, candidates: int = 1
    ) -> dict:
        self.archive = {}
        await self.assess(initial)
        for _ in range(rounds):
            ranked = sorted(self.archive, key=self.archive.get, reverse=True)
            texts = await self.contrast(
                ranked[:pool_size], ranked[-pool_size:], candidates, provider=self.provider
            )
            await self.assess(texts)
        best = max(self.archive, key=self.archive.get)
        return {
            "best": best,
            "score": self.archive[best],
            "archive": dict(self.archive),
            "evaluations": len(self.archive),
        }
