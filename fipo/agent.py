"""Compose FIPO's optional response modules for its trained prompt optimizer."""

import math
from collections.abc import Awaitable, Callable

from slick import prompt
from slick.providers import Provider


def _instruction(generated):
    result = generated.strip().removeprefix("Golden Prompt:").strip()
    if not result:
        raise ValueError(f"empty optimized instruction; response={generated!r}")
    return result


class FIPO:
    """Serve the authors' FIPO checkpoint through `provider`; score externally.

    This is the deployment algorithm, not the POP dataset or preference trainer.
    The caller provides optional observed and reference responses. Errors propagate.
    """

    def __init__(self, task: str, provider: Provider, evaluate: Callable[[str], Awaitable[float]]):
        self.task, self.provider, self.evaluate = task, provider, evaluate

    @prompt(template="rewrite.j2")
    async def rewrite(self, max_words: int, *, generated: str) -> str:
        """Optimize without response evidence."""
        return _instruction(generated)

    @prompt(template="response.j2")
    async def with_response(self, response: str, max_words: int, *, generated: str) -> str:
        """Optimize with an observed response."""
        return _instruction(generated)

    @prompt(template="reference.j2")
    async def with_reference(self, reference: str, max_words: int, *, generated: str) -> str:
        """Optimize with a reference response."""
        return _instruction(generated)

    @prompt(template="both.j2")
    async def with_both(
        self, response: str, reference: str, max_words: int, *, generated: str
    ) -> str:
        """Optimize with observed and reference responses."""
        return _instruction(generated)

    async def _rewrite(self, response, reference, max_words):
        execution = {"provider": self.provider}
        if response is not None and reference is not None:
            return await self.with_both(response, reference, max_words, **execution)
        if response is not None:
            return await self.with_response(response, max_words, **execution)
        if reference is not None:
            return await self.with_reference(reference, max_words, **execution)
        return await self.rewrite(max_words, **execution)

    async def run(
        self, *, response: str | None = None, reference: str | None = None, max_words: int = 200
    ) -> dict:
        """Select modules in Python, rewrite once, and measure the result."""
        instruction = await self._rewrite(response, reference, max_words)
        score = float(await self.evaluate(instruction))
        if not math.isfinite(score):
            raise ValueError("measured score must be finite")
        return {"prompt": instruction, "score": score, "optimizer_calls": 1, "evaluations": 1}
