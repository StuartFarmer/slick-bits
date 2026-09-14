"""Run BPO's learned prompt rewriter through an injected Slick provider."""

import math
from collections.abc import Awaitable, Callable

from slick import prompt
from slick.providers import Provider


class BPO:
    """The provider must serve the trained BPO optimizer for paper inference.

    Provider sampling settings belong to the caller. One sample is the paper's
    stable inference path. Multiple independent rewrites plus external selection
    are a task-agnostic extension; no optimizer training happens here.
    """

    def __init__(self, task: str, provider: Provider, evaluate: Callable[[str], Awaitable[float]]):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate

    @prompt(template="rewrite.j2")
    async def rewrite(self, *, generated: str) -> str:
        """Check the learned optimizer's plain text response."""
        result = generated.strip()
        if not result:
            raise ValueError(f"empty optimized instruction; response={generated!r}")
        return result

    async def _assess(self, instruction):
        self.evaluations += 1
        score = float(await self.evaluate(instruction))
        if not math.isfinite(score):
            raise ValueError("measured score must be finite")
        return {"prompt": instruction, "score": score}

    async def run(self, *, samples: int = 1) -> dict:
        """Rewrite the original task independently, evaluate, and select the best."""
        self.optimizer_calls = self.evaluations = 0
        self.candidates = []
        for _ in range(samples):
            self.optimizer_calls += 1
            instruction = await self.rewrite(provider=self.provider)
            self.candidates.append(await self._assess(instruction))
        return {
            "best": max(self.candidates, key=lambda item: item["score"]),
            "candidates": self.candidates.copy(),
            "optimizer_calls": self.optimizer_calls,
            "evaluations": self.evaluations,
        }
