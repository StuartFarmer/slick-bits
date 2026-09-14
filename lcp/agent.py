"""Contrast high- and low-performing instructions after diverse error induction."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from slick import prompt
from slick.providers import Provider


def checked(text):
    if not text.strip():
        raise ValueError(f"empty generated text; response={text!r}")
    return text.strip()


@dataclass
class Candidate:
    prompt: str
    score: float


class LCP:
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        failures: Callable[[str], Awaitable[Sequence[dict]]],
    ):
        self.task, self.provider = task, provider
        self.evaluate, self.failures = evaluate, failures

    @prompt(template="reason.j2")
    async def reason(self, instruction: str, failure: dict, *, generated: str) -> str:
        return checked(generated)

    @prompt(template="induce.j2")
    async def induce(self, instruction: str, reasons: list[dict], *, generated: str) -> str:
        return checked(generated)

    @prompt(template="contrast.j2")
    async def contrast(self, good: list[Candidate], bad: list[Candidate], *, generated: str) -> str:
        return checked(generated)

    async def _measure(self, text):
        if text not in self.archive:
            self.evaluations += 1
            score = float(await self.evaluate(text))
            if not math.isfinite(score):
                raise ValueError("fitness must be finite")
            self.archive[text] = Candidate(text, score)
        return self.archive[text]

    async def _diversify(self, current, count, batch_size, adaptation):
        rows = list(await self.failures(current.prompt))
        if adaptation:
            rows = [row for row in rows if row["source_correct"]]
        reasons = []
        for row in rows:
            self.optimizer_calls += 1
            reason = await self.reason(current.prompt, row, provider=self.provider)
            reasons.append({"failure": row, "reason": reason})
        for _ in range(count if reasons else 0):
            batch = self.rng.sample(reasons, min(batch_size, len(reasons)))
            self.optimizer_calls += 1
            text = await self.induce(current.prompt, batch, provider=self.provider)
            await self._measure(text)
        return bool(reasons)

    async def run(
        self,
        initial_prompt: str,
        *,
        iterations: int = 5,
        diversity: int = 10,
        batch_size: int = 4,
        top_k: int = 3,
        adaptation: bool = False,
        seed: int = 0,
    ) -> dict:
        self.archive, self.history, self.rng = {}, [], random.Random(seed)
        self.evaluations = self.optimizer_calls = 0
        current = await self._measure(initial_prompt)
        for _ in range(iterations):
            if not await self._diversify(current, diversity, batch_size, adaptation):
                break
            ranked = sorted(self.archive.values(), key=lambda item: item.score, reverse=True)
            count = min(top_k, len(ranked) // 2)
            if not count:
                break
            good, bad = ranked[:count], ranked[-count:]
            self.optimizer_calls += 1
            text = await self.contrast(good, bad, provider=self.provider)
            current = await self._measure(text)
            self.history.append({"good": good, "bad": bad, "selected": current})
        return {
            "best": max(self.archive.values(), key=lambda item: item.score),
            "current": current,
            "archive": list(self.archive.values()),
            "history": self.history,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
        }
