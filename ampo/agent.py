"""Analyze error patterns, revise conditional branches, and prune each proposal."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel
from slick import prompt
from slick.providers import Provider


def checked(text):
    if not text.strip():
        raise ValueError(f"empty generated text; response={text!r}")
    return text.strip()


class Pattern(BaseModel):
    description: str
    importance: float


class Patterns(BaseModel):
    patterns: list[Pattern]


@dataclass
class Candidate:
    prompt: str
    score: float


class AMPO:
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        failures: Callable[[str], Awaitable[Sequence[dict]]],
    ):
        self.task, self.provider = task, provider
        self.evaluate, self.failures = evaluate, failures

    @prompt(template="analyze.j2")
    async def analyze(self, instruction: str, failure: dict, *, generated: str) -> str:
        return checked(generated)

    @prompt(template="patterns.j2", output_type=Patterns)
    async def summarize(self, reasons: list[str], *, generated: Patterns) -> list[Pattern]:
        for pattern in generated.patterns:
            if not pattern.description.strip() or not math.isfinite(pattern.importance):
                raise ValueError(f"invalid generated pattern: {generated!r}")
        return generated.patterns

    @prompt(template="revise.j2")
    async def revise(self, instruction: str, pattern: Pattern, *, generated: str) -> str:
        return checked(generated)

    @prompt(template="prune.j2")
    async def prune(self, instruction: str, *, generated: str) -> str:
        return checked(generated)

    async def _measure(self, text):
        self.evaluations += 1
        score = float(await self.evaluate(text))
        if not math.isfinite(score):
            raise ValueError("fitness must be finite")
        return Candidate(text, score)

    async def _propose(self, current, batch_size, branches):
        failures = list(await self.failures(current.prompt))
        batch = self.rng.sample(failures, min(batch_size, len(failures)))
        if not batch:
            return []
        reasons = []
        for failure in batch:
            self.optimizer_calls += 1
            reasons.append(await self.analyze(current.prompt, failure, provider=self.provider))
        self.optimizer_calls += 1
        patterns = await self.summarize(reasons, provider=self.provider)
        selected = sorted(patterns, key=lambda pattern: pattern.importance, reverse=True)[:branches]
        children = []
        for pattern in selected:
            self.optimizer_calls += 1
            expanded = await self.revise(current.prompt, pattern, provider=self.provider)
            self.optimizer_calls += 1
            pruned = await self.prune(expanded, provider=self.provider)
            child = await self._measure(pruned)
            children.append(child)
            self.history.append(
                {
                    "parent": current.prompt,
                    "pattern": pattern,
                    "expanded": expanded,
                    "pruned": child,
                }
            )
        return children

    async def run(
        self,
        initial_prompt: str,
        *,
        iterations: int = 5,
        batch_size: int = 8,
        branches: int = 3,
        pre_prune: bool = True,
        min_gain: float = 0,
        seed: int = 0,
    ) -> dict:
        self.rng, self.history = random.Random(seed), []
        self.evaluations = self.optimizer_calls = 0
        current = best = await self._measure(initial_prompt)
        for _ in range(iterations):
            children = await self._propose(current, batch_size, branches)
            if not children:
                break
            chosen = max(children, key=lambda item: item.score)
            gain = chosen.score - current.score
            current = chosen
            if current.score > best.score:
                best = current
            if pre_prune and gain <= min_gain:
                break
        return {
            "best": best,
            "current": current,
            "history": self.history,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
        }
