"""Feedback refinement followed by preference-history alignment."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Example:
    input: str
    reference: str


class APEER:
    def __init__(
        self,
        task: str,
        provider: Provider,
        respond: Callable[[str, str], Awaitable[str]],
        evaluate: Callable[[str], Awaitable[float]],
    ):
        self.task, self.provider = task, provider
        self.respond, self.evaluate = respond, evaluate

    @prompt(template="feedback.j2")
    async def feedback(
        self, *, instruction: str, example: Example, response: str, generated: str
    ) -> str:
        if not generated.strip():
            raise ValueError(f"empty feedback; response={generated!r}")
        return generated.strip()

    @prompt(template="refine.j2")
    async def refine(self, *, instruction: str, feedbacks: list[str], generated: str) -> str:
        if not generated.strip():
            raise ValueError(f"empty refinement; response={generated!r}")
        return generated.strip()

    @prompt(template="prefer.j2")
    async def prefer(self, *, instruction: str, pairs: list[dict], generated: str) -> str:
        if not generated.strip():
            raise ValueError(f"empty preference rewrite; response={generated!r}")
        return generated.strip()

    async def _measure(self, instruction):
        self.evaluations += 1
        score = float(await self.evaluate(instruction))
        if not math.isfinite(score):
            raise ValueError("measured score must be finite")
        return {"prompt": instruction, "score": score}

    async def _record(self, instruction):
        record = await self._measure(instruction)
        destination = self.positive if record["score"] > self.baseline else self.negative
        destination.append(record)
        return record

    async def _feedback_step(self, current, batch):
        feedbacks = []
        for example in batch:
            self.response_calls += 1
            response = await self.respond(current, example.input)
            self.optimizer_calls += 1
            feedbacks.append(
                await self.feedback(
                    instruction=current, example=example, response=response, provider=self.provider
                )
            )
        self.optimizer_calls += 1
        refined = await self.refine(
            instruction=current, feedbacks=feedbacks, provider=self.provider
        )
        return await self._record(refined)

    async def _preference_step(self, refined, pair_count):
        good = sorted(self.positive, key=lambda item: item["score"], reverse=True)[:pair_count]
        bad = sorted(self.negative, key=lambda item: item["score"])[:pair_count]
        pairs = [{"preferred": p, "dispreferred": n} for p, n in zip(good, bad)]
        self.optimizer_calls += 1
        aligned = await self.prefer(
            instruction=refined["prompt"], pairs=pairs, provider=self.provider
        )
        return await self._record(aligned)

    async def run(
        self,
        initial_positive: str,
        initial_negative: str,
        examples: Sequence[Example],
        *,
        iterations=3,
        batch_size=1,
        pair_count=1,
        seed=0,
    ) -> dict:
        """Reference judgments reach feedback generation, never the response callback."""
        self.optimizer_calls = self.response_calls = self.evaluations = 0
        self.positive = [await self._measure(initial_positive)]
        self.negative = [await self._measure(initial_negative)]
        self.baseline = self.positive[0]["score"]
        rng = random.Random(seed)
        history = []
        for _ in range(iterations):
            current = max(self.positive, key=lambda item: item["score"])["prompt"]
            batch = rng.sample(list(examples), min(batch_size, len(examples)))
            refined = await self._feedback_step(current, batch)
            aligned = await self._preference_step(refined, pair_count)
            history.append({"feedback": refined, "preference": aligned})
        return {
            "best": max(self.positive, key=lambda item: item["score"]),
            "positive": self.positive,
            "negative": self.negative,
            "history": history,
            "optimizer_calls": self.optimizer_calls,
            "response_calls": self.response_calls,
            "evaluations": self.evaluations,
        }
