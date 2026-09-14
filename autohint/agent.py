"""Enrich instructions by inducing and summarizing hints from residual mistakes."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from slick import prompt
from slick.providers import Provider


@dataclass
class Example:
    input: str
    target: str


@dataclass
class Hint:
    input: str
    target: str
    prediction: str
    hint: str


class AutoHint:
    """Implement Algorithm 1 from the paper; no official implementation was located.

    Hints are generated for every incorrect training example before sampling.
    Optional cluster(hints) supplies group IDs using the caller's chosen encoder
    and clustering; balanced sampling groups by ground-truth target instead.
    Validation evaluation is separate from training residual inference.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        correct: Callable[[str, str], bool] = lambda prediction, target: prediction == target,
        cluster: Callable[[Sequence[Hint]], Sequence[int]] | None = None,
        actor_provider: Provider | None = None,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.correct, self.cluster = correct, cluster
        self.actor_provider = provider if actor_provider is None else actor_provider

    @prompt(template="infer.j2")
    async def infer(self, instruction: str, input: str, *, generated: str) -> str:
        return generated.strip()

    @prompt(template="generate_hint.j2")
    async def generate_hint(
        self, instruction: str, input: str, target: str, *, generated: str
    ) -> str:
        if not generated.strip():
            raise ValueError(f"empty hint; response={generated!r}")
        return generated.strip()

    @prompt(template="summarize.j2")
    async def summarize(self, instruction: str, hints: list[dict], *, generated: str) -> str:
        if not generated.strip():
            raise ValueError(f"empty hint summary; response={generated!r}")
        return generated.strip()

    async def _residual_hints(self, instruction, examples):
        residual = []
        for example in examples:
            self.inference_calls += 1
            prediction = await self.infer(instruction, example.input, provider=self.actor_provider)
            if not self.correct(prediction, example.target):
                residual.append((example, prediction))
        hints = []
        for example, prediction in residual:
            self.optimizer_calls += 1
            hint = await self.generate_hint(
                instruction, example.input, example.target, provider=self.provider
            )
            hints.append(Hint(example.input, example.target, prediction, hint))
        return hints

    def _sample(self, hints, sampling, sample_size):
        if sampling == "random":
            return self.rng.sample(hints, min(sample_size, len(hints)))
        keys = self.cluster(hints) if sampling == "cluster" else [hint.target for hint in hints]
        groups = {}
        for key, hint in zip(keys, hints):
            groups.setdefault(key, []).append(hint)
        return [
            hint
            for group in groups.values()
            for hint in self.rng.sample(group, min(sample_size, len(group)))
        ]

    async def _score(self, text):
        self.evaluations += 1
        score = float(await self.evaluate(text))
        if not math.isfinite(score):
            raise ValueError("fitness must be finite")
        return {"prompt": text, "score": score}

    async def run(
        self,
        initial_prompt: str,
        examples: Sequence[Example],
        *,
        iterations: int = 1,
        sampling: Literal["random", "balanced", "cluster"] = "random",
        sample_size: int = 5,
        seed: int = 0,
    ) -> dict:
        self.rng, self.history = random.Random(seed), []
        self.evaluations = self.inference_calls = self.optimizer_calls = 0
        current = best = await self._score(initial_prompt)
        for step in range(iterations):
            hints = await self._residual_hints(current["prompt"], examples)
            if not hints:
                break
            selected = self._sample(hints, sampling, sample_size)
            self.optimizer_calls += 1
            summary = await self.summarize(
                current["prompt"], [vars(hint) for hint in selected], provider=self.provider
            )
            text = current["prompt"] + "\n\nHint: " + summary
            current = await self._score(text)
            if current["score"] > best["score"]:
                best = current
            self.history.append(
                {
                    "step": step,
                    "hints": hints,
                    "selected": selected,
                    "summary": summary,
                    "current": current.copy(),
                }
            )
        return {
            "best": best,
            "current": current,
            "history": self.history.copy(),
            "evaluations": self.evaluations,
            "inference_calls": self.inference_calls,
            "optimizer_calls": self.optimizer_calls,
        }
