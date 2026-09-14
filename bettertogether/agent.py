"""Alternate prompt search and weight optimization on caller-owned programs."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from typing import Any


class BetterTogether:
    """Compose existing Slick optimizers with a model fine-tuning implementation.

    Each optimizer receives (program, training_examples), returns a new program.
    `evaluate` measures held-out quality; larger is better. `clone` must preserve
    independent checkpoints, including model identity, while allowing provider
    handles to be shared when they refer to immutable model versions.
    """

    def __init__(
        self,
        task: str,
        prompt_optimize: Callable[[Any, tuple], Awaitable[Any]],
        weight_optimize: Callable[[Any, tuple], Awaitable[Any]],
        evaluate: Callable[[Any], Awaitable[float]],
        *,
        clone: Callable[[Any], Any] = deepcopy,
    ):
        self.task, self.evaluate, self.clone = task, evaluate, clone
        self.optimizers = {"p": prompt_optimize, "w": weight_optimize}

    async def _measure(self, program, strategy):
        self.evaluations += 1
        score = float(await self.evaluate(self.clone(program)))
        if not math.isfinite(score):
            raise ValueError("measured validation score must be finite")
        return {"program": self.clone(program), "score": score, "strategy": strategy}

    async def run(
        self,
        program: Any,
        training: Sequence,
        *,
        strategy: tuple[str, ...] = ("p", "w", "p"),
        seed=0,
    ) -> dict:
        current = self.clone(program)
        examples = list(training)
        rng = random.Random(seed)
        self.evaluations = 0
        history = [await self._measure(current, ())]
        for index, step in enumerate(strategy):
            rng.shuffle(examples)
            current = await self.optimizers[step](self.clone(current), tuple(examples))
            history.append(await self._measure(current, strategy[: index + 1]))
        return {
            "best": max(history, key=lambda item: item["score"]),
            "history": history,
            "evaluations": self.evaluations,
        }
