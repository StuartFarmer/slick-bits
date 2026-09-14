"""Optimize discrete token coordinates from caller-supplied model gradients."""

import math
from collections.abc import Awaitable, Callable, Sequence

import numpy as np


class GCG:
    """Own GCG token sampling and loss selection, independent of task/model.

    `gradient(tokens)` returns d(loss)/d(one-hot tokens), positions by vocabulary.
    `evaluate(tokens)` returns loss (lower wins). `accept` can enforce tokenizer
    round-trip length and artifact restrictions before evaluation. No generation
    prompts are involved; the caller owns differentiable model execution.
    """

    def __init__(
        self,
        gradient: Callable[[tuple[int, ...]], Awaitable[np.ndarray]],
        evaluate: Callable[[tuple[int, ...]], Awaitable[float]],
        *,
        accept: Callable[[tuple[int, ...]], bool] = lambda tokens: True,
    ):
        self.gradient = gradient
        self.evaluate = evaluate
        self.accept = accept

    async def _loss(self, tokens):
        self.evaluations += 1
        value = float(await self.evaluate(tokens))
        if not math.isfinite(value):
            raise ValueError("measured loss must be finite")
        return {"tokens": tokens, "loss": value}

    async def _sample(self, current, batch_size, top_k, forbidden_tokens):
        self.gradient_calls += 1
        grad = np.array(await self.gradient(current["tokens"]), dtype=float, copy=True)
        if not np.isfinite(grad).all():
            raise ValueError("measured token gradients must be finite")
        grad[:, list(forbidden_tokens)] = math.inf
        top = np.argsort(grad, axis=1, kind="stable")[:, :top_k]
        candidates = []
        for row in range(batch_size):
            # Match upstream's evenly distributed single-coordinate batch.
            position = row * len(current["tokens"]) // batch_size
            token = int(self.rng.choice(top[position]))
            if not math.isfinite(grad[position, token]):
                self.rejected_candidates += 1
                continue
            trial = list(current["tokens"])
            trial[position] = token
            trial = tuple(trial)
            if not self.accept(trial):
                self.rejected_candidates += 1
                continue
            candidates.append(trial)
        return candidates

    async def run(
        self,
        tokens: Sequence[int],
        *,
        iterations: int = 100,
        batch_size: int = 64,
        top_k: int = 32,
        forbidden_tokens: Sequence[int] = (),
        seed: int = 0,
    ) -> dict:
        """Sample single-coordinate replacements and take the batch loss minimum.

        The trajectory may worsen, as in upstream GCG; retain best-so-far separately.
        No candidate cache, retries, or hidden provider calls. Repeated runs reset state.
        """
        self.rng = np.random.default_rng(seed)
        self.evaluations = self.gradient_calls = self.rejected_candidates = 0
        current = best = await self._loss(tuple(tokens))
        history = [current]
        for _ in range(iterations):
            candidates = await self._sample(current, batch_size, top_k, forbidden_tokens)
            if candidates:
                assessed = [await self._loss(candidate) for candidate in candidates]
                current = min(assessed, key=lambda item: item["loss"])
                if current["loss"] < best["loss"]:
                    best = current
            history.append(current)
        return {
            "best": best,
            "current": current,
            "history": history,
            "evaluations": self.evaluations,
            "gradient_calls": self.gradient_calls,
            "rejected_candidates": self.rejected_candidates,
        }
