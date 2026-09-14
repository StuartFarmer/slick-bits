"""Aggregate robust training contexts for shared-token GCG optimization."""

import math
from collections.abc import Awaitable, Callable, Sequence

import numpy as np

from gcg import GCG


class RPO:
    """Optimize a shared suffix against caller-owned model/context objectives.

    Contexts describe training model/prompt/perturbation combinations. The caller
    defines desired-response and optional control losses. All contexts share the
    same token vocabulary; distinct-tokenizer proposal groups are not supported.
    """

    def __init__(
        self,
        task: str,
        contexts: Sequence[object],
        gradient: Callable[[tuple[int, ...], object], Awaitable[np.ndarray]],
        evaluate: Callable[[tuple[int, ...], object], Awaitable[float]],
        *,
        accept: Callable[[tuple[int, ...]], bool] = lambda tokens: True,
    ):
        self.task, self.contexts = task, tuple(contexts)
        self.gradient, self.evaluate, self.accept = gradient, evaluate, accept

    async def _gradient(self, tokens):
        gradients = []
        for context in self.contexts:
            self.model_gradient_calls += 1
            grad = np.asarray(await self.gradient(tokens, context), dtype=float)
            if not np.isfinite(grad).all():
                raise ValueError("measured gradients must be finite")
            # Scaling first preserves directions when squaring raw values would
            # overflow or underflow, while zero rows still stay zero.
            scales = np.max(np.abs(grad), axis=-1, keepdims=True)
            scaled = grad / np.where(scales == 0, 1, scales)
            norms = np.linalg.norm(scaled, axis=-1, keepdims=True)
            gradients.append(scaled / np.where(norms == 0, 1, norms))
        return np.stack(gradients).sum(axis=0)

    async def _loss(self, tokens):
        losses = []
        for context in self.contexts:
            self.model_evaluations += 1
            value = float(await self.evaluate(tokens, context))
            if not math.isfinite(value):
                raise ValueError("measured context loss must be finite")
            losses.append(value)
        return sum(loss / len(losses) for loss in losses)

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
        """Aggregate normalized gradients, sample coordinates, and minimize mean loss."""
        self.model_evaluations = self.model_gradient_calls = 0
        self.search = GCG(self._gradient, self._loss, accept=self.accept)
        result = await self.search.run(
            tokens,
            iterations=iterations,
            batch_size=batch_size,
            top_k=top_k,
            forbidden_tokens=forbidden_tokens,
            seed=seed,
        )
        return {
            **result,
            "model_evaluations": self.model_evaluations,
            "model_gradient_calls": self.model_gradient_calls,
        }
