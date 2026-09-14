"""Optimize latent prompt embeddings with gradients at discrete projections."""

import math
from collections.abc import Awaitable, Callable

import numpy as np


class PEZ:
    """Hard Prompts Made Easy's straight-through nearest-token projection.

    `gradient(projected_embeddings)` computes the model loss gradient at the hard
    prompt, not at the continuous latent values. `evaluate(token_ids)` measures
    discrete loss. Both are async and caller-owned; lower measured loss wins.
    """

    def __init__(
        self,
        task: str,
        vocabulary: np.ndarray,
        gradient: Callable[[np.ndarray], Awaitable[np.ndarray]],
        evaluate: Callable[[tuple[int, ...]], Awaitable[float]],
    ):
        self.task = task
        self.vocabulary = np.array(vocabulary, dtype=float, copy=True)
        self.gradient, self.evaluate = gradient, evaluate

    def _project(self, latent):
        norm = np.linalg.norm(latent, axis=-1, keepdims=True)
        queries = latent / np.where(norm == 0, 1, norm)
        # ponytail: dense vocabulary scan; chunk it if prompt*vocabulary memory matters.
        ids = np.argmax(queries @ self.normalized.T, axis=-1)
        return tuple(int(i) for i in ids), self.vocabulary[ids].copy()

    async def _measure(self, ids):
        self.evaluations += 1
        loss = float(await self.evaluate(ids))
        if not math.isfinite(loss):
            raise ValueError("measured loss must be finite")
        return {"tokens": ids, "loss": loss}

    async def run(
        self,
        initial_embeddings: np.ndarray,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        weight_decay: float = 0.01,
    ) -> dict:
        """Project, measure, differentiate at the projection, and update latent AdamW."""
        latent = np.array(initial_embeddings, dtype=float, copy=True)
        norms = np.linalg.norm(self.vocabulary, axis=-1, keepdims=True)
        self.normalized = self.vocabulary / np.where(norms == 0, 1, norms)
        self.gradient_calls = self.evaluations = 0
        first, second = np.zeros_like(latent), np.zeros_like(latent)
        ids, projected = self._project(latent)
        best = await self._measure(ids)
        history = [best]
        for step in range(1, iterations + 1):
            self.gradient_calls += 1
            grad = np.asarray(await self.gradient(projected), dtype=float)
            if not np.isfinite(grad).all():
                raise ValueError("measured embedding gradients must be finite")
            first = 0.9 * first + 0.1 * grad
            second = 0.999 * second + 0.001 * grad**2
            latent *= 1 - learning_rate * weight_decay
            latent -= (
                learning_rate
                * (first / (1 - 0.9**step))
                / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
            ids, projected = self._project(latent)
            candidate = await self._measure(ids)
            history.append(candidate)
            if candidate["loss"] < best["loss"]:
                best = candidate
        return {
            "best": best,
            "latent": latent,
            "history": history,
            "gradient_calls": self.gradient_calls,
            "evaluations": self.evaluations,
        }
