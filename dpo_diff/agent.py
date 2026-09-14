"""Optimize compact categorical prompt spaces with Gumbel-softmax and RMSprop."""

import math
from collections.abc import Awaitable, Callable, Sequence

import numpy as np


class DPODiff:
    """Port DPO-Diff's gradient optimizer independently of its diffusion backend.

    Each slot contains alternative text fragments and their aligned embeddings.
    `gradient(mixed_embeddings, timestep)` supplies shortcut text gradients with
    respect to each mixed embedding. The local algorithm differentiates the
    Gumbel-softmax mixture and updates categorical logits. `evaluate(parts)`
    measures a discrete candidate's loss; lower is better.
    """

    def __init__(
        self,
        task: str,
        gradient: Callable[[list[np.ndarray], int], Awaitable[Sequence[np.ndarray]]],
        evaluate: Callable[[tuple[str, ...]], Awaitable[float]],
    ):
        self.task, self.gradient, self.evaluate = task, gradient, evaluate

    def _mixtures(self, temperature):
        weights = []
        for logits in self.logits:
            noisy = (logits + self.rng.gumbel(size=logits.shape)) / temperature
            exp = np.exp(noisy - noisy.max())
            weights.append(exp / exp.sum())
        return weights

    async def _update(self, embeddings, timestep, temperature, learning_rate):
        weights = self._mixtures(temperature)
        mixed = [np.tensordot(w, emb, axes=1) for w, emb in zip(weights, embeddings)]
        self.gradient_calls += 1
        gradients = await self.gradient(mixed, timestep)
        for i, (w, emb, gradient) in enumerate(zip(weights, embeddings, gradients, strict=True)):
            gradient = np.asarray(gradient)
            if not np.isfinite(gradient).all():
                raise ValueError("measured embedding gradients must be finite")
            dw = emb.reshape(len(w), -1) @ gradient.reshape(-1)
            grad = np.clip(w * (dw - w @ dw) / temperature, -0.025, 0.025)
            self.squares[i] = 0.99 * self.squares[i] + 0.01 * grad**2
            self.momentum[i] = 0.5 * self.momentum[i] + grad / (np.sqrt(self.squares[i]) + 1e-8)
            self.logits[i] = np.clip(self.logits[i] - learning_rate * self.momentum[i], 0, 3)

    async def _sample(self, slots, samples, max_attempts):
        seen, assessed = set(), []
        for _ in range(max_attempts):
            if len(assessed) >= samples:
                break
            self.sample_attempts += 1
            ids = tuple(int(np.argmax(w)) for w in self._mixtures(1.0))
            if ids in seen:
                continue
            seen.add(ids)
            parts = tuple(slot[i] for slot, i in zip(slots, ids))
            self.evaluations += 1
            loss = float(await self.evaluate(parts))
            if not math.isfinite(loss):
                raise ValueError("measured discrete loss must be finite")
            assessed.append({"parts": parts, "ids": ids, "loss": loss})
        return assessed

    async def run(
        self,
        slots: Sequence[Sequence[str]],
        embeddings: Sequence[np.ndarray],
        *,
        iterations: int = 50,
        samples: int = 20,
        timestep: int = 25,
        temperature: float = 1.0,
        learning_rate: float = 0.1,
        max_sample_attempts: int = 1000,
        seed: int = 0,
    ) -> dict:
        """Train the categorical distributions, then sample and score unique prompts.

        Alternative zero in every slot is the original fragment. The initial logit
        for it is 1; others start at 0. A bounded sample-attempt budget prevents
        infinite loops when requested samples exceed the discrete domain size.
        """
        self.rng = np.random.default_rng(seed)
        self.gradient_calls = self.evaluations = self.sample_attempts = 0
        self.logits = [np.r_[1.0, np.zeros(len(slot) - 1)] for slot in slots]
        self.squares = [np.zeros_like(x) for x in self.logits]
        self.momentum = [np.zeros_like(x) for x in self.logits]
        for _ in range(iterations):
            await self._update(embeddings, timestep, temperature, learning_rate)
        candidates = await self._sample(slots, samples, max_sample_attempts)
        return {
            "best": min(candidates, key=lambda item: item["loss"]),
            "candidates": candidates,
            "logits": [x.copy() for x in self.logits],
            "gradient_calls": self.gradient_calls,
            "evaluations": self.evaluations,
            "sample_attempts": self.sample_attempts,
        }
