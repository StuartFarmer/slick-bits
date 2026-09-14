"""Search rounded n-gram coordinates with a Gaussian-process UCB surrogate."""

import math
from collections.abc import Awaitable, Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from instructzero.agent import matern52


class BayesianPrompt:
    """Sabbatella et al.'s GP search over a supplied finite n-gram vocabulary.

    Training scores drive acquisition; separate validation scores select the
    returned prompt. Both callbacks receive text and prefer higher finite scores.
    No model generation or soft-token access is required for this discrete path.
    """

    def __init__(
        self,
        task: str,
        ngrams: Sequence[str],
        evaluate: Callable[[str], Awaitable[float]],
        validate: Callable[[str], Awaitable[float]],
    ):
        self.task, self.ngrams = task, tuple(ngrams)
        self.evaluate, self.validate = evaluate, validate

    async def _observe(self, point):
        ids = tuple(int(i) for i in np.rint(point))
        text = " ".join(self.ngrams[i] for i in ids)
        self.evaluations += 1
        score = float(await self.evaluate(text))
        if not math.isfinite(score):
            raise ValueError("measured training score must be finite")
        self.validations += 1
        validation = float(await self.validate(text))
        if not math.isfinite(validation):
            raise ValueError("measured validation score must be finite")
        self.history.append(
            {"prompt": text, "ids": ids, "score": score, "validation_score": validation}
        )

    def _acquire(self, length, beta, length_scale, noise, restarts, raw_samples):
        x = np.asarray([entry["ids"] for entry in self.history], dtype=float)
        values = np.array([entry["score"] for entry in self.history])
        mean = values.mean()
        scale = values.std() or 1.0
        y = (values - mean) / scale
        covariance = matern52(x, x, length_scale) + noise * np.eye(len(x))
        factor = np.linalg.cholesky(covariance)
        alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))

        def ucb(point):
            cross = matern52(np.asarray(point).reshape(1, -1), x, length_scale).ravel()
            solved = np.linalg.solve(factor, cross)
            variance = max(0.0, 1 - solved @ solved)
            return float(cross @ alpha + math.sqrt(beta * variance))

        starts = self.rng.uniform(0, len(self.ngrams) - 1, (raw_samples, length))
        starts = sorted(starts, key=ucb, reverse=True)[:restarts]
        candidates = list(starts)
        for start in starts:
            result = minimize(
                lambda p: -ucb(p),
                start,
                method="L-BFGS-B",
                bounds=[(0, len(self.ngrams) - 1)] * length,
            )
            candidates.append(result.x)
        return np.rint(max(candidates, key=ucb)).astype(int)

    async def run(
        self,
        *,
        length: int = 6,
        initial_samples: int = 10,
        iterations: int = 30,
        beta: float = 0.4,
        length_scale: float = 1.0,
        noise: float = 1e-4,
        restarts: int = 5,
        raw_samples: int = 50,
        seed: int = 0,
    ) -> dict:
        """Observe random coordinates, optimize UCB, round, and measure both splits."""
        self.rng = np.random.default_rng(seed)
        self.history = []
        self.evaluations = self.validations = 0
        for point in self.rng.integers(len(self.ngrams), size=(initial_samples, length)):
            await self._observe(point)
        for _ in range(iterations):
            point = self._acquire(length, beta, length_scale, noise, restarts, raw_samples)
            await self._observe(point)
        return {
            "best": max(self.history, key=lambda item: item["validation_score"]),
            "history": self.history.copy(),
            "evaluations": self.evaluations,
            "validations": self.validations,
        }
