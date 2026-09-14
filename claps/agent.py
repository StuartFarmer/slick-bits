"""Cluster vocabulary embeddings, prune by influence, then greedily search tokens."""

from collections.abc import Awaitable, Callable

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.special import rel_entr


class ClaPS:
    def __init__(
        self,
        task: str,
        embeddings: np.ndarray,
        probabilities: Callable[[tuple[int, ...]], Awaitable[np.ndarray]],
        evaluate: Callable[[tuple[int, ...]], Awaitable[float]],
    ):
        self.task, self.embeddings = task, embeddings
        self.probabilities, self.evaluate = probabilities, evaluate

    def _representatives(self, count, rng, iterations):
        if count >= len(self.embeddings):
            return np.arange(len(self.embeddings))
        centers, _ = kmeans2(self.embeddings, count, iter=iterations, minit="++", rng=rng)
        distances = np.sum((centers[:, None] - self.embeddings[None, :]) ** 2, axis=-1)
        return np.unique(np.argmin(distances, axis=1))

    async def run(
        self,
        *,
        clusters: int = 2000,
        percentile: float = 90,
        prompt_length: int = 5,
        clustering_iterations: int = 20,
        influence: str = "kl",
        seed: int = 0,
    ) -> dict:
        rng = np.random.default_rng(seed)
        representatives = self._representatives(clusters, rng, clustering_iterations)
        self.model_calls = self.evaluations = 0
        values = []
        if influence == "kl":
            self.model_calls += 1
            baseline = np.asarray(await self.probabilities(()))
        else:
            self.evaluations += 1
            baseline = float(await self.evaluate(()))
        for token in representatives:
            if influence == "kl":
                self.model_calls += 1
                probabilities = np.asarray(await self.probabilities((int(token),)))
                value = float(rel_entr(probabilities, baseline).sum())
            else:
                self.evaluations += 1
                value = float(await self.evaluate((int(token),))) - baseline
            if not np.isfinite(value):
                raise ValueError("measured token influence must be finite")
            values.append(value)
        cutoff = np.percentile(values, percentile)
        retained = representatives[np.asarray(values) > cutoff]
        self.history = []
        current, score = (), None
        for _ in range(prompt_length if len(retained) else 0):
            proposals = []
            for token in retained:
                candidate = (*current, int(token))
                self.evaluations += 1
                value = float(await self.evaluate(candidate))
                if not np.isfinite(value):
                    raise ValueError("measured prompt reward must be finite")
                proposals.append((candidate, value))
            current, score = max(proposals, key=lambda item: item[1])
            self.history.append({"proposals": proposals, "selected": current, "score": score})
        return {
            "tokens": current,
            "score": score,
            "representatives": representatives,
            "influences": values,
            "retained": retained,
            "history": self.history,
            "model_calls": self.model_calls,
            "evaluations": self.evaluations,
        }
