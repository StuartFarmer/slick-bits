"""Adapt the author's penalized K-means selection to a caller-owned encoder.

Derived from McDaniel7/GreenTEA utils/text_utils.py (MIT; see LICENSE.upstream).
"""

import random
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.cluster import KMeans


class KMeansTopics:
    """Return indices of examples in the largest reference-answer topic.

    Select k by normalized inertia + penalty * k, then refit with the author's
    final clustering seed. `encode` is a synchronous batch embedding function.
    """

    def __init__(
        self,
        encode: Callable[[Sequence[str]], Sequence[Sequence[float]]],
        *,
        max_examples: int = 10,
        sample_size: int = 100,
        min_clusters: int = 5,
        max_clusters: int = 20,
        penalty: float = 0.02,
    ):
        self.encode = encode
        self.max_examples = max_examples
        self.sample_size = sample_size
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.penalty = penalty

    def __call__(self, texts: Sequence[str], rng: random.Random) -> list[int]:
        indices = list(range(len(texts)))
        if len(indices) > self.sample_size:
            indices = rng.sample(indices, self.sample_size)
        if len(indices) <= 1:
            return indices[: self.max_examples]
        vectors = np.asarray(self.encode([texts[i] for i in indices]), dtype=float)
        if not np.isfinite(vectors).all():
            raise ValueError("encoder returned non-finite embeddings")
        distinct = len(np.unique(vectors, axis=0))
        if distinct == 1:
            return indices[: self.max_examples]
        inertia = np.square(vectors - vectors.mean(axis=0)).sum()
        counts = range(min(distinct, self.min_clusters), min(distinct, self.max_clusters) + 1)
        best_k = min(
            counts,
            key=lambda k: KMeans(n_clusters=k, random_state=0, n_init=1).fit(vectors).inertia_
            / inertia
            + self.penalty * k,
        )
        labels = KMeans(n_clusters=best_k, max_iter=200, n_init=1, random_state=10).fit_predict(
            vectors
        )
        largest = np.bincount(labels).argmax()
        return [i for i, label in zip(indices, labels) if label == largest][: self.max_examples]
