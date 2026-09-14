"""SEED query grouping, adapted from upstream legacy reorder_data/get_even_clusters."""

import random

import numpy as np


def _balanced_clusters(embeddings, count, rng):
    from scipy.optimize import linear_sum_assignment

    n = len(embeddings)
    capacities = [n // count + (i < n % count) for i in range(count)]
    centers = embeddings[rng.sample(range(n), count)]
    previous = None
    # ponytail: 10 balanced Lloyd iterations; raise only if grouping quality warrants it.
    for _ in range(10):
        slots = np.repeat(np.arange(count), capacities)
        distance = np.linalg.norm(embeddings[:, None] - centers[slots], axis=2)
        rows, columns = linear_sum_assignment(distance)
        assignment = np.empty(n, dtype=int)
        assignment[rows] = slots[columns]
        if np.array_equal(assignment, previous):
            break
        previous = assignment.copy()
        centers = np.array([embeddings[assignment == i].mean(axis=0) for i in range(count)])
    return [[int(j) for j in np.flatnonzero(assignment == i)] for i in range(count)]


def form_batches(count, size, strategy="RND", embeddings=None, *, seed=0):
    """Return input-index batches, including the final partial batch.

    PRX/DIV use balanced clustering with SciPy's assignment solver. CLS/FAR
    compare distance to the current batch (paper), rather than all prior records
    (legacy code). RND shuffles a local index list without mutating caller data.
    """
    if count == 0:
        return []
    rng = random.Random(seed)
    indices = list(range(count))
    if strategy == "RND":
        rng.shuffle(indices)
        return [indices[i : i + size] for i in range(0, count, size)]
    embeddings = np.asarray(embeddings, dtype=float)
    if strategy == "PRX":
        return _balanced_clusters(embeddings, (count + size - 1) // size, rng)
    if strategy == "DIV":
        clusters = _balanced_clusters(embeddings, min(size, count), rng)
        for cluster in clusters:
            rng.shuffle(cluster)
        return [
            [cluster[i] for cluster in clusters if i < len(cluster)]
            for i in range(max(map(len, clusters)))
        ]
    batches = []
    while indices:
        batch = [indices.pop(rng.randrange(len(indices)))]
        while indices and len(batch) < size:
            distance = np.linalg.norm(embeddings[indices, None] - embeddings[batch], axis=2).min(
                axis=1
            )
            next_index = np.argmax(distance) if strategy == "FAR" else np.argmin(distance)
            batch.append(indices.pop(int(next_index)))
        batches.append(batch)
    return batches
