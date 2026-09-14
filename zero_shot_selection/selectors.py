"""Array-based interpretations of Algorithms 1 and 2; return parent indices with replacement."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _normalize(values: NDArray, epsilon: float) -> NDArray:
    return (values - values.min()) / (np.ptp(values) + epsilon)


def kimi_selection(
    normalized_errors: ArrayLike,
    case_variances: ArrayLike,
    sizes: ArrayLike,
    k: int,
    stage: float,
    *,
    rng: np.random.Generator | None = None,
    epsilon: float = 1e-12,
) -> NDArray[np.intp]:
    """Algorithm 1: stage-weighted case curriculum, novelty, parsimony, and tournaments.

    Rows represent candidates, columns shared cases. Supply R_n and the case
    variances explicitly: the paper does not fully define their preprocessing.
    Sizes are any meaningful complexity measure. Inputs must be finite, population
    and case dimensions nonempty for k > 0, and stage in [0, 1]. No inputs mutate.
    """
    if k == 0:
        return np.empty(0, dtype=np.intp)
    rng = np.random.default_rng() if rng is None else rng
    errors = np.asarray(normalized_errors, dtype=float)
    variances = np.asarray(case_variances, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    novelty = 1 - np.abs(errors @ errors.mean(axis=0))
    weights = (1 - stage) + stage * variances / (variances.mean() + epsilon)
    fitness = -(errors * errors) @ weights / errors.shape[1]
    z_fitness = (fitness - fitness.mean()) / (fitness.std() + epsilon)
    z_novelty = (novelty - novelty.mean()) / (novelty.std() + epsilon)
    parsimony = 1 - _normalize(sizes, epsilon)
    score = stage * z_fitness + (1 - stage) * z_novelty + 0.1 * parsimony
    width = max(2, int(2 + 6 * stage))
    tournaments = rng.integers(len(errors), size=(k, width))
    return tournaments[np.arange(k), score[tournaments].argmax(axis=1)]


def gpt_selection(
    behaviors: ArrayLike,
    mean_errors: ArrayLike,
    sizes: ArrayLike,
    heights: ArrayLike,
    k: int,
    stage: float,
    *,
    rng: np.random.Generator | None = None,
    epsilon: float = 1e-12,
    elite_downweight: float = 0.5,
) -> NDArray[np.intp]:
    """Algorithm 2: behavior crowding, parsimony, elite slots, and Boltzmann sampling.

    Supply the behavior matrix P as used in the desired experiment. This function
    follows clip(P @ P.T, -1, 1) literally; it does not silently center or normalize
    rows. Lower mean_errors is better. The paper omits the elite down-weight factor;
    0.5 is a configurable interpretation, not a recovered upstream constant.
    Inputs obey the finite/nonempty/stage contract of kimi_selection.
    """
    if k == 0:
        return np.empty(0, dtype=np.intp)
    rng = np.random.default_rng() if rng is None else rng
    behaviors = np.asarray(behaviors, dtype=float)
    errors = np.asarray(mean_errors, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    heights = np.asarray(heights, dtype=float)
    # Algorithm 2 explicitly uses all pairs: O(n²) memory and O(n²d) work.
    similarity = np.clip(behaviors @ behaviors.T, -1, 1)
    dispersion = 1 - similarity.mean(axis=1)
    fit = 1 - _normalize(errors, epsilon)
    diversity = _normalize(dispersion, epsilon)
    parsimony = (1 - _normalize(sizes, epsilon)) + (1 - _normalize(heights, epsilon))
    score = (
        (1.2 + 1.8 * stage) * fit
        + (0.65 * (1 - stage) + 0.15) * diversity
        + (0.15 + 0.55 * stage) / 2 * parsimony
    )
    elite_count = min(len(errors), int(np.ceil(k * (0.08 + 0.12 * stage))))
    elites = np.argsort(-score, kind="stable")[:elite_count]
    temperature = max(0.05, 0.9 - 0.7 * stage)
    logits = (score - score.max()) / temperature
    threshold = 0.92 - 0.25 * (1 - stage)
    crowding = 1 / (1 + (similarity > threshold).sum(axis=1))
    mass = np.exp(logits) * (0.35 + 0.65 * crowding)
    mass[elites] *= elite_downweight
    sampled = rng.choice(len(errors), size=k - elite_count, p=mass / mass.sum())
    return np.concatenate((elites, sampled))
