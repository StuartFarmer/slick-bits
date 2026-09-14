"""AMA's official sparse/low-rank structure estimator and MeTaL label model.

Structure learning adapted from HazyResearch/ama_prompting/boosting/run_ws.py
at 460843d93a9e4bf2115eb35e4a02e82b6b67feac (Apache-2.0; see LICENSE).
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def majority_vote(votes: Sequence[str | None]) -> str | None:
    """Ignore abstentions; ties go to the first encountered answer."""
    counts = Counter(vote for vote in votes if vote is not None)
    return counts.most_common(1)[0][0] if counts else None


def learn_structure(votes: np.ndarray) -> np.ndarray:
    """Solve the official convex objective using its covariance and defaults."""
    import cvxpy as cp
    from scipy.linalg import sqrtm

    n, m = votes.shape
    covariance = votes.T @ votes / (n - 1) - np.outer(votes.mean(0), votes.mean(0))
    root = np.real(sqrtm((covariance + covariance.T) / 2))
    low_rank = cp.Variable((m, m), PSD=True)
    sparse = cp.Variable((m, m), PSD=True)
    residual = cp.Variable((m, m), PSD=True)
    objective = cp.Minimize(
        0.5 * cp.norm(residual @ root, "fro") ** 2
        - cp.trace(residual)
        + (1 / np.sqrt(m)) * (1e-8 * cp.pnorm(sparse, 1) + cp.norm(low_rank, "nuc"))
    )
    problem = cp.Problem(objective, [residual == sparse - low_rank, low_rank >> 0])
    problem.solve(solver=cp.SCS, verbose=False)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or sparse.value is None:
        raise RuntimeError(f"AMA structure learning failed: {problem.status}")
    result = sparse.value.copy()
    if not np.isfinite(result).all():
        raise RuntimeError("AMA structure learning returned nonfinite values")
    np.fill_diagonal(result, 0)
    return result


def select_dependency(structure: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Appendix A.3: retain the strongest edge unless all off-diagonals >= 1."""
    rows, columns = np.triu_indices(len(structure), 1)
    values = np.abs(structure[rows, columns])
    if not len(values) or values.min() >= 1 or values.max() == 0:
        return ()
    index = values.argmax()
    return ((int(rows[index]), int(columns[index])),)


@dataclass(frozen=True)
class WSConfig:
    seed: int = 123
    epochs: int = 10000
    learning_rate: float = 1e-5
    dependency_epochs: int = 80000
    dependency_learning_rate: float = 1e-6
    # None learns structure; () is the paper's no-dependency ablation.
    dependencies: tuple[tuple[int, int], ...] | None = None
    class_balance: tuple[float, ...] | None = None


class WeakSupervision:
    """Fit the pinned official MeTaL implementation without any gold labels.

    Labels are shared across rows; None encodes abstention. The official model
    sets global Python/NumPy/Torch RNG seeds and runs synchronously on CPU.
    Keep concurrent numerical fits in separate processes. Numerical failures
    propagate; no majority-vote fallback hides them.
    """

    def __init__(self, labels: Sequence[str], config: WSConfig = WSConfig()):
        self.labels = tuple(labels)
        self.config = config
        self.dependencies: tuple[tuple[int, int], ...] = ()
        self.structure: np.ndarray | None = None
        self.model = None

    def _encode(self, votes: Sequence[Sequence[str | None]]) -> np.ndarray:
        codes = {label: index + 1 for index, label in enumerate(self.labels)}
        codes[None] = 0
        return np.asarray([[codes[vote] for vote in row] for row in votes], dtype=int)

    def fit(self, votes: Sequence[Sequence[str | None]]) -> "WeakSupervision":
        # A failed refit must not leave a previous or partially fitted model usable.
        self.model, self.structure, self.dependencies = None, None, ()
        matrix = self._encode(votes)
        n, m = matrix.shape
        if n < 2 or m < 3 or len(self.labels) < 2:
            raise ValueError("weak supervision needs >=2 examples, >=3 chains and >=2 fixed labels")
        if np.any(np.ptp(matrix, axis=0) == 0):
            raise ValueError(
                "constant or all-abstaining chain: collect more varied unlabeled votes"
            )
        from metal.label_model import LabelModel

        self.dependencies = self.config.dependencies or ()
        if self.config.dependencies is None:
            # Preserve upstream structure encoding: abstention merges with class 0.
            unscaled = np.maximum(matrix - 1, 0).astype(float)
            if len(self.labels) == 2:
                self.structure = learn_structure(unscaled)
            else:
                self.structure = np.mean(
                    [
                        learn_structure((unscaled == c).astype(float))
                        for c in range(len(self.labels))
                    ],
                    axis=0,
                )
            self.dependencies = select_dependency(self.structure)
        model = LabelModel(k=len(self.labels), seed=self.config.seed, verbose=False)
        options = dict(
            class_balance=self.config.class_balance,
            abstains=bool((matrix == 0).any()),
            symmetric=False,
            log_train_every=50,
        )
        model.train_model(
            matrix, n_epochs=self.config.epochs, lr=self.config.learning_rate, **options
        )
        if self.dependencies:
            # Upstream additionally hard-codes 20,000 epochs for its mu phase.
            model.train_model(
                matrix,
                deps=list(self.dependencies),
                n_epochs=self.config.dependency_epochs,
                lr=self.config.dependency_learning_rate,
                **options,
            )
        self.model = model
        return self

    def predict_proba(self, votes: Sequence[Sequence[str | None]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit the label model before prediction")
        matrix = self._encode(votes)
        if matrix.shape[1] != self.model.m:
            raise ValueError("prediction columns must match the fitted prompt chains")
        probabilities = self.model.predict_proba(matrix)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("MeTaL returned nonfinite probabilities")
        return probabilities
