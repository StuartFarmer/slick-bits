"""Fit the paper's ten-ReLU regression network using separate validation observations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TrainingData:
    """Rows are observations; columns match selected candidates in their exact order.

    Targets are already aligned to the outcome horizon. The caller owns splitting,
    purging overlapping horizons, and keeping test observations out of this object.
    """

    x_train: Array
    y_train: Array
    x_validation: Array
    y_validation: Array


@dataclass(frozen=True)
class Combiner:
    minimum: Array
    scale: Array
    w1: Array
    b1: Array
    w2: Array
    b2: float
    validation_loss: float
    best_epoch: int

    def predict(self, values: Array) -> Array:
        """Predict scalar outcomes for rows of ordered candidate signals."""
        hidden = np.maximum((np.asarray(values) - self.minimum) / self.scale @ self.w1 + self.b1, 0)
        return hidden @ self.w2 + self.b2

    def local_weights(self, values: Array) -> tuple[Array, Array]:
        """Return exact row-specific affine weights and intercepts in raw input units."""
        active = ((np.asarray(values) - self.minimum) / self.scale @ self.w1 + self.b1) > 0
        weights = (active * self.w2) @ self.w1.T / self.scale
        intercepts = (active * self.w2) @ self.b1 + self.b2 - weights @ self.minimum
        return weights, intercepts


def fit_combiner(
    data: TrainingData,
    *,
    epochs: int = 1000,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    regularization: float = 0.001,
    seed: int = 0,
) -> Combiner:
    """Minibatch SGD on MSE + L2; retain the lowest validation-MSE checkpoint.

    Feature range normalization follows the release, fitted to training rows only.
    Validation participates only in checkpoint choice, never gradient updates.
    The initial checkpoint is eligible, so zero epochs gives an untrained model.
    """
    raw = np.asarray(data.x_train, dtype=float)
    y = np.asarray(data.y_train, dtype=float)
    minimum = raw.min(axis=0)
    scale = np.ptp(raw, axis=0)
    scale = np.where(scale == 0, 1, scale)
    x = (raw - minimum) / scale
    validation = (np.asarray(data.x_validation, dtype=float) - minimum) / scale
    target = np.asarray(data.y_validation, dtype=float)
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, np.sqrt(2 / x.shape[1]), (x.shape[1], 10))
    b1 = np.zeros(10)
    w2 = rng.normal(0, np.sqrt(1 / 10), 10)
    b2 = 0.0
    best = None
    for epoch in range(epochs + 1):
        prediction = np.maximum(validation @ w1 + b1, 0) @ w2 + b2
        loss = float(np.mean((prediction - target) ** 2))
        if not np.isfinite(loss):
            raise ValueError("combiner validation loss must be finite")
        if best is None or loss < best.validation_loss:
            best = Combiner(minimum, scale, w1.copy(), b1.copy(), w2.copy(), b2, loss, epoch)
        if epoch == epochs:
            break
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            batch = order[start : start + batch_size]
            inputs = x[batch]
            hidden = np.maximum(inputs @ w1 + b1, 0)
            error = 2 * (hidden @ w2 + b2 - y[batch]) / len(batch)
            hidden_error = error[:, None] * w2 * (hidden > 0)
            w1 -= learning_rate * (inputs.T @ hidden_error + regularization * w1)
            b1 -= learning_rate * hidden_error.sum(axis=0)
            w2 -= learning_rate * (hidden.T @ error + regularization * w2)
            b2 -= learning_rate * float(error.sum())
    return best
