"""Concentration-reweighted soft prompt optimization (paper equations 8–10)."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np


def concentration_loss(features, labels, temperature=0.5):
    """Return strength loss, summed supervised contrastive loss, and their d/dC.

    Features are mean lookback attention to each prompt token, one row per input.
    Singleton labels have no positive pairs and contribute zero contrastive loss.
    """
    features = np.asarray(features, dtype=float)
    n = len(features)
    raw_norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(raw_norms, 1e-12)
    unit = features / norms
    logits = unit @ unit.T / temperature
    np.fill_diagonal(logits, -np.inf)
    same = np.equal.outer(labels, labels)
    np.fill_diagonal(same, False)
    valid = same.sum(axis=1) > 0
    target = same / np.maximum(same.sum(axis=1, keepdims=True), 1)
    probabilities = np.zeros_like(logits)
    logp = np.zeros_like(logits)
    if n > 1:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        logp = shifted - np.log(exp.sum(axis=1, keepdims=True))
    contrastive = -float(logp[same] @ target[same])
    dlogits = (probabilities - target) * valid[:, None]
    dunit = (dlogits + dlogits.T) @ unit / temperature
    projection = unit * (dunit * unit).sum(axis=1, keepdims=True)
    # Below the clamp the denominator is constant, so its derivative is zero.
    dfeatures = (dunit - np.where(raw_norms > 1e-12, projection, 0)) / norms
    strength = 1 - float(features.sum(axis=1).mean())
    return strength, contrastive, np.full_like(features, -1 / n), dfeatures


def global_score(features, labels, margins, weights=(1.0, 1.0, -1.0)):
    """Hard-prompt GCS (equations 11–13); a negative fluctuation weight penalizes KL."""
    features = np.asarray(features, dtype=float)
    centered = features - features.max(axis=1, keepdims=True)
    logp = centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))
    fluctuation = 0.0
    for label in set(labels):
        mask = np.asarray(labels) == label
        mean = features[mask].mean(axis=0)
        mean -= mean.max()
        logmean = mean - np.log(np.exp(mean).sum())
        fluctuation += float((np.exp(logp[mask]) * (logp[mask] - logmean)).sum())
    return float(np.dot(weights, [np.sum(margins), features.sum(axis=1).mean(), fluctuation]))


@dataclass(frozen=True)
class Observation:
    task_loss: float
    task_gradient: np.ndarray
    attention: np.ndarray
    attention_jacobian: np.ndarray
    labels: tuple


class ConcentrateAttention:
    """Caller supplies a differentiable model observation; local code owns the loss.

    attention has shape (batch, prompt_tokens), and its Jacobian has those axes
    followed by the soft parameter shape. Backpropagation through a particular LM
    stays in `observe`; this optimizer contracts its own objective derivatives.
    """

    def __init__(self, task: str, observe: Callable[[np.ndarray], Awaitable[Observation]]):
        self.task, self.observe = task, observe

    async def _objective(self, parameters, weights, temperature):
        self.model_calls += 1
        observed = await self.observe(parameters.copy())
        strength, contrastive, ds, dc = concentration_loss(
            observed.attention, observed.labels, temperature
        )
        loss = float(np.dot(weights, [observed.task_loss, strength, contrastive]))
        gradient = weights[0] * np.asarray(observed.task_gradient) + np.tensordot(
            weights[1] * ds + weights[2] * dc,
            observed.attention_jacobian,
            axes=((0, 1), (0, 1)),
        )
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError("measured loss and gradients must be finite")
        return loss, gradient

    async def run(
        self,
        initial: np.ndarray,
        *,
        iterations=100,
        learning_rate=0.001,
        weights=(1.0, 1.0, 1.0),
        temperature=0.5,
    ) -> dict:
        parameters = np.array(initial, dtype=float, copy=True)
        first, second = np.zeros_like(parameters), np.zeros_like(parameters)
        self.model_calls = 0
        history = []
        best = None
        for step in range(iterations + 1):
            loss, gradient = await self._objective(parameters, weights, temperature)
            history.append(loss)
            if best is None or loss < best["loss"]:
                best = {"parameters": parameters.copy(), "loss": loss}
            if step == iterations:
                break
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            parameters -= (
                learning_rate
                * (first / (1 - 0.9 ** (step + 1)))
                / (np.sqrt(second / (1 - 0.999 ** (step + 1))) + 1e-8)
            )
        return {
            "best": best,
            "parameters": parameters,
            "history": history,
            "model_calls": self.model_calls,
        }
