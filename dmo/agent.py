"""Fit DAEMON metric-matching energies and sample continuations by importance resampling."""

from collections.abc import Awaitable, Callable, Sequence

import numpy as np
from scipy.special import softmax

Sample = Callable[[str, float, int], Awaitable[str]]
Metrics = Callable[[str, str], Awaitable[np.ndarray]]


def matching_loss(coefficients: np.ndarray, metrics: np.ndarray, target: np.ndarray):
    weights = softmax(-metrics @ coefficients)
    mean = weights @ metrics
    relative = mean / target - 1
    loss = float(np.sqrt(np.mean(relative**2)))
    centered = metrics - mean
    covariance = centered.T @ (weights[:, None] * centered)
    gradient = np.zeros_like(coefficients)
    if loss > 0:
        gradient = -covariance @ (relative / target) / (len(target) * loss)
    return loss, gradient, mean, weights


class DMO:
    """DAEMON Algorithms 1 and 2; sample/metric primitives require no model gradients."""

    def __init__(self, task: str, sample: Sample, evaluate: Metrics):
        self.task, self.sample, self.evaluate = task, sample, evaluate

    async def run(
        self,
        references: Sequence[tuple[str, str]],
        prefixes: Sequence[str],
        *,
        fit_samples_per_prefix: int = 4,
        iterations: int = 1000,
        learning_rate: float = 0.005,
        tolerance: float = 0.001,
        particles: int = 25,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> dict:
        self.rng = np.random.default_rng(seed)
        self.evaluations, self.generations = 0, 0
        target, proposals = await self.calibrate(references, fit_samples_per_prefix)
        self.coefficients, history = self.fit(
            target, proposals, iterations, learning_rate, tolerance
        )
        outputs = [await self.decode(prefix, particles, temperature) for prefix in prefixes]
        loss, _, achieved, _ = matching_loss(self.coefficients, proposals, target)
        return {
            "coefficients": self.coefficients.copy(),
            "target": target,
            "estimated_metrics": achieved,
            "error": loss,
            "converged": loss <= tolerance,
            "history": history,
            "outputs": outputs,
            "evaluations": self.evaluations,
            "generations": self.generations,
        }

    async def measure(self, prefix, text):
        vector = np.array(await self.evaluate(prefix, text), dtype=float, copy=True)
        self.evaluations += 1
        if not np.isfinite(vector).all():
            raise ValueError("Nonfinite sequence metrics")
        return vector

    async def generate(self, prefix, temperature):
        text = await self.sample(prefix, temperature, int(self.rng.integers(2**31)))
        self.generations += 1
        if not text.strip():
            raise ValueError("Empty generated continuation")
        return text

    async def calibrate(self, references, count):
        targets, proposals = [], []
        for prefix, reference in references:
            targets.append(await self.measure(prefix, reference))
            for _ in range(count):
                text = await self.generate(prefix, 1.0)
                proposals.append(await self.measure(prefix, text))
        target = np.mean(targets, axis=0)
        if np.any(target == 0):
            raise ValueError(
                "Measured zero target makes the paper's relative-error objective undefined"
            )
        return target, np.array(proposals)

    def fit(self, target, proposals, iterations, rate, tolerance):
        coefficients = np.zeros(len(target))
        moment, variance = np.zeros_like(coefficients), np.zeros_like(coefficients)
        best, best_loss, history = coefficients.copy(), np.inf, []
        for step in range(iterations + 1):
            loss, gradient, _, _ = matching_loss(coefficients, proposals, target)
            if not np.isfinite(loss) or not np.isfinite(gradient).all():
                raise ValueError("Nonfinite coefficient objective")
            history.append(loss)
            if loss < best_loss:
                best, best_loss = coefficients.copy(), loss
            if loss <= tolerance or step == iterations:
                break
            moment = 0.9 * moment + 0.1 * gradient
            variance = 0.999 * variance + 0.001 * gradient**2
            coefficients -= (
                rate
                * (moment / (1 - 0.9 ** (step + 1)))
                / (np.sqrt(variance / (1 - 0.999 ** (step + 1))) + 1e-8)
            )
        return best, history

    async def decode(self, prefix: str, particles: int = 25, temperature: float = 1.0) -> dict:
        texts, metrics = [], []
        for _ in range(particles):
            text = await self.generate(prefix, temperature)
            texts.append(text)
            metrics.append(await self.measure(prefix, text))
        weights = softmax(-np.array(metrics) @ self.coefficients)
        if not np.isfinite(weights).all():
            raise ValueError("Nonfinite importance weights")
        selected = int(self.rng.choice(particles, p=weights))
        return {
            "prefix": prefix,
            "text": texts[selected],
            "metrics": metrics[selected],
            "particles": texts,
            "weights": weights,
        }
