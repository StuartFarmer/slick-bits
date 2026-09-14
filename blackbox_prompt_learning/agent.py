"""Optimize independent discrete prompt distributions using variance-reduced policy gradients."""

from collections.abc import Awaitable, Callable, Sequence

import numpy as np

Evaluate = Callable[[tuple[str, ...], str], Awaitable[float]]


def project_simplex(vector: np.ndarray) -> np.ndarray:
    ordered = np.sort(vector)[::-1]
    thresholds = (np.cumsum(ordered) - 1) / np.arange(1, len(vector) + 1)
    active = np.flatnonzero(ordered > thresholds)[-1]
    return np.maximum(vector - thresholds[active], 0)


def variance_reduced_gradient(probabilities: np.ndarray, samples: np.ndarray, losses: np.ndarray):
    """Algebraic form of the release's centered +/-1/p estimator, avoiding unused 1/0."""
    gradient = np.zeros_like(probabilities)
    centered = losses - losses.mean()
    for sample, loss in zip(samples, centered):
        for position, token in enumerate(sample):
            gradient[position, token] += 2 * loss / probabilities[position, token]
    return gradient / (len(samples) - 1)


class BlackBoxPromptLearning:
    """BDPL learns probability rows, without policy/target-model gradient access."""

    def __init__(self, task: str, evaluate: Evaluate):
        self.task, self.evaluate = task, evaluate

    async def run(
        self,
        vocabulary: Sequence[str],
        batches: Sequence[str],
        *,
        prompt_length: int = 5,
        epochs: int = 10,
        samples_per_batch: int = 20,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        max_grad_norm: float = 3.0,
        seed: int = 0,
    ) -> dict:
        self.vocabulary = tuple(vocabulary)
        self.probabilities = np.full((prompt_length, len(vocabulary)), 1 / len(vocabulary))
        self.rng = np.random.default_rng(seed)
        self.moment, self.variance = (
            np.zeros_like(self.probabilities),
            np.zeros_like(self.probabilities),
        )
        self.updates, self.evaluations, self.history = 0, 0, []
        for _ in range(epochs):
            for batch in batches:
                samples, losses = await self.sample_batch(batch, samples_per_batch)
                gradient = variance_reduced_gradient(self.probabilities, samples, losses)
                self.update(gradient, learning_rate, weight_decay, max_grad_norm)
                self.history.append(
                    {"mean_loss": float(losses.mean()), "samples": samples, "losses": losses}
                )
        prompt = tuple(self.vocabulary[index] for index in np.argmax(self.probabilities, axis=1))
        return {
            "prompt": prompt,
            "probabilities": self.probabilities.copy(),
            "history": self.history,
            "evaluations": self.evaluations,
        }

    async def sample_batch(self, batch, count):
        samples, losses = [], []
        for _ in range(count):
            indices = np.array([self.rng.choice(len(row), p=row) for row in self.probabilities])
            prompt = tuple(self.vocabulary[index] for index in indices)
            loss = float(await self.evaluate(prompt, batch))
            self.evaluations += 1
            if not np.isfinite(loss):
                raise ValueError("Nonfinite black-box loss")
            samples.append(indices)
            losses.append(loss)
        return np.array(samples), np.array(losses)

    def update(self, gradient, rate, decay, max_norm):
        if not np.isfinite(gradient).all():
            raise ValueError("Nonfinite categorical gradient")
        gradient *= min(1.0, max_norm / (np.linalg.norm(gradient) + 1e-6))
        self.updates += 1
        self.moment = 0.9 * self.moment + 0.1 * gradient
        self.variance = 0.999 * self.variance + 0.001 * gradient**2
        self.probabilities *= 1 - rate * decay
        self.probabilities -= (
            rate
            * (self.moment / (1 - 0.9**self.updates))
            / (np.sqrt(self.variance / (1 - 0.999**self.updates)) + 1e-8)
        )
        if not np.isfinite(self.probabilities).all():
            raise ValueError("Nonfinite probability update")
        self.probabilities = np.array([project_simplex(row) for row in self.probabilities])
