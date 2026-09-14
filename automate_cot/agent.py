"""Learn categorical demonstration-slot policies with centered policy gradients."""

from collections.abc import Awaitable, Callable, Sequence

import numpy as np
from slick import prompt
from slick.providers import Provider


def policy_gradient(
    probabilities: np.ndarray, samples: np.ndarray, losses: np.ndarray
) -> np.ndarray:
    """Source's signed inverse-probability estimator, with independent sample arrays."""
    derivatives = np.broadcast_to(-1 / probabilities, (len(samples), *probabilities.shape)).copy()
    for j, indices in enumerate(samples):
        derivatives[j, np.arange(len(indices)), indices] *= -1
    return np.einsum("n,nij->ij", losses - losses.mean(), derivatives) / (len(samples) - 1)


def project_simplex(values: np.ndarray, floor: float = 0.0001) -> np.ndarray:
    """Euclidean projection onto a unit simplex with the source's positive floor."""
    shifted = values - floor
    ordered = np.sort(shifted)[::-1]
    cumulative = np.cumsum(ordered) - (1 - len(values) * floor)
    rho = np.flatnonzero(ordered - cumulative / np.arange(1, len(values) + 1) > 0)[-1]
    return np.maximum(shifted - cumulative[rho] / (rho + 1), 0) + floor


class AutomateCoT:
    """Own policy sampling, centered loss estimates, clipped Adam and projection.

    evaluate receives ordered demonstration strings and returns a finite loss to
    minimize on caller-owned training examples. Rationale generation is upstream.
    """

    def __init__(
        self, task: str, provider: Provider, evaluate: Callable[[Sequence[str]], Awaitable[float]]
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate

    @prompt(template="answer.j2")
    async def answer(self, question: str, *, generated: str) -> str:
        return generated

    async def _sample_losses(self, pool, count, rng):
        samples, losses = [], []
        for _ in range(count):
            sample = [int(rng.choice(len(pool), p=row / row.sum())) for row in self.probabilities]
            # The source prepends each slot, so the final prompt reverses slot order.
            demonstrations = tuple(pool[i] for i in reversed(sample))
            self.evaluations += 1
            loss = float(await self.evaluate(demonstrations))
            if not np.isfinite(loss):
                raise ValueError("demonstration loss must be finite")
            samples.append(sample)
            losses.append(loss)
        return np.array(samples), np.array(losses)

    async def run(
        self,
        pool: Sequence[str],
        *,
        slots: int = 8,
        steps: int = 20,
        samples_per_step: int = 10,
        learning_rate: float = 0.001,
        seed: int = 0,
    ) -> dict:
        """Optimize independent slot distributions; deploy their deterministic modes."""
        rng = np.random.default_rng(seed)
        self.probabilities = np.full((slots, len(pool)), 1 / len(pool))
        first = np.zeros_like(self.probabilities)
        second = np.zeros_like(first)
        self.evaluations = 0
        history = []
        for step in range(1, steps + 1):
            samples, losses = await self._sample_losses(pool, samples_per_step, rng)
            gradient = policy_gradient(self.probabilities, samples, losses)
            gradient *= min(1, 3 / (np.linalg.norm(gradient) + 1e-12))
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            update = (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            self.probabilities = np.array(
                [project_simplex(row) for row in self.probabilities - learning_rate * update]
            )
            history.append({"samples": samples, "losses": losses, "gradient": gradient.copy()})
        indices = np.argmax(self.probabilities, axis=1)[::-1].tolist()
        self.demonstrations = tuple(pool[i] for i in indices)
        self.evaluations += 1
        loss = float(await self.evaluate(self.demonstrations))
        if not np.isfinite(loss):
            raise ValueError("demonstration loss must be finite")
        return {
            "demonstrations": self.demonstrations,
            "indices": indices,
            "loss": loss,
            "probabilities": self.probabilities.copy(),
            "history": history,
            "evaluations": self.evaluations,
        }
