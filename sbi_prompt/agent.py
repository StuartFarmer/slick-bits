"""Sample a soft-prompt ensemble with ABC-SMC and held-out ensemble selection."""

import math
from collections.abc import Awaitable, Callable

import numpy as np


class SBIPrompt:
    """Reliable Gradient-free and Likelihood-free Prompt Tuning's ABC-SMC path.

    The evaluator maps a low-dimensional soft prompt to training accuracy. Model
    conditioning and any fixed random projection belong to the caller. Validation
    scores a complete weighted ensemble, separately from training acceptance.
    """

    def __init__(
        self,
        task: str,
        evaluate: Callable[[np.ndarray], Awaitable[float]],
        validate: Callable[[np.ndarray, np.ndarray], Awaitable[float]],
    ):
        self.task, self.evaluate, self.validate = task, evaluate, validate

    async def _measure(self, theta):
        self.evaluations += 1
        score = float(await self.evaluate(theta.copy()))
        if not math.isfinite(score):
            raise ValueError("measured training accuracy must be finite")
        return score

    async def _population(self, previous, threshold, dimension, count, variance, max_attempts):
        accepted = []
        if previous is not None:
            variances = previous.var(axis=0)
            # Degenerate particle clouds must still allow exploration.
            variances = np.maximum(variances, np.finfo(float).eps)
            scale = np.sqrt(10 * variances / variances.mean())
        else:
            scale = math.sqrt(variance)
        for _ in range(max_attempts):
            center = (
                np.zeros(dimension)
                if previous is None
                else previous[self.rng.integers(len(previous))]
            )
            theta = center + self.rng.normal(size=dimension) * scale
            if await self._measure(theta) >= threshold:
                accepted.append(theta)
                if len(accepted) == count:
                    return np.stack(accepted)
        return None

    async def run(
        self,
        *,
        dimension: int,
        training_size: int,
        particles: int = 20,
        prior_variance: float = 1.0,
        initial_samples: int = 20,
        max_attempts_per_stage: int = 1000,
        seed: int = 0,
    ) -> dict:
        """Raise the acceptance threshold by one correct training example per stage."""
        self.rng = np.random.default_rng(seed)
        self.evaluations = self.validations = 0
        pilots = [
            self.rng.normal(size=dimension) * math.sqrt(prior_variance)
            for _ in range(initial_samples)
        ]
        initial_accuracy = max([await self._measure(theta) for theta in pilots])
        initial_correct = round(initial_accuracy * training_size)
        previous, best = None, None
        history = []
        stopped = "perfect_threshold"
        # Count correct examples explicitly so rounding cannot skip accuracy 1.
        for correct in range(initial_correct, training_size + 1):
            threshold = correct / training_size
            population = await self._population(
                previous, threshold, dimension, particles, prior_variance, max_attempts_per_stage
            )
            if population is None:
                stopped = "stage_budget"
                break
            weights = np.full(particles, 1 / particles)
            self.validations += 1
            score = float(await self.validate(population.copy(), weights.copy()))
            if not math.isfinite(score):
                raise ValueError("measured validation score must be finite")
            history.append({"threshold": threshold, "validation_score": score})
            if best is None or score >= best["validation_score"]:
                best = {
                    "particles": population.copy(),
                    "weights": weights.copy(),
                    "validation_score": score,
                }
            previous = population
        if best is None:
            raise RuntimeError("no complete accepted ensemble within the stage budget")
        return {
            **best,
            "history": history,
            "stopped": stopped,
            "evaluations": self.evaluations,
            "validations": self.validations,
        }
