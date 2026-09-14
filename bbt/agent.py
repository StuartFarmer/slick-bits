"""Black-box tuning in a fixed low-dimensional random prompt subspace."""

import math
from collections.abc import Awaitable, Callable

import numpy as np

from .cma import CMA


def random_projection(rng, dimension, output_size, embedding_std, sigma, alpha):
    return rng.normal(
        0, alpha * embedding_std / (math.sqrt(dimension) * sigma), (output_size, dimension)
    )


class BBT:
    def __init__(
        self,
        task: str,
        evaluate: Callable[[np.ndarray], Awaitable[float]],
        initial_prompt: np.ndarray,
        *,
        projection: np.ndarray | None = None,
    ):
        self.task, self.evaluate = task, evaluate
        self.initial_prompt = np.array(initial_prompt, dtype=float, copy=True)
        self.projection = projection

    async def run(
        self,
        *,
        dimension: int = 500,
        budget: int = 8000,
        population: int = 20,
        sigma: float = 1,
        embedding_std: float = 1,
        alpha: float = 1,
        seed: int = 0,
    ) -> dict:
        rng = np.random.default_rng(seed)
        projection = self.projection
        if projection is None:
            projection = random_projection(
                rng, dimension, self.initial_prompt.size, embedding_std, sigma, alpha
            )
        dimension = projection.shape[1]
        self.strategy = CMA(np.zeros(dimension), sigma, population, rng)
        self.evaluations, self.history = 0, []
        for _ in range(budget // population):
            solutions = self.strategy.ask()
            losses = []
            for latent in solutions:
                prompt = self.initial_prompt + (projection @ latent).reshape(
                    self.initial_prompt.shape
                )
                self.evaluations += 1
                losses.append(float(await self.evaluate(prompt)))
            self.strategy.tell(solutions, np.array(losses))
            self.history.append(
                {"losses": losses, "mean": self.strategy.mean.copy(), "sigma": self.strategy.sigma}
            )
        best = self.initial_prompt + (projection @ self.strategy.best).reshape(
            self.initial_prompt.shape
        )
        return {
            "prompt": best,
            "latent": self.strategy.best.copy(),
            "loss": self.strategy.best_loss if self.evaluations else None,
            "projection": projection,
            "history": self.history,
            "evaluations": self.evaluations,
        }
