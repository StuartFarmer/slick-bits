"""Alternating layerwise deep-prompt tuning with persistent CMA search states."""

from collections.abc import Awaitable, Callable, Sequence

import numpy as np

from bbt.agent import random_projection
from bbt.cma import CMA


class BBTv2:
    def __init__(
        self,
        task: str,
        evaluate: Callable[[np.ndarray], Awaitable[float]],
        initial_prompts: np.ndarray,
        *,
        projections: Sequence[np.ndarray] | None = None,
    ):
        self.task, self.evaluate = task, evaluate
        self.initial_prompts = np.array(initial_prompts, dtype=float, copy=True)
        self.projections = projections

    async def run(
        self,
        *,
        dimension: int = 500,
        budget: int = 8000,
        population: int = 20,
        sigma: float = 0.2,
        layer_stds: Sequence[float] = (),
        alpha: float = 0.2,
        seed: int = 0,
    ) -> dict:
        rng = np.random.default_rng(seed)
        layers = len(self.initial_prompts)
        projections = self.projections
        if projections is None:
            projections = [
                random_projection(
                    rng, dimension, self.initial_prompts[i].size, layer_stds[i], sigma, alpha
                )
                for i in range(layers)
            ]
        self.strategies = [
            CMA(np.zeros(matrix.shape[1]), sigma, population, rng) for matrix in projections
        ]
        current = self.initial_prompts.copy()
        self.evaluations, self.history = 0, []
        for cycle in range(budget // (population * layers)):
            for layer, strategy in enumerate(self.strategies):
                solutions, losses = strategy.ask(), []
                for latent in solutions:
                    proposal = current.copy()
                    proposal[layer] = self.initial_prompts[layer] + (
                        projections[layer] @ latent
                    ).reshape(current[layer].shape)
                    self.evaluations += 1
                    losses.append(float(await self.evaluate(proposal)))
                strategy.tell(solutions, np.array(losses))
                # Official deepbbt.py installs this layer's best-ever latent,
                # even though its historical loss used other-layer contexts.
                current[layer] = self.initial_prompts[layer] + (
                    projections[layer] @ strategy.best
                ).reshape(current[layer].shape)
                self.history.append(
                    {"cycle": cycle, "layer": layer, "losses": losses, "prompts": current.copy()}
                )
        self.evaluations += 1
        loss = float(await self.evaluate(current.copy()))
        if not np.isfinite(loss):
            raise ValueError("measured final loss must be finite")
        return {
            "prompts": current,
            "loss": loss,
            "projections": projections,
            "history": self.history,
            "evaluations": self.evaluations,
        }
