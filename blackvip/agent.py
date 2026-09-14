"""Input-dependent visual prompting with two-sided SPSA gradient correction."""

from collections.abc import Awaitable, Callable, Sequence

import numpy as np
from scipy.special import logsumexp


class BlackVIP:
    def __init__(
        self,
        task: str,
        coordinate: Callable[[np.ndarray, np.ndarray], Awaitable[np.ndarray]],
        forward: Callable[[np.ndarray], Awaitable[np.ndarray]],
        clip: Callable[[np.ndarray], np.ndarray] = lambda image: np.clip(image, 0, 1),
    ):
        self.task, self.coordinate, self.forward, self.clip = task, coordinate, forward, clip

    async def loss(self, parameters, images, labels, scale):
        self.coordinator_calls += 1
        prompts = await self.coordinate(parameters.copy(), images)
        self.model_calls += 1
        logits = np.asarray(await self.forward(self.clip(images + scale * prompts)))
        loss = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]))
        if not np.isfinite(loss):
            raise ValueError("measured classification loss must be finite")
        return loss

    async def run(
        self,
        initial_parameters: np.ndarray,
        batches: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        epochs: int = 1,
        learning_rate: float = 0.1,
        perturbation: float = 0.1,
        offset: float = 0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        momentum: float = 0.9,
        averages: int = 1,
        prompt_scale: float = 1,
        seed: int = 0,
    ) -> dict:
        rng = np.random.default_rng(seed)
        parameters = np.array(initial_parameters, dtype=float, copy=True)
        velocity = np.zeros_like(parameters)
        self.model_calls = self.coordinator_calls = 0
        self.history = []
        step = 0
        for _ in range(epochs):
            for images, labels in batches:
                step += 1
                rate = learning_rate / (step + offset) ** alpha
                radius = perturbation / step**gamma
                gradients, pair_losses = [], []
                for _ in range(averages):
                    delta = rng.uniform(0.5, 1, parameters.shape) * rng.choice(
                        [-1, 1], parameters.shape
                    )
                    positive = await self.loss(
                        parameters + radius * delta, images, labels, prompt_scale
                    )
                    negative = await self.loss(
                        parameters - radius * delta, images, labels, prompt_scale
                    )
                    gradients.append((positive - negative) / (2 * radius * delta))
                    pair_losses.append((positive, negative))
                gradient = np.mean(gradients, axis=0)
                velocity = momentum * velocity + gradient
                correction = gradient + momentum * velocity
                parameters -= rate * correction
                self.history.append(
                    {
                        "gradient": gradient.copy(),
                        "velocity": velocity.copy(),
                        "rate": rate,
                        "radius": radius,
                        "loss_pairs": pair_losses,
                        "parameters": parameters.copy(),
                    }
                )
        return {
            "parameters": parameters,
            "history": self.history,
            "model_calls": self.model_calls,
            "coordinator_calls": self.coordinator_calls,
        }
