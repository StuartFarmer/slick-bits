"""Learn autoregressive discrete prompts with the released on-policy soft Q loss."""

from collections.abc import Awaitable, Callable, Sequence

import numpy as np
from scipy.special import logsumexp, softmax

Forward = Callable[[np.ndarray, str, tuple[int, ...]], Awaitable[tuple[np.ndarray, np.ndarray]]]
Evaluate = Callable[[str, tuple[int, ...]], Awaitable[float]]


def soft_q_loss(logits: np.ndarray, target: np.ndarray, actions: np.ndarray, reward: float):
    """Default v2/v3 mean, summed over time; target is detached. Return dL/dlogits."""
    values = logsumexp(logits, axis=-1)
    target_values = logsumexp(target, axis=-1)
    advantage = logits[np.arange(len(actions)), actions] - values
    next_values = np.append(target_values[1:], reward)
    temporal_error = advantage - (next_values - target_values)
    cumulative_error = np.cumsum(advantage[::-1])[::-1] - (reward - target_values)
    loss = 0.5 * np.sum(temporal_error**2 + cumulative_error**2)
    derivative = temporal_error + np.cumsum(cumulative_error)
    action_gradient = -softmax(logits, axis=-1)
    action_gradient[np.arange(len(actions)), actions] += 1
    return float(loss), derivative[:, None] * action_gradient


class RLPrompt:
    """The forward adapter exposes next-token Q logits and their parameter Jacobian."""

    def __init__(self, task: str, forward: Forward, evaluate: Evaluate):
        self.task = task
        self.forward = forward
        self.evaluate = evaluate

    async def run(
        self,
        parameters: np.ndarray,
        queries: Sequence[str],
        *,
        iterations: int = 100,
        prompt_length: int = 5,
        samples_per_query: int = 8,
        learning_rate: float = 5e-5,
        target_rate: float = 0.001,
        normalize_rewards: bool = True,
        reward_scale: float = 1.0,
        reward_offset: float = 0.0,
        seed: int = 0,
    ) -> dict:
        self.parameters = np.array(parameters, dtype=float, copy=True)
        self.target = self.parameters.copy()
        self.rng = np.random.default_rng(seed)
        self.moment = np.zeros_like(self.parameters)
        self.variance = np.zeros_like(self.parameters)
        self.history = []
        for step in range(1, iterations + 1):
            self.target += target_rate * (self.parameters - self.target)
            samples = await self.sample_batch(queries, prompt_length, samples_per_query)
            loss, gradient = self.batch_gradient(
                samples, normalize_rewards, reward_scale, reward_offset
            )
            self.update(gradient, learning_rate, step)
            self.history.append({"loss": loss, "rewards": [s[3] for s in samples]})
        prompts = [await self.decode(query, prompt_length) for query in queries]
        return {
            "parameters": self.parameters.copy(),
            "target_parameters": self.target.copy(),
            "prompts": prompts,
            "history": self.history,
            "evaluations": iterations * len(queries) * samples_per_query,
        }

    async def checked_forward(self, parameters, query, prefix):
        logits, jacobian = await self.forward(parameters.copy(), query, prefix)
        if not np.isfinite(logits).all() or not np.isfinite(jacobian).all():
            raise ValueError("Nonfinite policy logits or Jacobian")
        return np.array(logits, copy=True), np.array(jacobian, copy=True)

    async def sample_batch(self, queries, length, count):
        samples = []
        for query_index, query in enumerate(queries):
            for _ in range(count):
                prefix = ()
                logits, targets, jacobians = [], [], []
                for _ in range(length):
                    live, jacobian = await self.checked_forward(self.parameters, query, prefix)
                    target, _ = await self.checked_forward(self.target, query, prefix)
                    action = int(self.rng.choice(len(live), p=softmax(live)))
                    logits.append(live)
                    targets.append(target)
                    jacobians.append(jacobian)
                    prefix += (action,)
                reward = float(await self.evaluate(query, prefix))
                if not np.isfinite(reward):
                    raise ValueError("Nonfinite prompt reward")
                samples.append(
                    (
                        query_index,
                        prefix,
                        (np.array(logits), np.array(targets), np.array(jacobians)),
                        reward,
                    )
                )
        return samples

    def batch_gradient(self, samples, normalize, scale, offset):
        rewards = np.array([sample[3] for sample in samples])
        if normalize:
            groups = np.array([sample[0] for sample in samples])
            for group in np.unique(groups):
                mask = groups == group
                rewards[mask] = (rewards[mask] - rewards[mask].mean()) / (
                    rewards[mask].std() + 1e-4
                )
        rewards = rewards * scale + offset
        gradient = np.zeros_like(self.parameters)
        loss = 0.0
        for sample, reward in zip(samples, rewards):
            live, target, jacobian = sample[2]
            value, derivative = soft_q_loss(live, target, np.array(sample[1]), reward)
            loss += value
            gradient += np.einsum("tv,tvp->p", derivative, jacobian)
        return loss / len(samples), gradient / len(samples)

    def update(self, gradient, rate, step):
        if not np.isfinite(gradient).all():
            raise ValueError("Nonfinite soft Q gradient")
        self.moment = 0.9 * self.moment + 0.1 * gradient
        self.variance = 0.999 * self.variance + 0.001 * gradient**2
        self.parameters -= (
            rate
            * (self.moment / (1 - 0.9**step))
            / (np.sqrt(self.variance / (1 - 0.999**step)) + 1e-8)
        )
        if not np.isfinite(self.parameters).all():
            raise ValueError("Nonfinite learned parameters")

    async def decode(self, query: str, length: int) -> tuple[int, ...]:
        prefix = ()
        for _ in range(length):
            logits, _ = await self.checked_forward(self.parameters, query, prefix)
            prefix += (int(np.argmax(logits)),)
        return prefix
