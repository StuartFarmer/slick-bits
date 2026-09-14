"""Learn a reflection model from return improvements, reward preferences and PPO."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from slick import Prompt


@dataclass(frozen=True)
class Episode:
    trajectory: str
    score: float
    success: bool


@dataclass(frozen=True)
class RatedReflection:
    context: str
    text: str
    rating: float


Rollout = Callable[[str, str], Awaitable[Episode]]
Generate = Callable[[np.ndarray, str, int], Awaitable[str]]
Policy = Callable[[np.ndarray, str, str], Awaitable[tuple[float, np.ndarray, float, np.ndarray]]]
Reward = Callable[[np.ndarray, str, str], Awaitable[tuple[float, np.ndarray]]]


def preference_loss(chosen: float, rejected: float):
    difference = chosen - rejected
    derivative = -float(expit(-difference))
    return float(np.logaddexp(0, -difference)), derivative, -derivative


def sequence_ppo_loss(logprob, value, old_logprob, old_value, advantage, target, clip, value_coef):
    ratio = np.exp(logprob - old_logprob)
    clipped_ratio = np.clip(ratio, 1 - clip, 1 + clip)
    policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
    active = (advantage >= 0 and ratio <= 1 + clip) or (advantage < 0 and ratio >= 1 - clip)
    dl = -ratio * advantage * active
    raw = value - target
    clipped = old_value + np.clip(value - old_value, -clip, clip) - target
    dv = raw if raw**2 >= clipped**2 else clipped * (abs(value - old_value) <= clip)
    return float(policy_loss + 0.5 * value_coef * max(raw**2, clipped**2)), dl, value_coef * dv


class Retroformer:
    """One-step reflection-bandit reconstruction of the paper's offline RLHF pipeline."""

    def __init__(
        self, task: str, rollout: Rollout, generate: Generate, policy: Policy, reward: Reward
    ):
        self.task, self.rollout, self.generate = task, rollout, generate
        self.policy, self.reward = policy, reward

    async def run(
        self,
        parameters: np.ndarray,
        reward_parameters: np.ndarray,
        training_queries: Sequence[str],
        *,
        trials: int = 3,
        sft_epochs: int = 2,
        reward_epochs: int = 3,
        ppo_iterations: int = 4,
        ppo_epochs: int = 4,
        sft_rate: float = 1e-5,
        reward_rate: float = 2.5e-5,
        policy_rate: float = 1.4e-5,
        kl_coefficient: float = 0.1,
        clip: float = 0.2,
        value_coef: float = 0.5,
        seed: int = 0,
    ) -> dict:
        self.parameters, self.reward_parameters = (
            np.array(parameters, dtype=float, copy=True),
            np.array(reward_parameters, dtype=float, copy=True),
        )
        self.rng = np.random.default_rng(seed)
        self.environment_calls, self.generations = 0, 0
        pairs, ratings = await self.collect(training_queries, trials)
        sft_losses = await self.supervised(ratings, sft_epochs, sft_rate)
        reward_losses = await self.train_reward(pairs, reward_epochs, reward_rate)
        self.reference = self.parameters.copy()
        contexts = list(dict.fromkeys(rating.context for rating in ratings))
        policy_losses = []
        self.policy_moment, self.policy_variance, self.policy_steps = (
            np.zeros_like(self.parameters),
            np.zeros_like(self.parameters),
            0,
        )
        for _ in range(ppo_iterations):
            samples = await self.policy_batch(contexts, kl_coefficient)
            policy_losses.extend(
                await self.train_policy(samples, ppo_epochs, policy_rate, clip, value_coef)
            )
        return {
            "parameters": self.parameters.copy(),
            "reward_parameters": self.reward_parameters.copy(),
            "ratings": ratings,
            "preference_pairs": pairs,
            "sft_losses": sft_losses,
            "reward_losses": reward_losses,
            "policy_losses": policy_losses,
            "environment_calls": self.environment_calls,
            "generations": self.generations,
        }

    def context(self, query, episode):
        return Prompt("reflect.j2")(
            task=self.task, query=query, trajectory=episode.trajectory, score=episode.score
        )

    async def execute(self, query, reflection):
        episode = await self.rollout(query, reflection)
        self.environment_calls += 1
        if not np.isfinite(episode.score):
            raise ValueError("Nonfinite environment return")
        return episode

    async def reflect(self, context):
        text = await self.generate(self.parameters.copy(), context, int(self.rng.integers(2**31)))
        self.generations += 1
        if not text.strip():
            raise ValueError("Empty generated reflection")
        return text

    async def collect(self, queries, trials):
        pairs, ratings = [], []
        for query in queries:
            episode = await self.execute(query, "")
            for _ in range(trials):
                if episode.success:
                    break
                context = self.context(query, episode)
                candidates = []
                for _ in range(2):
                    text = await self.reflect(context)
                    outcome = await self.execute(query, text)
                    rating = RatedReflection(context, text, outcome.score - episode.score)
                    ratings.append(rating)
                    candidates.append((rating, outcome))
                candidates.sort(key=lambda item: item[0].rating, reverse=True)
                if candidates[0][0].rating > candidates[1][0].rating:
                    pairs.append((candidates[0][0], candidates[1][0]))
                episode = candidates[0][1]
        return pairs, ratings

    async def policy_values(self, parameters, context, text):
        logprob, gradient, value, value_gradient = await self.policy(
            parameters.copy(), context, text
        )
        if not all(np.isfinite(item).all() for item in (logprob, gradient, value, value_gradient)):
            raise ValueError("Nonfinite reflection policy output")
        return (
            float(logprob),
            np.array(gradient, copy=True),
            float(value),
            np.array(value_gradient, copy=True),
        )

    async def reward_value(self, context, text):
        value, gradient = await self.reward(self.reward_parameters.copy(), context, text)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise ValueError("Nonfinite reward model output")
        return float(value), np.array(gradient, copy=True)

    def adam(self, parameters, gradient, moment, variance, step, rate):
        if not np.isfinite(gradient).all():
            raise ValueError("Nonfinite learning gradient")
        moment = 0.9 * moment + 0.1 * gradient
        variance = 0.999 * variance + 0.001 * gradient**2
        parameters = parameters - rate * (moment / (1 - 0.9**step)) / (
            np.sqrt(variance / (1 - 0.999**step)) + 1e-8
        )
        if not np.isfinite(parameters).all():
            raise ValueError("Nonfinite learned parameters")
        return parameters, moment, variance

    async def supervised(self, ratings, epochs, rate):
        positives = [rating for rating in ratings if rating.rating > 0]
        moment, variance, step, losses = (
            np.zeros_like(self.parameters),
            np.zeros_like(self.parameters),
            0,
            [],
        )
        for _ in range(epochs):
            for rating in positives:
                logprob, gradient, _, _ = await self.policy_values(
                    self.parameters, rating.context, rating.text
                )
                step += 1
                self.parameters, moment, variance = self.adam(
                    self.parameters, -gradient, moment, variance, step, rate
                )
                losses.append(-logprob)
        return losses

    async def train_reward(self, pairs, epochs, rate):
        moment, variance, step, losses = (
            np.zeros_like(self.reward_parameters),
            np.zeros_like(self.reward_parameters),
            0,
            [],
        )
        for _ in range(epochs):
            for chosen, rejected in pairs:
                high, dh = await self.reward_value(chosen.context, chosen.text)
                low, dl = await self.reward_value(rejected.context, rejected.text)
                loss, gh, gl = preference_loss(high, low)
                step += 1
                self.reward_parameters, moment, variance = self.adam(
                    self.reward_parameters, gh * dh + gl * dl, moment, variance, step, rate
                )
                losses.append(loss)
        return losses

    async def policy_batch(self, contexts, beta):
        samples = []
        for context in contexts:
            text = await self.reflect(context)
            logprob, _, value, _ = await self.policy_values(self.parameters, context, text)
            reference, _, _, _ = await self.policy_values(self.reference, context, text)
            reward, _ = await self.reward_value(context, text)
            target = reward - beta * (logprob - reference)
            samples.append((context, text, logprob, value, target - value, target))
        return samples

    async def train_policy(self, samples, epochs, rate, clip, value_coef):
        if not samples:
            return []
        losses = []
        for _ in range(epochs):
            gradient, loss = np.zeros_like(self.parameters), 0.0
            for context, text, old_logprob, old_value, advantage, target in samples:
                logprob, dlogprob, value, dvalue = await self.policy_values(
                    self.parameters, context, text
                )
                value_loss, dl, dv = sequence_ppo_loss(
                    logprob, value, old_logprob, old_value, advantage, target, clip, value_coef
                )
                loss += value_loss
                gradient += dl * dlogprob + dv * dvalue
            self.policy_steps += 1
            self.parameters, self.policy_moment, self.policy_variance = self.adam(
                self.parameters,
                gradient / len(samples),
                self.policy_moment,
                self.policy_variance,
                self.policy_steps,
                rate,
            )
            losses.append(loss / len(samples))
        return losses

    async def improve(self, query: str, *, trials: int = 3, best_of: int = 4) -> dict:
        episode = await self.execute(query, "")
        reflections = []
        for _ in range(trials):
            if episode.success:
                break
            context = self.context(query, episode)
            candidates = []
            for _ in range(best_of):
                text = await self.reflect(context)
                reward, _ = await self.reward_value(context, text)
                candidates.append((reward, text))
            text = max(candidates, key=lambda pair: pair[0])[1]
            reflections.append(text)
            episode = await self.execute(query, text)
        return {"episode": episode, "reflections": reflections}
