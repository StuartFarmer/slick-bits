"""Train a PPO policy for query-conditioned exemplar and verbalizer edits."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.special import log_softmax, softmax


@dataclass(frozen=True)
class PromptState:
    examples: tuple[int, ...]
    verbalizers: tuple[int, ...]


@dataclass(frozen=True)
class Edit:
    kind: str
    first: int = 0
    second: int = 0


Encode = Callable[[str, PromptState, tuple[Edit, ...]], Awaitable[np.ndarray]]
Forward = Callable[
    [np.ndarray, np.ndarray], Awaitable[tuple[np.ndarray, np.ndarray, float, np.ndarray]]
]
Evaluate = Callable[[str, PromptState], Awaitable[float]]


def ppo_loss(
    logits, value, action, old_logprob, old_value, advantage, target, clip, value_coef, entropy_coef
):
    """Single-transition clipped policy/value objective and derivatives."""
    logprob = log_softmax(logits)
    probabilities = np.exp(logprob)
    ratio = np.exp(logprob[action] - old_logprob)
    clipped_ratio = np.clip(ratio, 1 - clip, 1 + clip)
    policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
    active = (advantage >= 0 and ratio <= 1 + clip) or (advantage < 0 and ratio >= 1 - clip)
    policy_gradient = probabilities.copy()
    policy_gradient[action] -= 1
    policy_gradient *= ratio * advantage * active
    clipped_value = old_value + np.clip(value - old_value, -clip, clip)
    raw_error, clipped_error = value - target, clipped_value - target
    value_loss = 0.5 * max(raw_error**2, clipped_error**2)
    value_gradient = raw_error
    if clipped_error**2 > raw_error**2:
        value_gradient = clipped_error * (abs(value - old_value) <= clip)
    entropy = -float(probabilities @ logprob)
    policy_gradient += entropy_coef * probabilities * (logprob + entropy)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return float(loss), policy_gradient, value_coef * value_gradient


def generalized_advantages(rewards, values, gamma, lam):
    advantages = np.zeros(len(rewards))
    future = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        future = delta + gamma * lam * future
        advantages[index] = future
    return advantages, advantages + values


class TEMPERA:
    """Released exemplar/verbalizer action family; test-time editing uses no evaluator."""

    def __init__(
        self,
        task: str,
        encode: Encode,
        forward: Forward,
        evaluate: Evaluate,
        *,
        pool_size: int,
        verbalizer_count: int,
    ):
        self.task, self.encode, self.forward, self.evaluate = task, encode, forward, evaluate
        self.pool_size, self.verbalizer_count = pool_size, verbalizer_count

    def actions(self, state: PromptState) -> tuple[Edit, ...]:
        edits = [Edit("stop")]
        edits.extend(Edit("swap", i, j) for i, j in combinations(range(len(state.examples)), 2))
        edits.extend(
            Edit("replace", i, candidate)
            for i in range(len(state.examples))
            for candidate in range(self.pool_size)
            if candidate not in state.examples
        )
        edits.extend(
            Edit("verbalize", i, verbalizer)
            for i in range(len(state.examples))
            for verbalizer in range(self.verbalizer_count)
        )
        return tuple(edits)

    def apply(self, state: PromptState, edit: Edit) -> PromptState:
        examples, verbalizers = list(state.examples), list(state.verbalizers)
        i, j = edit.first, edit.second
        if edit.kind == "swap":
            examples[i], examples[j] = examples[j], examples[i]
            verbalizers[i], verbalizers[j] = verbalizers[j], verbalizers[i]
        elif edit.kind == "replace":
            examples[i] = j
        elif edit.kind == "verbalize":
            verbalizers[i] = j
        return PromptState(tuple(examples), tuple(verbalizers))

    async def run(
        self,
        parameters: np.ndarray,
        training_queries: Sequence[str],
        initial: PromptState,
        *,
        iterations: int = 10,
        max_edits: int = 8,
        epochs: int = 4,
        learning_rate: float = 6e-4,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        seed: int = 0,
    ) -> dict:
        self.parameters = np.array(parameters, dtype=float, copy=True)
        self.rng = np.random.default_rng(seed)
        self.mean, self.m2, self.count = None, None, 0
        self.moment, self.variance = np.zeros_like(parameters), np.zeros_like(parameters)
        self.updates, self.evaluations = 0, 0
        history = []
        for _ in range(iterations):
            transitions = await self.collect(
                training_queries, initial, max_edits, gamma, gae_lambda
            )
            losses = await self.learn(
                transitions, epochs, learning_rate, clip, value_coef, entropy_coef, max_grad_norm
            )
            history.append({"losses": losses, "transitions": len(transitions)})
        return {
            "parameters": self.parameters.copy(),
            "history": history,
            "evaluations": self.evaluations,
            "normalization_count": self.count,
        }

    async def features(self, query, state, actions, training):
        raw = np.array(await self.encode(query, state, actions), dtype=float, copy=True)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite environment observation")
        if training:
            if self.mean is None:
                self.mean, self.m2 = np.zeros_like(raw), np.zeros_like(raw)
            self.count += 1
            delta = raw - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (raw - self.mean)
        if self.count == 0:
            return raw
        return np.clip((raw - self.mean) / np.sqrt(self.m2 / self.count + 1e-8), -10, 10)

    async def policy(self, features):
        logits, jacobian, value, value_gradient = await self.forward(
            self.parameters.copy(), features.copy()
        )
        if not all(np.isfinite(item).all() for item in (logits, jacobian, value, value_gradient)):
            raise ValueError("Nonfinite policy outputs")
        return (
            np.array(logits, copy=True),
            np.array(jacobian, copy=True),
            float(value),
            np.array(value_gradient, copy=True),
        )

    async def score(self, query, state):
        value = float(await self.evaluate(query, state))
        self.evaluations += 1
        if not np.isfinite(value):
            raise ValueError("Nonfinite environment score")
        return value

    async def collect(self, queries, initial, max_edits, gamma, lam):
        transitions = []
        for query in queries:
            state, score = initial, await self.score(query, initial)
            episode, rewards, values = [], [], []
            for _ in range(max_edits):
                actions = self.actions(state)
                features = await self.features(query, state, actions, True)
                logits, _, value, _ = await self.policy(features)
                action = int(self.rng.choice(len(actions), p=softmax(logits)))
                next_state = self.apply(state, actions[action])
                next_score = (
                    score if actions[action].kind == "stop" else await self.score(query, next_state)
                )
                episode.append((features, action, float(log_softmax(logits)[action]), value))
                rewards.append(next_score - score)
                values.append(value)
                state, score = next_state, next_score
                if actions[action].kind == "stop":
                    break
            advantages, targets = generalized_advantages(
                np.array(rewards), np.array(values), gamma, lam
            )
            transitions.extend(
                (*transition, advantage, target)
                for transition, advantage, target in zip(episode, advantages, targets)
            )
        return transitions

    async def learn(self, transitions, epochs, rate, clip, value_coef, entropy_coef, max_norm):
        advantages = np.array([transition[4] for transition in transitions])
        std = advantages.std(ddof=1) if len(advantages) > 1 else 0.0
        advantages = (advantages - advantages.mean()) / (std + 1e-5)
        losses = []
        for _ in range(epochs):
            gradient, loss = np.zeros_like(self.parameters), 0.0
            for transition, advantage in zip(transitions, advantages):
                features, action, old_logprob, old_value, _, target = transition
                logits, jacobian, value, value_gradient = await self.policy(features)
                value_loss, dl, dv = ppo_loss(
                    logits,
                    value,
                    action,
                    old_logprob,
                    old_value,
                    advantage,
                    target,
                    clip,
                    value_coef,
                    entropy_coef,
                )
                loss += value_loss
                gradient += dl @ jacobian + dv * value_gradient
            gradient /= len(transitions)
            if not np.isfinite(gradient).all():
                raise ValueError("Nonfinite PPO gradient")
            gradient *= min(1.0, max_norm / (np.linalg.norm(gradient) + 1e-6))
            self.updates += 1
            self.moment = 0.9 * self.moment + 0.1 * gradient
            self.variance = 0.999 * self.variance + 0.001 * gradient**2
            self.parameters -= (
                rate
                * (self.moment / (1 - 0.9**self.updates))
                / (np.sqrt(self.variance / (1 - 0.999**self.updates)) + 1e-5)
            )
            if not np.isfinite(self.parameters).all():
                raise ValueError("Nonfinite PPO parameters")
            losses.append(loss / len(transitions))
        return losses

    async def edit(self, query: str, initial: PromptState, *, max_edits: int = 8) -> PromptState:
        state = initial
        for _ in range(max_edits):
            actions = self.actions(state)
            features = await self.features(query, state, actions, False)
            logits, _, _, _ = await self.policy(features)
            action = actions[int(np.argmax(logits))]
            if action.kind == "stop":
                break
            state = self.apply(state, action)
        return state
