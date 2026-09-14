"""Learn sequential demonstration selection with replay-based deep Q-learning."""

import math
import random
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Transition:
    states: np.ndarray
    actions: np.ndarray
    action: int
    reward: float
    next_states: np.ndarray | None
    next_actions: np.ndarray | None


def td_loss(values: np.ndarray, action: int, target: float, conservative_weight: float) -> tuple:
    """Undiscounted L1 TD loss plus optional conservative Q-learning regularization."""
    shifted = values - values.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    logsumexp = values.max() + np.log(np.exp(shifted).sum())
    loss = abs(values[action] - target) + conservative_weight * (logsumexp - values[action])
    derivative = conservative_weight * probabilities
    derivative[action] += np.sign(values[action] - target) - conservative_weight
    return float(loss), derivative


class ActiveExampleSelection:
    """Own episodes, incremental rewards, epsilon exploration, replay and DQN updates.

    represent(demonstrations, available) returns a state vector and an action-feature
    matrix from model observations. q_model(parameters, state_history, actions)
    returns Q values and parameter Jacobians, including a final learned STOP action.
    These callbacks execute models; selection and all loss/optimizer math stay here.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Sequence[str]], Awaitable[float]],
        represent: Callable[
            [Sequence[str], Sequence[str]], Awaitable[tuple[np.ndarray, np.ndarray]]
        ],
        q_model: Callable[
            [np.ndarray, np.ndarray, np.ndarray], Awaitable[tuple[np.ndarray, np.ndarray]]
        ],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.represent, self.q_model = represent, q_model

    @prompt(template="answer.j2")
    async def answer(self, input: str, *, generated: str) -> str:
        return generated

    async def _score(self, demonstrations):
        self.evaluations += 1
        score = float(await self.evaluate(tuple(demonstrations)))
        if not math.isfinite(score):
            raise ValueError("task score must be finite")
        return score

    async def _observe(self, selected, available):
        self.representation_calls += 1
        state, actions = await self.represent(selected, available)
        state, actions = np.array(state, copy=True), np.array(actions, copy=True)
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("model observations must be finite")
        return state, actions

    async def _q(self, parameters, states, actions):
        self.q_calls += 1
        values, jacobian = await self.q_model(parameters, states, actions)
        values, jacobian = np.asarray(values), np.asarray(jacobian)
        if not np.isfinite(values).all() or not np.isfinite(jacobian).all():
            raise ValueError("Q values and Jacobians must be finite")
        return values, jacobian

    async def _episode(self, pool, epsilon, max_examples, cost, terminal_bonus, remember):
        selected, remaining = [], list(range(len(pool)))
        previous, peak = self.baseline, self.baseline
        state, actions = await self._observe((), pool)
        states = state[None]
        rewards = []
        while remaining and len(selected) < max_examples:
            if self.rng.random() < epsilon:
                action = self.rng.randrange(len(remaining) + 1)
            else:
                values, _ = await self._q(self.parameters, states, actions)
                action = int(np.argmax(values))
            stopped = action == len(remaining)
            if stopped:
                reward = terminal_bonus * (peak - self.baseline)
                terminal = True
            else:
                selected.append(remaining.pop(action))
                score = await self._score([pool[i] for i in selected])
                reward, previous = score - previous - cost, score
                peak = max(peak, score)
                terminal = len(selected) == max_examples or not remaining
                if terminal:
                    reward += terminal_bonus * (peak - self.baseline)
            next_states = next_actions = None
            if not terminal:
                state, next_actions = await self._observe(
                    [pool[i] for i in selected], [pool[i] for i in remaining]
                )
                next_states = np.concatenate((states, state[None]))
            if remember:
                self.replay.append(
                    Transition(
                        states.copy(),
                        actions.copy(),
                        action,
                        reward,
                        None if next_states is None else next_states.copy(),
                        None if next_actions is None else next_actions.copy(),
                    )
                )
            rewards.append(reward)
            if terminal:
                break
            states, actions = next_states, next_actions
        return {"indices": selected, "score": previous, "rewards": rewards}

    async def _optimize(self, batch_size, conservative_weight, learning_rate, weight_decay):
        if len(self.replay) < batch_size:
            return None
        batch = self.rng.sample(list(self.replay), batch_size)
        # Preserve the released trainer's minimum nonterminal replay requirement.
        if sum(row.next_states is not None for row in batch) < 2:
            return None
        gradient, losses = np.zeros_like(self.parameters), []
        for row in batch:
            values, jacobian = await self._q(self.parameters, row.states, row.actions)
            target = row.reward
            if row.next_states is not None:
                future, _ = await self._q(self.target_parameters, row.next_states, row.next_actions)
                target += float(future.max())
            loss, derivative = td_loss(values, row.action, target, conservative_weight)
            gradient += derivative @ jacobian / batch_size
            losses.append(loss)
        gradient *= min(1, 10 / (np.linalg.norm(gradient) + 1e-12))
        self.updates += 1
        self.first = 0.9 * self.first + 0.1 * gradient
        self.second = 0.999 * self.second + 0.001 * gradient**2
        update = (
            self.first
            / (1 - 0.9**self.updates)
            / (np.sqrt(self.second / (1 - 0.999**self.updates)) + 1e-8)
        )
        self.parameters *= 1 - learning_rate * weight_decay
        self.parameters -= learning_rate * update
        return float(np.mean(losses))

    async def run(
        self,
        pool: Sequence[str],
        parameters: np.ndarray,
        *,
        episodes: int = 100,
        max_examples: int = 8,
        batch_size: int = 4,
        replay_size: int = 1000,
        epsilon_start: float = 0.99,
        epsilon_end: float = 0.2,
        target_update_every: int = 10,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
        cost_per_example: float = 0.0,
        terminal_bonus: float = 0.0,
        conservative_weight: float = 0.0,
        seed: int = 0,
    ) -> dict:
        self.rng = random.Random(seed)
        self.parameters = np.array(parameters, dtype=float, copy=True)
        self.target_parameters = self.parameters.copy()
        self.first, self.second = np.zeros_like(self.parameters), np.zeros_like(self.parameters)
        self.evaluations = self.representation_calls = self.q_calls = self.updates = 0
        self.replay = deque(maxlen=replay_size)
        self.baseline = await self._score(())
        history = []
        for episode in range(episodes):
            fraction = episode / max(episodes - 1, 1)
            epsilon = epsilon_start * (epsilon_end / epsilon_start) ** fraction
            rollout = await self._episode(
                pool, epsilon, max_examples, cost_per_example, terminal_bonus, True
            )
            loss = await self._optimize(
                batch_size, conservative_weight, learning_rate, weight_decay
            )
            history.append({**rollout, "epsilon": epsilon, "loss": loss})
            if (episode + 1) % target_update_every == 0:
                self.target_parameters = self.parameters.copy()
        selected = await self._episode(
            pool, 0.0, max_examples, cost_per_example, terminal_bonus, False
        )
        self.demonstrations = tuple(pool[i] for i in selected["indices"])
        return {
            "demonstrations": self.demonstrations,
            **selected,
            "parameters": self.parameters.copy(),
            "history": history,
            "updates": self.updates,
            "evaluations": self.evaluations,
            "representation_calls": self.representation_calls,
            "q_calls": self.q_calls,
        }
