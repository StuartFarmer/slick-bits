"""Jointly select ordered exemplars and instructions with transport filtering and neural UCB."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from slick import prompt
from slick.prompts import Prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Configuration:
    instruction: str
    indices: tuple[int, ...]


class EASE:
    """Own ordered-domain sampling, optimal transport, diagonal UCB and model refits.

    evaluate(instruction, ordered_examples) returns a finite higher score.
    hidden_states(texts) returns order-sensitive model representations, and
    surrogate(theta, contexts) returns predictions and parameter Jacobians only.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str, Sequence[str]], Awaitable[float]],
        hidden_states: Callable[[Sequence[str]], Awaitable[np.ndarray]],
        surrogate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.hidden_states, self.surrogate = hidden_states, surrogate
        self.context_template = Prompt("context.j2")

    @prompt(template="induce.j2")
    async def induce(self, examples: Sequence[str], *, generated: str) -> str:
        """Generate an instruction candidate for joint exemplar optimization."""
        if not generated.strip():
            raise ValueError(f"empty instruction; response={generated!r}")
        return generated.strip()

    async def _represent(self, configurations):
        texts = [
            self.context_template(
                instance=self,
                instruction=c.instruction,
                examples=[self.examples[i] for i in c.indices],
            )
            for c in configurations
        ]
        vectors = np.asarray(await self.hidden_states(texts))
        if not np.isfinite(vectors).all():
            raise ValueError("hidden-state features must be finite")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors, dtype=float), where=norms > 0)

    def transport_cost(self, indices: Sequence[int]) -> float:
        """Compute exact uniform optimal-transport cost to representative validation examples."""
        costs = 1 - self.example_features[list(indices)] @ self.validation_features.T
        rows, columns = costs.shape
        constraints = np.vstack(
            (np.kron(np.eye(rows), np.ones((1, columns))), np.tile(np.eye(columns), (1, rows)))
        )
        masses = np.concatenate((np.full(rows, 1 / rows), np.full(columns, 1 / columns)))
        result = linprog(
            costs.ravel(), A_eq=constraints, b_eq=masses, bounds=(0, None), method="highs"
        )
        if not result.success:
            raise ValueError(f"optimal transport failed: {result.message}")
        return float(result.fun)

    def _predict(self, contexts):
        values, gradients = self.surrogate(self.theta, contexts)
        values, gradients = np.asarray(values).reshape(-1), np.asarray(gradients)
        if not np.isfinite(values).all() or not np.isfinite(gradients).all():
            raise ValueError("surrogate values and Jacobians must be finite")
        return values, gradients

    def _ucb(self, contexts, regularization, exploration):
        standardized = (contexts - self.mean) / self.std
        values, gradients = self._predict(standardized)
        uncertainty = np.sqrt(np.sum(regularization * exploration * gradients**2 / self.u, axis=1))
        acquisition = values + uncertainty
        arm = int(np.argmax(acquisition))
        self.u += gradients[arm] ** 2
        return arm, float(acquisition[arm])

    async def _measure(self, configuration, context):
        if configuration in self.cache:
            self.cache_hits += 1
            score = self.cache[configuration]
        else:
            self.evaluations += 1
            score = float(
                await self.evaluate(
                    configuration.instruction, [self.examples[i] for i in configuration.indices]
                )
            )
            if not np.isfinite(score):
                raise ValueError("evaluation score must be finite")
            self.cache[configuration] = score
        self.records.append(
            {"configuration": configuration, "score": score, "context": context.copy()}
        )

    def _refit(self, steps, learning_rate, regularization):
        self.theta = self.initial_theta.copy()
        training = np.array([r["context"] for r in self.records])
        rewards = np.array([r["score"] for r in self.records])
        self.mean = training.mean(axis=0)
        deviation = training.std(axis=0, ddof=int(len(training) > 1))
        self.std = np.where(deviation > 0, deviation, 1)
        standardized = (training - self.mean) / self.std
        first, second = np.zeros_like(self.theta), np.zeros_like(self.theta)
        for step in range(1, steps + 1):
            values, gradients = self._predict(standardized)
            gradient = 2 * gradients.T @ (values - rewards) / len(rewards)
            gradient += regularization / len(rewards) * self.theta
            first, second = 0.9 * first + 0.1 * gradient, 0.999 * second + 0.001 * gradient**2
            self.theta -= (
                learning_rate
                * (first / (1 - 0.9**step))
                / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
        self.training_runs += 1

    async def _acquire(
        self,
        shots,
        proposal_sets,
        retained_sets,
        instructions_per_set,
        domain_batches,
        regularization,
        exploration,
    ):
        winner = None
        for _ in range(domain_batches):
            orders = [
                tuple(int(i) for i in self.rng.choice(len(self.examples), shots, replace=False))
                for _ in range(proposal_sets)
            ]
            costs = [self.transport_cost(order) for order in orders]
            retained = np.argsort(costs, kind="stable")[:retained_sets]
            configurations = []
            for index in retained:
                for instruction in self.rng.choice(
                    len(self.instructions), instructions_per_set, replace=False
                ):
                    configurations.append(
                        Configuration(self.instructions[instruction], orders[index])
                    )
            contexts = await self._represent(configurations)
            arm, acquisition = self._ucb(contexts, regularization, exploration)
            self.acquisitions.append(
                {
                    "costs": costs,
                    "retained": retained.tolist(),
                    "configuration": configurations[arm],
                    "ucb": acquisition,
                }
            )
            if winner is None or acquisition > winner[0]:
                winner = acquisition, configurations[arm], contexts[arm]
        return winner[1], winner[2]

    async def run(
        self,
        instructions: Sequence[str],
        examples: Sequence[str],
        validation_examples: Sequence[str],
        initial_theta: np.ndarray,
        *,
        additional_instructions: int = 0,
        shots: int = 3,
        initial_samples: int = 5,
        iterations: int = 20,
        proposal_sets: int = 50,
        retained_sets: int = 5,
        instructions_per_set: int = 1,
        domain_batches: int = 1,
        regularization: float = 0.1,
        exploration: float = 1,
        training_steps: int = 30,
        learning_rate: float = 1e-3,
        seed: int = 0,
    ) -> dict:
        """Fit initial ordered prompts, filter fresh sets by transport and acquire with UCB."""
        self.rng = np.random.default_rng(seed)
        self.examples, self.instructions = list(examples), list(instructions)
        self.initial_theta = np.asarray(initial_theta).copy()
        self.theta = self.initial_theta.copy()
        self.u = np.full(self.theta.shape, regularization, dtype=float)
        self.records, self.acquisitions, self.cache = [], [], {}
        self.optimizer_calls = self.evaluations = self.cache_hits = self.training_runs = 0
        for _ in range(additional_instructions):
            self.optimizer_calls += 1
            self.instructions.append(await self.induce(examples, provider=self.provider))
        for _ in range(initial_samples):
            configuration = Configuration(
                self.instructions[int(self.rng.integers(len(self.instructions)))],
                tuple(int(i) for i in self.rng.choice(len(examples), shots, replace=False)),
            )
            await self._measure(configuration, (await self._represent([configuration]))[0])
        self._refit(training_steps, learning_rate, regularization)
        self.example_features = await self._represent(
            [Configuration("", (i,)) for i in range(len(examples))]
        )
        vectors = np.asarray(
            await self.hidden_states(
                [
                    self.context_template(instance=self, instruction="", examples=[e])
                    for e in validation_examples
                ]
            )
        )
        if not np.isfinite(vectors).all():
            raise ValueError("validation hidden-state features must be finite")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.validation_features = np.divide(
            vectors, norms, out=np.zeros_like(vectors, dtype=float), where=norms > 0
        )
        for _ in range(iterations):
            configuration, context = await self._acquire(
                shots,
                proposal_sets,
                retained_sets,
                instructions_per_set,
                domain_batches,
                regularization,
                exploration,
            )
            await self._measure(configuration, context)
            self._refit(training_steps, learning_rate, regularization)
        return {
            "best": max(self.records, key=lambda r: r["score"]),
            "records": self.records,
            "acquisitions": self.acquisitions,
            "theta": self.theta.copy(),
            "u": self.u.copy(),
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "optimizer_calls": self.optimizer_calls,
            "training_runs": self.training_runs,
        }
