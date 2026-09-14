"""Select soft-prompt instructions with INSTINCT's diagonal neural UCB."""

from collections.abc import Awaitable, Callable, Sequence

import numpy as np
from scipy.stats import qmc
from slick import prompt
from slick.providers import Provider


class INSTINCT:
    """Own neural-UCB statistics, reset-and-refit regression, and prompt selection.

    condition supplies true soft-token prefix conditioning. hidden_states maps a
    batch of projected prefixes to the inducing model's hidden-state features.
    surrogate(theta, contexts) returns scalar predictions and parameter Jacobians;
    it computes model derivatives only. Adam fitting and UCB stay in this class.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        condition: Callable[[Provider, np.ndarray], Provider],
        hidden_states: Callable[[np.ndarray], Awaitable[np.ndarray]],
        surrogate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.condition = condition
        self.hidden_states = hidden_states
        self.surrogate = surrogate

    @prompt(template="induce.j2")
    async def induce(self, demonstrations: Sequence[str], *, generated: str) -> str:
        """Induce a hard instruction using the selected soft-conditioned provider."""
        if not generated.strip():
            raise ValueError(f"empty induced instruction; response={generated!r}")
        return generated.strip()

    async def _measure(self, arm):
        self.optimizer_calls += 1
        instruction = await self.induce(
            self.demonstrations, provider=self.condition(self.provider, self.prefixes[arm])
        )
        if instruction in self.cache:
            self.cache_hits += 1
            score = self.cache[instruction]
        else:
            self.evaluations += 1
            score = float(await self.evaluate(instruction))
            if not np.isfinite(score):
                raise ValueError("evaluation score must be finite")
            self.cache[instruction] = score
        self.records.append({"arm": int(arm), "prompt": instruction, "score": score})
        return score

    def _predict(self, contexts):
        predictions, jacobians = self.surrogate(self.theta, contexts)
        predictions = np.asarray(predictions).reshape(-1)
        jacobians = np.asarray(jacobians)
        if not np.isfinite(predictions).all() or not np.isfinite(jacobians).all():
            raise ValueError("surrogate predictions and Jacobians must be finite")
        return predictions, jacobians

    def _select(self, subset, regularization, exploration):
        contexts = self.contexts[subset]
        if self.mean is not None:
            contexts = (contexts - self.mean) / self.std
        prediction, jacobian = self._predict(contexts)
        uncertainty = np.sqrt(np.sum(regularization * exploration * jacobian**2 / self.u, axis=1))
        slot = int(np.argmax(prediction + uncertainty))
        self.u += jacobian[slot] ** 2
        self.acquisitions.append(
            {
                "arms": subset.copy(),
                "predictions": prediction.copy(),
                "uncertainty": uncertainty.copy(),
                "chosen": int(subset[slot]),
            }
        )
        return int(subset[slot])

    def _refit(self, steps, learning_rate, regularization):
        self.theta = self.initial_theta.copy()
        training = self.contexts[[r["arm"] for r in self.records]]
        rewards = np.array([r["score"] for r in self.records])
        self.mean = training.mean(axis=0)
        self.std = training.std(axis=0, ddof=1) + 1e-30
        standardized = (training - self.mean) / self.std
        first, second = np.zeros_like(self.theta), np.zeros_like(self.theta)
        for step in range(1, steps + 1):
            predictions, jacobians = self._predict(standardized)
            gradient = 2 * jacobians.T @ (predictions - rewards) / len(rewards)
            gradient += regularization / len(rewards) * self.theta
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            self.theta -= (
                learning_rate
                * (first / (1 - 0.9**step))
                / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
        self.training_runs += 1

    async def run(
        self,
        demonstrations: Sequence[str],
        projection: np.ndarray,
        initial_theta: np.ndarray,
        *,
        domain_size: int = 100,
        initial_samples: int = 5,
        iterations: int = 20,
        candidates_per_step: int | None = None,
        regularization: float = 1.0,
        exploration: float = 1.0,
        training_steps: int = 30,
        learning_rate: float = 1e-3,
        seed: int = 0,
    ) -> dict:
        """Build a Sobol prefix domain, observe seeds, select by UCB and refit.

        As in the released loop, the first UCB acquisition uses the untrained
        initial surrogate. Initial observations enter training after acquisition.
        Repeated arm choices and duplicate hard instructions remain valid.
        """
        self.demonstrations = demonstrations
        self.initial_theta = np.asarray(initial_theta).copy()
        self.theta = self.initial_theta.copy()
        self.u = np.full(self.theta.shape, regularization, dtype=float)
        self.mean = self.std = None
        self.records, self.acquisitions, self.cache = [], [], {}
        self.optimizer_calls = self.evaluations = self.cache_hits = self.training_runs = 0
        rng = np.random.default_rng(seed)
        sampler = qmc.Sobol(projection.shape[1], scramble=True, seed=seed)
        domain = sampler.random_base2(int(np.ceil(np.log2(domain_size))))[:domain_size]
        self.prefixes = domain @ projection.T
        self.contexts = np.asarray(await self.hidden_states(self.prefixes))
        if not np.isfinite(self.contexts).all():
            raise ValueError("hidden-state features must be finite")
        for arm in rng.choice(domain_size, initial_samples, replace=False):
            await self._measure(arm)
        size = domain_size if candidates_per_step is None else candidates_per_step
        for _ in range(iterations):
            subset = (
                np.arange(domain_size)
                if size == domain_size
                else rng.choice(domain_size, size, replace=False)
            )
            arm = self._select(subset, regularization, exploration)
            await self._measure(arm)
            self._refit(training_steps, learning_rate, regularization)
        return {
            "best": max(self.records, key=lambda p: p["score"]),
            "records": self.records,
            "acquisitions": self.acquisitions,
            "theta": self.theta.copy(),
            "u": self.u.copy(),
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "cache_hits": self.cache_hits,
            "training_runs": self.training_runs,
        }
