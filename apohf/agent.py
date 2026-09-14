"""Optimize instructions from pairwise feedback using APOHF's neural dueling UCB."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

import numpy as np
from scipy.special import expit
from slick import prompt
from slick.providers import Provider


class APOHF:
    """Own pair acquisition, diagonal uncertainty and Bradley-Terry model fitting.

    evaluate(left, right) returns preference for left: 1 left, 0 right, or 0.5
    a tie. embed supplies instruction vectors. surrogate(theta, vectors) returns
    utilities and their parameter Jacobians; numerical learning stays local.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str, str], Awaitable[float]],
        embed: Callable[[Sequence[str]], Awaitable[np.ndarray]],
        surrogate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.embed, self.surrogate = embed, surrogate

    @prompt(template="induce.j2")
    async def induce(self, examples: Sequence[str], *, generated: str) -> str:
        """Create a candidate instruction from demonstration evidence."""
        if not generated.strip():
            raise ValueError(f"empty induced candidate; response={generated!r}")
        return generated.strip()

    @prompt(template="paraphrase.j2")
    async def paraphrase(self, instruction: str, *, generated: str) -> str:
        """Create a rephrased candidate for the fixed comparison domain."""
        if not generated.strip():
            raise ValueError(f"empty paraphrase; response={generated!r}")
        return generated.strip()

    async def _initialize(self, initial_prompts, examples, additional_candidates, candidate_method):
        candidates = list(initial_prompts)
        for _ in range(additional_candidates):
            self.optimizer_calls += 1
            if candidate_method == "induction":
                candidates.append(await self.induce(examples, provider=self.provider))
            else:
                candidates.append(await self.paraphrase(initial_prompts[0], provider=self.provider))
        self.prompts = list(dict.fromkeys(candidates))
        self.contexts = np.asarray(await self.embed(self.prompts))
        if not np.isfinite(self.contexts).all():
            raise ValueError("instruction embeddings must be finite")

    def _predict(self):
        means, gradients = self.surrogate(self.theta, self.contexts)
        means, gradients = np.asarray(means).reshape(-1), np.asarray(gradients)
        if not np.isfinite(means).all() or not np.isfinite(gradients).all():
            raise ValueError("surrogate utilities and Jacobians must be finite")
        return means, gradients

    async def _compare(self, left, right):
        self.comparisons += 1
        preference = float(await self.evaluate(self.prompts[left], self.prompts[right]))
        if not np.isfinite(preference) or not 0 <= preference <= 1:
            raise ValueError("measured preference must be finite and between zero and one")
        self.pairs.append((int(left), int(right)))
        self.preferences.append(preference)

    def _select_pair(self, regularization, exploration):
        means, gradients = self._predict()
        left = int(np.argmax(means))
        pairs = np.array(self.pairs)
        differences = gradients[pairs[:, 0]] - gradients[pairs[:, 1]]
        self.u = regularization + np.sum(differences**2, axis=0)
        relative = gradients - gradients[left]
        uncertainty = np.sqrt(np.sum(exploration * relative**2 / self.u, axis=1))
        acquisition = means + uncertainty
        acquisition[left] = -np.inf
        right = int(np.argmax(acquisition))
        self.acquisitions.append(
            {
                "pair": (left, right),
                "means": means.copy(),
                "uncertainty": uncertainty.copy(),
                "u": self.u.copy(),
            }
        )
        return left, right

    def _refit(self, training_steps, learning_rate, regularization):
        self.theta = self.initial_theta.copy()
        first, second = np.zeros_like(self.theta), np.zeros_like(self.theta)
        pairs, labels = np.array(self.pairs), np.array(self.preferences)
        for step in range(1, training_steps + 1):
            means, gradients = self._predict()
            logits = means[pairs[:, 0]] - means[pairs[:, 1]]
            differences = gradients[pairs[:, 0]] - gradients[pairs[:, 1]]
            gradient = differences.T @ (expit(logits) - labels) / len(pairs)
            gradient += regularization / (len(pairs) + 50) * self.theta
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
        initial_prompts: Sequence[str],
        initial_theta: np.ndarray,
        *,
        examples: Sequence[str] = (),
        additional_candidates: int = 0,
        candidate_method: Literal["induction", "rephrase"] = "induction",
        initial_pairs: int = 5,
        iterations: int = 20,
        regularization: float = 1.0,
        exploration: float = 1.0,
        training_steps: int = 30,
        learning_rate: float = 1e-3,
        seed: int = 0,
    ) -> dict:
        """Build a fixed domain, compare seed pairs, fit, then acquire informative duels."""
        self.initial_theta = np.asarray(initial_theta).copy()
        self.theta = self.initial_theta.copy()
        self.pairs, self.preferences, self.acquisitions = [], [], []
        self.optimizer_calls = self.comparisons = self.training_runs = 0
        self.u = np.full(self.theta.shape, regularization, dtype=float)
        rng = np.random.default_rng(seed)
        await self._initialize(initial_prompts, examples, additional_candidates, candidate_method)
        for _ in range(initial_pairs):
            await self._compare(*rng.choice(len(self.prompts), 2, replace=False))
        self._refit(training_steps, learning_rate, regularization)
        for _ in range(iterations):
            await self._compare(*self._select_pair(regularization, exploration))
            self._refit(training_steps, learning_rate, regularization)
        means, _ = self._predict()
        queried = list(dict.fromkeys(index for pair in self.pairs for index in pair))
        best = max(queried, key=lambda index: means[index])
        return {
            "best": {"prompt": self.prompts[best], "utility": float(means[best]), "arm": best},
            "prompts": self.prompts,
            "pairs": self.pairs,
            "preferences": self.preferences,
            "theta": self.theta.copy(),
            "acquisitions": self.acquisitions,
            "comparisons": self.comparisons,
            "optimizer_calls": self.optimizer_calls,
            "training_runs": self.training_runs,
        }
