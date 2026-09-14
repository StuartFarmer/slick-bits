"""Identify a prompt under a fixed evaluation budget using TRIPLE's BAI rules."""

import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal


class EPO:
    """Select from caller-supplied prompts; each evaluation is one noisy arm pull.

    Higher rewards win. Continuous rejection assumes rewards in [0, 1], as in
    the paper's bounded-reward analysis. No score caching or model generation.
    Errors propagate; counters include attempted evaluations.
    """

    def __init__(self, task: str, evaluate: Callable[[str], Awaitable[float]]):
        self.task = task
        self.evaluate = evaluate

    async def _pull(self, arm):
        self.evaluations += 1
        reward = float(await self.evaluate(self.prompts[arm]))
        if not math.isfinite(reward):
            raise ValueError("measured reward must be finite")
        self.pulls[arm] += 1
        n = self.pulls[arm]
        self.means[arm] = reward if n == 1 else self.means[arm] * ((n - 1) / n) + reward / n

    async def _halve(self):
        rounds = max(1, math.ceil(math.log2(len(self.prompts))))
        while self.evaluations < self.budget:
            per_arm = max(1, self.budget // (rounds * len(self.active)))
            for arm in self.active:
                for _ in range(min(per_arm, self.budget - self.evaluations)):
                    await self._pull(arm)
            if len(self.active) > 1:
                self.active.sort(key=lambda arm: self.means[arm], reverse=True)
                self.active = self.active[: math.ceil(len(self.active) / 2)]
                self.history.append(tuple(self.active))

    def _reject(self):
        if len(self.active) < 2 or any(self.pulls[arm] == 0 for arm in self.active):
            return
        worst = min(self.active, key=lambda arm: self.means[arm])
        if self.rejected and self.pulls[worst] <= max(self.pulls[i] for i in self.rejected):
            return
        harmonic = 0.5 + sum(1 / i for i in range(2, len(self.active) + 1))
        available = self.budget - sum(self.pulls[i] for i in self.rejected)
        beta = harmonic * sum(self.pulls[i] for i in self.active) / available
        threshold = 1 / math.sqrt(beta) - 1
        others = [self.means[i] for i in self.active if i != worst]
        # The official implementation enables CR-A OR CR-C; CR-A subsumes CR-C.
        gap = sum(others) / len(others) - self.means[worst]
        if gap > threshold:
            self.active.remove(worst)
            self.rejected.append(worst)
            self.history.append(tuple(self.active))

    async def _continuous(self):
        while self.evaluations < self.budget:
            arm = min(self.active, key=lambda i: self.pulls[i])
            await self._pull(arm)
            self._reject()

    async def run(
        self,
        prompts: Sequence[str],
        *,
        budget: int = 100,
        algorithm: Literal["sequential_halving", "continuous_rejects"] = "sequential_halving",
    ) -> dict:
        """Allocate pulls, eliminate arms, and return the best measured survivor."""
        self.prompts = list(prompts)
        self.budget = budget
        self.evaluations = 0
        self.pulls = [0] * len(prompts)
        self.means = [-math.inf] * len(prompts)
        self.active = list(range(len(prompts)))
        self.rejected = []
        self.history = [tuple(self.active)]
        await {"sequential_halving": self._halve, "continuous_rejects": self._continuous}[
            algorithm
        ]()
        measured = [i for i in self.active if self.pulls[i]]
        best = max(measured, key=lambda i: self.means[i])
        return {
            "best": {"prompt": self.prompts[best], "score": self.means[best]},
            "pulls": self.pulls.copy(),
            "means": self.means.copy(),
            "survivors": self.active.copy(),
            "history": self.history.copy(),
            "evaluations": self.evaluations,
        }
