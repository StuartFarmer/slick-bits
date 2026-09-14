"""Filter prompt candidates using draft/target rank agreement on a probe set."""

import math
import random
from collections.abc import Awaitable, Callable
from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Candidates(BaseModel, extra="forbid"):
    prompts: list[Text] = Field(min_length=1)


def _ranks(values):
    """Average ranks for ties, as in scipy.stats.spearmanr."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = np.empty(len(values))
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[start]] == values[ordered[end]]:
            end += 1
        ranks[ordered[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


class ProbeSampling:
    """Use two caller-owned loss evaluators (lower is better) and a Slick operator.

    This is the general prompt-optimization application of the filtering rule.
    Generation replaces the official GCG token proposal step; the probe/rank/filter
    algorithm is unchanged. Evaluations are reused only within a single batch.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        draft_evaluate: Callable[[str], Awaitable[float]],
        target_evaluate: Callable[[str], Awaitable[float]],
    ):
        self.task, self.provider = task, provider
        self.draft_evaluate, self.target_evaluate = draft_evaluate, target_evaluate

    @prompt(template="propose.j2", output_type=Candidates)
    async def propose(self, current: str, count: int, *, generated: Candidates) -> list[str]:
        """Generate one fixed-size batch before either model evaluates it."""
        if len(generated.prompts) != count:
            raise ValueError("generated batch has the wrong number of candidates")
        return generated.prompts

    async def _measure(self, text, *, draft=False):
        if draft:
            self.draft_evaluations += 1
            score = float(await self.draft_evaluate(text))
        else:
            self.target_evaluations += 1
            score = float(await self.target_evaluate(text))
        if not math.isfinite(score):
            raise ValueError("measured loss must be finite")
        return score

    async def _filter(self, candidates, probe_size, filtered_size):
        draft = [await self._measure(text, draft=True) for text in candidates]
        probes = self.rng.sample(range(len(candidates)), min(probe_size, len(candidates)))
        target = {i: await self._measure(candidates[i]) for i in probes}
        x, y = _ranks([draft[i] for i in probes]), _ranks([target[i] for i in probes])
        x, y = x - x.mean(), y - y.mean()
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        correlation = float(np.clip(x @ y / denominator, -1, 1)) if denominator else None
        keep = (
            filtered_size
            if correlation is None
            else max(int((1 - correlation) / 2 * filtered_size), 1)
        )
        selected = sorted(range(len(candidates)), key=draft.__getitem__)[:keep]
        for i in selected:
            if i not in target:
                target[i] = await self._measure(candidates[i])
        best = min(target, key=target.__getitem__)
        self.history.append(
            {"correlation": correlation, "retained": len(selected), "probes": probes}
        )
        return {"prompt": candidates[best], "loss": target[best]}

    async def run(
        self,
        initial: str,
        *,
        iterations: int = 10,
        candidates: int = 16,
        probe_size: int = 4,
        filtered_size: int = 8,
        seed: int = 0,
    ) -> dict:
        """Propose, probe, filter and select; retain the global target-loss minimum."""
        self.rng = random.Random(seed)
        self.draft_evaluations = self.target_evaluations = self.optimizer_calls = 0
        self.history = []
        current = best = {"prompt": initial, "loss": await self._measure(initial)}
        for _ in range(iterations):
            self.optimizer_calls += 1
            batch = await self.propose(current["prompt"], candidates, provider=self.provider)
            current = await self._filter(batch, probe_size, filtered_size)
            if current["loss"] < best["loss"]:
                best = current
        return {
            "best": best,
            "current": current,
            "history": self.history.copy(),
            "draft_evaluations": self.draft_evaluations,
            "target_evaluations": self.target_evaluations,
            "optimizer_calls": self.optimizer_calls,
        }
