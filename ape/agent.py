"""Induce instructions from demonstrations and allocate evaluations with APE's UCB."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence

from slick import prompt
from slick.providers import Provider


class APE:
    """Own instruction induction and batch UCB; higher finite scores are better.

    Each evaluator call returns the mean of an equally sized fresh sample.
    Generation, parsing and evaluator errors propagate without retry.
    """

    def __init__(self, task: str, provider: Provider, evaluate: Callable[[str], Awaitable[float]]):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate

    @prompt(template="induce.j2")
    async def induce(self, demonstrations: Sequence[str], *, generated: str) -> str:
        """Generate one instruction from sampled input-output demonstrations."""
        if not generated.strip():
            raise ValueError(f"empty induced instruction; response={generated!r}")
        return generated.strip()

    async def _induce_pool(self, demonstrations, subsamples, demos_per_sample, prompts_per_sample):
        pool = []
        for _ in range(subsamples):
            examples = self.rng.sample(list(demonstrations), demos_per_sample)
            for _ in range(prompts_per_sample):
                self.optimizer_calls += 1
                pool.append(await self.induce(examples, provider=self.provider))
        return list(dict.fromkeys(pool))

    def _choose(self, size, exploration):
        if not sum(self.counts):
            return self.rng.sample(range(len(self.pool)), size)
        adjusted = [n + 1e-3 for n in self.counts]
        total = sum(adjusted)
        upper = [
            (s / n if n else 0) + exploration * math.sqrt(math.log(total) / adjusted[i])
            for i, (s, n) in enumerate(zip(self.totals, self.counts))
        ]
        return sorted(range(len(self.pool)), key=upper.__getitem__, reverse=True)[:size]

    async def _evaluate_round(self, chosen, samples_per_eval):
        for i in chosen:
            self.evaluations += 1
            score = float(await self.evaluate(self.pool[i]))
            if not math.isfinite(score):
                raise ValueError("evaluation score must be finite")
            self.totals[i] += score * samples_per_eval
            self.counts[i] += samples_per_eval

    async def run(
        self,
        demonstrations: Sequence[str],
        *,
        subsamples: int = 10,
        demos_per_sample: int = 5,
        prompts_per_sample: int = 5,
        rounds: int = 20,
        prompts_per_round: int = 10,
        samples_per_eval: int = 5,
        exploration: float = 1.0,
        seed: int = 0,
    ) -> dict:
        """Induce, deduplicate, then repeatedly sample, evaluate and update UCB."""
        self.rng = random.Random(seed)
        self.optimizer_calls = self.evaluations = 0
        self.pool = await self._induce_pool(
            demonstrations, subsamples, demos_per_sample, prompts_per_sample
        )
        self.totals = [0.0] * len(self.pool)
        self.counts = [0] * len(self.pool)
        history = []
        for _ in range(rounds):
            chosen = self._choose(min(prompts_per_round, len(self.pool)), exploration)
            await self._evaluate_round(chosen, samples_per_eval)
            history.append([self.pool[i] for i in chosen])
        population = [
            {"prompt": p, "score": s / n if n else None, "samples": n}
            for p, s, n in zip(self.pool, self.totals, self.counts)
        ]
        measured = [p for p in population if p["samples"]]
        return {
            "best": max(measured, key=lambda p: p["score"], default=None),
            "population": population,
            "history": history,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
        }
