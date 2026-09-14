"""Optimize text-encoded solutions using OPRO's measured search trajectory."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence

from slick import prompt
from slick.providers import Provider


class OPRO:
    """Keep evaluated solutions and present the strongest from worst to best.

    Higher finite scores are better unless ``maximize=False``. Exact-text deduplication assumes
    repeatable evaluation. Provider, parser and evaluator failures propagate.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        *,
        maximize: bool = True,
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.maximize = maximize
        self.score_direction = "higher" if maximize else "lower"
        self.raw_responses: list[str] = []

    @prompt(template="propose.j2")
    async def propose(
        self, trajectory: list[dict], exemplars: Sequence[str], *, generated: str
    ) -> str:
        """Read the score trajectory and return one checked text-encoded solution."""
        self.raw_responses.append(generated)
        matches = re.findall(r"<TEXT>(.*?)</TEXT>", generated, flags=re.DOTALL)
        if (
            len(matches) != 1
            or generated.count("<TEXT>") != 1
            or generated.count("</TEXT>") != 1
            or not matches[0].strip()
        ):
            raise ValueError(f"expected one nonempty <TEXT>...</TEXT>; response={generated!r}")
        return matches[0].strip()

    async def _score_new(self, proposals: Sequence[str], iteration: int) -> None:
        for candidate in proposals:
            if candidate in self.seen:
                self.duplicates += 1
                continue
            self.evaluations += 1
            score = float(await self.evaluate(candidate))
            if not math.isfinite(score):
                raise ValueError("evaluation score must be finite")
            self.seen.add(candidate)
            self.archive.append({"prompt": candidate, "score": score, "iteration": iteration})

    async def _propose_batch(
        self, history_size: int, proposals_per_step: int, exemplars: Sequence[str]
    ) -> list[str]:
        trajectory = sorted(
            self.archive, key=lambda p: p["score"], reverse=not self.maximize
        )[-history_size:]
        proposals = []
        for _ in range(proposals_per_step):
            self.optimizer_calls += 1
            proposals.append(await self.propose(trajectory, exemplars, provider=self.provider))
        return proposals

    async def run(
        self,
        initial_prompts: Sequence[str],
        *,
        iterations: int = 20,
        proposals_per_step: int = 8,
        history_size: int = 20,
        exemplars: Sequence[str] = (),
        exemplars_per_step: int = 3,
        seed: int | None = None,
        patience: int | None = None,
    ) -> dict:
        """Evaluate seeds, generate frozen batches and retain every unique measurement.

        ``initial_prompts`` and result ``prompt`` fields hold arbitrary solution text.
        Supply at least one seed (the empty string is allowed). A fresh run resets state.
        Sample up to ``exemplars_per_step`` examples without replacement per step.
        ``patience`` optionally stops after that many steps without strict improvement.
        """
        self.seen = set()
        self.archive = []
        self.raw_responses = []
        self.optimizer_calls = self.evaluations = self.duplicates = 0
        await self._score_new(initial_prompts, -1)
        select_best = max if self.maximize else min
        best = select_best(self.archive, key=lambda p: p["score"])
        rng = random.Random(seed)
        history = []
        stalled = 0
        stop_reason = "iterations"
        for iteration in range(iterations):
            examples = rng.sample(list(exemplars), min(exemplars_per_step, len(exemplars)))
            proposals = await self._propose_batch(history_size, proposals_per_step, examples)
            await self._score_new(proposals, iteration)
            previous = best
            best = select_best(self.archive, key=lambda p: p["score"])
            stalled = stalled + 1 if best["score"] == previous["score"] else 0
            history.append(best.copy())
            if patience is not None and stalled >= patience:
                stop_reason = "patience"
                break
        return {
            "best": best,
            "archive": self.archive,
            "history": history,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "duplicates": self.duplicates,
            "raw_responses": self.raw_responses,
            "stop_reason": stop_reason,
        }
