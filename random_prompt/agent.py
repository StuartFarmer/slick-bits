"""Search random vocabulary or unconditional language-model separators."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Candidate:
    separator: str
    context: str
    score: float


class RandomPrompt:
    """Keep the context fixed and search its literal {separator} placeholders.

    evaluate receives the substituted context and returns a higher-is-better
    score. configure_sampling binds an unconditional provider's token budget.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        *,
        decode: Callable[[Sequence[int]], str] | None = None,
        configure_sampling: Callable[[Provider, int], Provider] | None = None,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.decode, self.configure_sampling = decode, configure_sampling

    @prompt(template="unconditional.j2")
    async def sample(self, *, generated: str) -> str:
        """Preserve separator whitespace and the source's empty model context."""
        return generated

    async def run(
        self,
        context: str,
        *,
        mode: Literal["vocabulary", "unconditional"] = "vocabulary",
        vocabulary_size: int = 50257,
        draws: int = 20,
        min_length: int = 1,
        max_length: int = 5,
        keep: int = 4,
        seed: int = 0,
    ) -> dict:
        """Generate the fixed candidate batch, score every draw, then rank stably."""
        rng = random.Random(seed)
        self.generations = self.evaluations = 0
        separators = []
        for _ in range(draws):
            length = rng.randint(min_length, max_length)
            if mode == "vocabulary":
                separators.append(self.decode(rng.sample(range(vocabulary_size), length)))
            else:
                self.generations += 1
                separators.append(
                    await self.sample(provider=self.configure_sampling(self.provider, length))
                )
        candidates = []
        for separator in separators:
            rendered = context.replace("{separator}", separator)
            self.evaluations += 1
            score = float(await self.evaluate(rendered))
            if not math.isfinite(score):
                raise ValueError("separator score must be finite")
            candidates.append(Candidate(separator, rendered, score))
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        return {
            "best": ranked[0],
            "selected": ranked[:keep],
            "candidates": candidates,
            "generations": self.generations,
            "evaluations": self.evaluations,
        }
