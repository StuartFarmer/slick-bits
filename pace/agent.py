"""Refine prompts using sampled actor responses and per-response critic advice."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from slick import prompt
from slick.providers import Provider


@dataclass
class Example:
    input: str
    target: str


class PACE:
    """Reconstruct the paper's actor/critic/update loop; no official code was located.

    Critiques use demonstrations, and separate evaluation selects the best-ever
    prompt. Iteration continues from its selected candidate even when it loses
    to the historical best. Responses are text, never executed as code.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        actor_provider: Provider | None = None,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.actor_provider = provider if actor_provider is None else actor_provider

    @prompt(template="act.j2")
    async def act(self, instruction: str, input: str, *, generated: str) -> str:
        return generated.strip()

    @prompt(template="criticize.j2")
    async def criticize(
        self, instruction: str, input: str, prediction: str, target: str, *, generated: str
    ) -> str:
        if not generated.strip():
            raise ValueError(f"empty critique; response={generated!r}")
        return generated.strip()

    @prompt(template="update.j2")
    async def update(self, instruction: str, critiques: list[str], *, generated: str) -> str:
        if not generated.strip():
            raise ValueError(f"empty instruction; response={generated!r}")
        return generated.strip()

    async def _score(self, instruction):
        self.evaluations += 1
        score = float(await self.evaluate(instruction))
        if not math.isfinite(score):
            raise ValueError("fitness must be finite")
        return {"prompt": instruction, "score": score}

    async def _candidate(self, current, examples, actors):
        critiques, observations = [], []
        for _ in range(actors):
            example = self.rng.choice(examples)
            self.actor_calls += 1
            response = await self.act(current, example.input, provider=self.actor_provider)
            self.optimizer_calls += 1
            critique = await self.criticize(
                current, example.input, response, example.target, provider=self.provider
            )
            critiques.append(critique)
            observations.append(
                {
                    "input": example.input,
                    "target": example.target,
                    "response": response,
                    "critique": critique,
                }
            )
        self.optimizer_calls += 1
        instruction = await self.update(current, critiques, provider=self.provider)
        self.history.append(
            {"parent": current, "prompt": instruction, "observations": observations}
        )
        return await self._score(instruction)

    async def run(
        self,
        initial_prompt: str,
        examples: Sequence[Example],
        *,
        iterations: int = 1,
        candidates: int = 2,
        actors: int = 4,
        seed: int = 0,
    ) -> dict:
        self.rng, self.history = random.Random(seed), []
        self.evaluations = self.actor_calls = self.optimizer_calls = 0
        current = best = await self._score(initial_prompt)
        for _ in range(iterations):
            trials = [
                await self._candidate(current["prompt"], examples, actors)
                for _ in range(candidates)
            ]
            selected = max(trials, key=lambda item: item["score"])
            if selected["score"] > best["score"]:
                best = selected
            unchanged = selected["prompt"] == current["prompt"]
            current = selected
            if unchanged:
                break
        return {
            "best": best,
            "current": current,
            "history": self.history.copy(),
            "evaluations": self.evaluations,
            "actor_calls": self.actor_calls,
            "optimizer_calls": self.optimizer_calls,
        }
