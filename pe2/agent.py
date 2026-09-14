"""Engineer prompts with explicit task context and two-stage example reasoning."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from slick import prompt
from slick.providers import Provider


def checked(text):
    if not text.strip():
        raise ValueError(f"empty generated text; response={text!r}")
    return text.strip()


@dataclass
class Candidate:
    prompt: str
    score: float
    history: list[str] = field(default_factory=list)


class PE2:
    """Select prompts on validation scores and inspect training observations.

    Observations contain input, output, label, optional reasoning, and score;
    score == 0 selects hard examples. The supplied full_template uses literal
    {instruction} and {input} markers to show the actual task-model context.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        observe: Callable[[str], Awaitable[Sequence[dict]]],
        full_template: str = "{instruction}\n{input}",
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.observe, self.full_template = observe, full_template

    @prompt(template="induce.j2")
    async def induce(self, examples: list[dict], *, generated: str) -> str:
        return checked(generated)

    @prompt(template="analyze.j2")
    async def analyze(
        self, instruction: str, context: str, examples: list[dict], *, generated: str
    ) -> str:
        return checked(generated)

    @prompt(template="revise.j2")
    async def revise(
        self, instruction: str, analysis: str, history: list[str], *, generated: str
    ) -> str:
        return checked(generated)

    @prompt(template="summarize_edit.j2")
    async def summarize_edit(
        self, before: str, after: str, analysis: str, *, generated: str
    ) -> str:
        return checked(generated)

    async def _score(self, text, history):
        self.evaluations += 1
        score = float(await self.evaluate(text))
        if not math.isfinite(score):
            raise ValueError("fitness must be finite")
        return Candidate(text, score, history)

    async def _expand(self, parent, children, batch_size, hard_examples, momentum):
        self.observation_calls += 1
        observations = list(await self.observe(parent.prompt))
        pool = [row for row in observations if row["score"] == 0] if hard_examples else observations
        offspring = []
        for _ in range(children):
            batch = self.rng.sample(pool, min(batch_size, len(pool)))
            if not batch:
                break
            self.optimizer_calls += 1
            analysis = await self.analyze(
                parent.prompt,
                self.full_template.replace("{instruction}", parent.prompt),
                batch,
                provider=self.provider,
            )
            self.optimizer_calls += 1
            text = await self.revise(
                parent.prompt, analysis, parent.history, provider=self.provider
            )
            record = {
                "parent": parent.prompt,
                "analysis": analysis,
                "prompt": text,
                "duplicate": text in self.seen,
            }
            self.history.append(record)
            if text in self.seen:
                continue
            history = parent.history.copy()
            if momentum:
                self.optimizer_calls += 1
                summary = await self.summarize_edit(
                    parent.prompt, text, analysis, provider=self.provider
                )
                history.extend([f"Validation score before edit: {parent.score}", summary])
            self.seen.add(text)
            offspring.append(await self._score(text, history))
        return offspring

    async def run(
        self,
        initial_prompts: Sequence[str],
        *,
        iterations: int = 5,
        beam_size: int = 4,
        children: int = 2,
        batch_size: int = 2,
        backtrack: bool = True,
        hard_examples: bool = True,
        momentum: bool = False,
        demonstrations: Sequence[dict] = (),
        initial_count: int = 1,
        seed: int = 0,
    ) -> dict:
        self.rng, self.history = random.Random(seed), []
        self.evaluations = self.optimizer_calls = self.observation_calls = 0
        initial = list(initial_prompts)
        if not initial:
            for _ in range(initial_count):
                self.optimizer_calls += 1
                initial.append(await self.induce(list(demonstrations), provider=self.provider))
        self.seen = set(initial)
        population = [await self._score(text, []) for text in dict.fromkeys(initial)]
        archive = population.copy()
        for _ in range(iterations):
            parents = sorted(
                archive if backtrack else population,
                key=lambda candidate: candidate.score,
                reverse=True,
            )[:beam_size]
            offspring = []
            for parent in parents:
                offspring.extend(
                    await self._expand(parent, children, batch_size, hard_examples, momentum)
                )
            if not offspring:
                break
            population = offspring
            archive.extend(offspring)
        return {
            "best": max(archive, key=lambda candidate: candidate.score),
            "population": population,
            "archive": archive,
            "history": self.history.copy(),
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "observation_calls": self.observation_calls,
        }
