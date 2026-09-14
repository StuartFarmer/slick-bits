"""Optimize prompt templates with multi-aspect critiques and optional suffix tuning."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean

from slick import prompt
from slick.providers import Provider

PLACEHOLDER = re.compile(r"INSERT_[A-Z0-9_]+_HERE")


@dataclass(frozen=True)
class Example:
    inputs: Mapping[str, str]
    reference: str


@dataclass
class Candidate:
    text: str
    prompt: str
    train_scores: dict[str, float]
    dev_scores: dict[str, float]
    critique: str = ""


def fill_prompt(template: str, inputs: Mapping[str, str]) -> str:
    """Substitute INSERT_*_HERE tokens once; inserted data is never interpreted."""
    return PLACEHOLDER.sub(lambda match: inputs[match.group()], template)


def average_ranks(scores: Sequence[Mapping[str, float]], metrics: Sequence[str]) -> list[float]:
    """Official AST convention: mean zero-based competition rank; lower is better."""
    ordered = {name: sorted((score[name] for score in scores), reverse=True) for name in metrics}
    # ponytail: O(metrics * candidates²); use rank maps if archives grow beyond small searches.
    return [mean(ordered[name].index(score[name]) for name in metrics) for score in scores]


class CriSPO:
    """Caller-owned task inference and metric evaluation; all metric values maximize.

    `respond` receives only the filled prompt. `evaluate` receives a split's examples
    and predictions and returns corpus-level metrics. Calls are sequential and have
    no hidden retries or conversational Session. Errors propagate; counters and raw
    meta-model responses remain on the instance for inspection.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        respond: Callable[[str], Awaitable[str]],
        evaluate: Callable[[Sequence[Example], Sequence[str]], Awaitable[Mapping[str, float]]],
        *,
        primary_metric: str,
        placeholders: Sequence[str] = ("INSERT_INPUT_HERE",),
        critique_provider: Provider | None = None,
    ):
        self.task = task
        self.provider = provider
        self.critique_provider = provider if critique_provider is None else critique_provider
        self.respond = respond
        self.evaluate = evaluate
        self.primary_metric = primary_metric
        self.placeholders = tuple(placeholders)
        self.generations: list[dict[str, str]] = []
        self.history: list[Candidate] = []
        self.suffix_history: list[Candidate] = []
        self.optimizer_calls = self.critique_calls = self.response_calls = 0
        self.evaluations = self.duplicates = 0

    def _extract(self, response: str, tag: str) -> str:
        self.generations.append({"operation": tag, "response": response})
        matches = re.findall(rf"<{tag}>(.*?)</{tag}>", response, flags=re.DOTALL)
        if (
            len(matches) != 1
            or response.count(f"<{tag}>") != 1
            or response.count(f"</{tag}>") != 1
            or not matches[0].strip()
        ):
            raise ValueError(f"expected one nonempty <{tag}>...</{tag}>")
        return matches[0].strip()

    @prompt(template="critique.j2")
    async def critique(self, instruction: str, observations: list[dict], *, generated: str) -> str:
        """Discover aspects and return textual critiques with actionable edits."""
        return self._extract(generated, "critique")

    @prompt(template="revise.j2")
    async def revise(self, trajectory: list[dict], *, generated: str) -> str:
        """Extract a revised template and enforce its data placeholders."""
        instruction = self._extract(generated, "instruction")
        if set(PLACEHOLDER.findall(instruction)) != set(self.placeholders) or any(
            instruction.count(token) != 1 for token in self.placeholders
        ):
            raise ValueError("generated instruction must contain each declared placeholder once")
        return instruction

    @prompt(template="critique_suffix.j2")
    async def critique_suffix(
        self,
        main_prompt: str,
        suffix: str,
        observations: list[dict],
        metrics: Sequence[str],
        *,
        generated: str,
    ) -> str:
        """Critique only the postscript using the existing and additional metrics."""
        return self._extract(generated, "critique")

    @prompt(template="revise_suffix.j2")
    async def revise_suffix(
        self, main_prompt: str, trajectory: list[dict], metrics: Sequence[str], *, generated: str
    ) -> str:
        """Extract only a postscript; the main template is composed in Python."""
        suffix = self._extract(generated, "postscript")
        if PLACEHOLDER.search(suffix):
            raise ValueError("generated postscript cannot introduce data placeholders")
        return suffix

    async def _measure(self, template: str, examples: Sequence[Example]):
        predictions = []
        for example in examples:
            filled = fill_prompt(template, example.inputs)
            self.response_calls += 1
            predictions.append(await self.respond(filled))
        self.evaluations += 1
        scores = {
            name: float(value)
            for name, value in (await self.evaluate(examples, predictions)).items()
        }
        if not all(math.isfinite(score) for score in scores.values()):
            raise ValueError("measured scores must be finite")
        scores[self.primary_metric]
        return scores, predictions

    async def _assess(self, text, train, dev, archive, *, main_prompt=None, metrics=()):
        if any(candidate.text == text for candidate in archive):
            self.duplicates += 1
            return
        template = text if main_prompt is None else main_prompt + ("\n\n" + text if text else "")
        train_scores, predictions = await self._measure(template, train)
        dev_scores, _ = await self._measure(template, dev)
        candidate = Candidate(text, template, train_scores, dev_scores)
        archive.append(candidate)
        indices = self.rng.sample(range(len(train)), min(self.critique_size, len(train)))
        observations = [
            {
                "inputs": dict(train[i].inputs),
                "reference": train[i].reference,
                "prediction": predictions[i],
            }
            for i in indices
        ]
        self.critique_calls += 1
        if main_prompt is None:
            candidate.critique = await self.critique(
                text, observations, provider=self.critique_provider
            )
        else:
            candidate.critique = await self.critique_suffix(
                main_prompt, text, observations, metrics, provider=self.critique_provider
            )

    def _utilities(self, archive: Sequence[Candidate], split: str, metrics: Sequence[str]):
        scores = [getattr(candidate, f"{split}_scores") for candidate in archive]
        if metrics:
            return [-rank for rank in average_ranks(scores, metrics)]
        return [score[self.primary_metric] for score in scores]

    def _trajectory(self, archive, history_size, metrics=()):
        utilities = self._utilities(archive, "train", metrics)
        # Select best first for stable ties, then present weakest to strongest.
        selected = sorted(range(len(archive)), key=lambda i: -utilities[i])[:history_size]
        return [
            {
                "text": archive[i].text,
                "score": utilities[i],
                "metrics": archive[i].train_scores,
                "critique": archive[i].critique,
            }
            for i in sorted(selected, key=lambda i: utilities[i])
        ]

    def _best(self, archive, metrics=()):
        utilities = self._utilities(archive, "dev", metrics)
        return archive[max(range(len(archive)), key=lambda i: utilities[i])]

    async def _tune_suffix(
        self, main_prompt, initial_suffix, train, dev, iterations, history_size, metrics
    ):
        await self._assess(
            initial_suffix,
            train,
            dev,
            self.suffix_history,
            main_prompt=main_prompt,
            metrics=metrics,
        )
        for _ in range(iterations):
            trajectory = self._trajectory(self.suffix_history, history_size, metrics)
            self.optimizer_calls += 1
            suffix = await self.revise_suffix(
                main_prompt, trajectory, metrics, provider=self.provider
            )
            await self._assess(
                suffix, train, dev, self.suffix_history, main_prompt=main_prompt, metrics=metrics
            )
        return self._best(self.suffix_history, metrics)

    async def run(
        self,
        initial_prompt: str,
        train: Sequence[Example],
        dev: Sequence[Example],
        *,
        iterations: int = 20,
        history_size: int = 5,
        critique_size: int = 10,
        seed: int = 0,
        initial_suffix: str | None = None,
        suffix_iterations: int = 20,
        suffix_metrics: Sequence[str] = (),
    ) -> dict:
        """Evaluate the seed and one proposal per iteration; optionally run AST.

        Train scores guide revision, dev scores choose the result. AST optimizes
        the primary metric plus `suffix_metrics` by average rank. Duplicate proposals
        consume their iteration but skip inference, evaluation and critique.
        """
        self.history, self.suffix_history, self.generations = [], [], []
        self.optimizer_calls = self.critique_calls = self.response_calls = 0
        self.evaluations = self.duplicates = 0
        self.rng, self.critique_size = random.Random(seed), critique_size
        await self._assess(initial_prompt, train, dev, self.history)
        for _ in range(iterations):
            trajectory = self._trajectory(self.history, history_size)
            self.optimizer_calls += 1
            instruction = await self.revise(trajectory, provider=self.provider)
            await self._assess(instruction, train, dev, self.history)
        main_best = best = self._best(self.history)
        if initial_suffix is not None:
            metrics = tuple(dict.fromkeys((self.primary_metric, *suffix_metrics)))
            best = await self._tune_suffix(
                main_best.prompt,
                initial_suffix,
                train,
                dev,
                suffix_iterations,
                history_size,
                metrics,
            )
        return {
            "best": best,
            "main_best": main_best,
            "history": self.history,
            "suffix_history": self.suffix_history,
            "generations": self.generations,
            "optimizer_calls": self.optimizer_calls,
            "critique_calls": self.critique_calls,
            "response_calls": self.response_calls,
            "evaluations": self.evaluations,
            "duplicates": self.duplicates,
        }
