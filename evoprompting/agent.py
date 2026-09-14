"""Evolve arbitrary candidate text with metric-conditioned crossover and prompt tuning."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Evaluation:
    """Measured error/cost; optional metrics and fitness adapt the search to other tasks."""

    error: float
    cost: float
    metrics: dict[str, float] = field(default_factory=dict)
    fitness: float | None = None


@dataclass(frozen=True)
class Individual:
    content: str
    evaluation: Evaluation
    fitness: float

    @property
    def metrics(self) -> dict[str, float]:
        return self.evaluation.metrics or {
            "num_params": self.evaluation.cost,
            "val_accuracy": 1 - self.evaluation.error,
        }


@dataclass(frozen=True)
class TuningSettings:
    epochs: int = 5
    prompt_length: int = 16
    batch_size: int = 16
    learning_rate: float = 0.1


ProviderFactory = Callable[[float], Provider]
Tune = Callable[
    [ProviderFactory, tuple[Individual, ...], TuningSettings], Awaitable[ProviderFactory]
]


class CandidateRejected(ValueError):
    """The evaluator rejected an artifact; infrastructure errors should propagate instead."""


@dataclass
class Attempt:
    round: int
    parents: tuple[Individual, ...]
    targets: dict[str, float]
    temperature: float
    raw: str = ""
    status: Literal["pending", "accepted", "duplicate", "filtered", "rejected", "failed"] = (
        "pending"
    )
    error: str = ""


@dataclass(frozen=True)
class Round:
    parents: tuple[Individual, ...]
    children: tuple[Individual, ...]
    selected: tuple[Individual, ...]
    training: tuple[Individual, ...]


@dataclass(frozen=True)
class Result:
    best: Individual | None
    top: tuple[Individual, ...]
    archive: tuple[Individual, ...]
    history: tuple[Round, ...]
    attempts: tuple[Attempt, ...]
    evaluations: int
    stop_reason: Literal["rounds", "no_parents"]
    provider: ProviderFactory


def paper_targets(parents: tuple[Individual, ...]) -> dict[str, float]:
    """Section 3.3 targets, including its unclamped accuracy and nearest-100 rounding."""
    return {
        "num_params": round(0.9 * min(p.evaluation.cost for p in parents) / 100) * 100,
        "val_accuracy": round(1.02 * max(1 - p.evaluation.error for p in parents), 3),
    }


class EvoPrompting:
    """Own Algorithms 1–3; the caller owns model training and isolated evaluation.

    `provider(temperature)` returns a stateless Slick provider bound to the current
    model. `tune(current_factory, children, settings)` updates only its soft prompt
    and returns the next factory. Pass `tune=None` explicitly for the no-tuning
    ablation. Search state resets on run; the current provider factory is retained.
    """

    def __init__(
        self,
        task: str,
        provider: ProviderFactory,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        *,
        tune: Tune | None,
        targets: Callable[[tuple[Individual, ...]], dict[str, float]] = paper_targets,
        rounds: int = 10,
        prompts_per_round: int = 10,
        samples_per_prompt: int = 16,
        examples_per_prompt: int = 2,
        survivors: int = 10,
        alpha: float = 0.5,
        temperatures: tuple[float, ...] = (0.2, 0.6, 0.8, 1.0),
        tuning: TuningSettings = TuningSettings(),
        seed: int = 0,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.tune, self.targets, self.tuning = tune, targets, tuning
        self.rounds, self.prompts_per_round = rounds, prompts_per_round
        self.samples_per_prompt, self.examples_per_prompt = samples_per_prompt, examples_per_prompt
        self.survivors, self.alpha, self.temperatures = survivors, alpha, temperatures
        self.seed = seed

    @prompt(template="crossmut.j2")
    async def crossmut(
        self, parents: tuple[Individual, ...], targets: dict[str, float], *, generated: str
    ) -> str:
        """The paper uses one LM operation for both crossover and mutation."""
        return generated

    async def _measure(self, content: str) -> Individual:
        self.evaluations += 1
        evaluation = await self.evaluate(content)
        score = evaluation.fitness
        if score is None:
            score = -evaluation.error * evaluation.cost
        if (
            not all(math.isfinite(x) for x in (evaluation.error, evaluation.cost, score))
            or evaluation.error < 0
            or evaluation.cost <= 0
            or not all(math.isfinite(x) for x in evaluation.metrics.values())
        ):
            raise CandidateRejected(
                "measured error, positive cost, fitness and metrics must be finite"
            )
        return Individual(content, evaluation, score)

    async def _initialize(self, seeds: Sequence[str]) -> tuple[Individual, ...]:
        parents = []
        for content in seeds:
            if content in self._seen:
                continue
            self._seen.add(content)
            # Seeds warm-start only round zero; they never enter the child archive.
            parents.append(await self._measure(content))
        return tuple(parents)

    async def _generate(self, parents: tuple[Individual, ...], round: int) -> list[Attempt]:
        attempts = []
        for _ in range(self.prompts_per_round):
            examples = tuple(self.rng.choices(parents, k=self.examples_per_prompt))
            targets = self.targets(examples)
            for _ in range(self.samples_per_prompt):
                attempt = Attempt(
                    round, examples, dict(targets), self.rng.choice(self.temperatures)
                )
                self.attempts.append(attempt)
                attempts.append(attempt)
                try:
                    attempt.raw = await self.crossmut(
                        examples, targets, provider=self.provider(attempt.temperature)
                    )
                except Exception as error:
                    # Record the failed request without turning transport bugs into rejections.
                    attempt.status, attempt.error = "failed", str(error)
                    raise
        return attempts

    async def _filter_and_evaluate(self, attempts: list[Attempt]) -> tuple[Individual, ...]:
        children = []
        for attempt in attempts:
            content = attempt.raw
            if content in self._seen:
                attempt.status = "duplicate"
                continue
            self._seen.add(content)
            try:
                if not content.strip():
                    raise CandidateRejected("blank candidate")
                child = await self._measure(content)
            except CandidateRejected as error:
                attempt.status, attempt.error = "rejected", str(error)
                continue
            if child.evaluation.error >= self.alpha:
                attempt.status = "filtered"
                continue
            attempt.status = "accepted"
            children.append(child)
        return tuple(children)

    def _select(self) -> tuple[Individual, ...]:
        selected = tuple(sorted(self.available, key=lambda c: -c.fitness)[: self.survivors])
        retired = {c.content for c in selected}
        self.available = [c for c in self.available if c.content not in retired]
        return selected

    async def run(self, seeds: Sequence[str]) -> Result:
        """Generate fixed batches, filter, retire parents, then tune on remaining children.

        Duplicate/invalid samples consume the generation budget without refill.
        Empty eligible pools stop the search; selected parents are never reused in
        later generations. Seed evaluation failures propagate. Higher fitness wins.
        """
        self.rng = random.Random(self.seed)
        self._seen: set[str] = set()
        self.evaluations = 0
        self.attempts: list[Attempt] = []
        self.archive: list[Individual] = []
        self.available: list[Individual] = []
        self.history: list[Round] = []
        parents = await self._initialize(seeds)
        reason = "rounds"
        for round in range(self.rounds):
            if not parents:
                reason = "no_parents"
                break
            attempts = await self._generate(parents, round)
            children = await self._filter_and_evaluate(attempts)
            self.archive.extend(children)
            self.available.extend(children)
            selected, training = (), ()
            if round < self.rounds - 1:
                selected = self._select()
                selected_contents = {c.content for c in selected}
                training = tuple(c for c in children if c.content not in selected_contents)
            self.history.append(Round(parents, children, selected, training))
            if selected and training and self.tune is not None:
                self.provider = await self.tune(self.provider, training, self.tuning)
            parents = selected
        return Result(
            best=max(self.archive, key=lambda c: c.fitness, default=None),
            top=tuple(sorted(self.available, key=lambda c: -c.fitness)[: self.survivors]),
            archive=tuple(self.archive),
            history=tuple(self.history),
            attempts=tuple(self.attempts),
            evaluations=self.evaluations,
            stop_reason=reason,
            provider=self.provider,
        )
