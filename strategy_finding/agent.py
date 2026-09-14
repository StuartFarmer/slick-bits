"""Discover categorized signals, assess confidence and risk, and fit a combiner."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

from .combiner import Combiner, TrainingData, fit_combiner

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Candidate(BaseModel, extra="forbid", frozen=True):
    category: Text
    name: Text
    content: str = Field(min_length=1)


class Filtered(BaseModel, extra="forbid"):
    content: str


class Categories(BaseModel, extra="forbid"):
    categories: list[Text]


class Proposal(BaseModel, extra="forbid"):
    name: Text
    content: str = Field(min_length=1)


class Proposals(BaseModel, extra="forbid"):
    candidates: list[Proposal]


class Assessment(BaseModel, extra="forbid"):
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: Text


@dataclass(frozen=True)
class Evaluation:
    """Caller-measured higher-is-better score and historical evidence for both judges."""

    score: float
    evidence: str


class CandidateRejected(ValueError):
    """The evaluator cannot evaluate this candidate; record it and continue."""


@dataclass(frozen=True)
class Rejection:
    candidate: Candidate
    reason: str


@dataclass(frozen=True)
class JudgedCandidate:
    candidate: Candidate
    evaluation: Evaluation
    confidence: Assessment
    risk: Assessment
    score: float


@dataclass
class Generation:
    prompt: str
    response: str | None = None
    error: str | None = None


class _RecordingProvider:
    """Keep raw responses even when Slick parsing or postprocessing rejects them."""

    def __init__(self, provider: Provider, records: list[Generation]):
        self.provider, self.records = provider, records

    async def acall(self, context, *, tools=None, tool_results=None):
        record = Generation(context)
        self.records.append(record)
        try:
            response, requests = await self.provider.acall(
                context, tools=tools, tool_results=tool_results
            )
        except Exception as exc:
            record.error = repr(exc)
            raise
        record.response = response
        return response, requests


@dataclass(frozen=True)
class Result:
    factory: tuple[Candidate, ...]
    selected: tuple[Candidate, ...]
    assessments: tuple[JudgedCandidate, ...]
    rejections: tuple[Rejection, ...]
    model: Combiner | None
    evaluations: int
    generation_calls: int
    generations: tuple[Generation, ...]


class StrategyFinder:
    """Run paper A.6 with task-defined candidate execution and aligned numeric data.

    Calls are sequential and stateless; exceptions propagate without retries except
    CandidateRejected from evaluation. Reuse the returned factory for new documents
    or conditions. Each run measures all candidates again and fits a fresh model.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Candidate, str], Awaitable[Evaluation]],
        prepare: Callable[[tuple[Candidate, ...]], Awaitable[TrainingData]],
        *,
        candidate_interface: str = "Return one numeric signal per observation.",
        risk_preference: str = "Prefer robust performance and limited downside.",
    ):
        self.task, self.provider = task, provider
        self.evaluate, self.prepare = evaluate, prepare
        self.candidate_interface, self.risk_preference = candidate_interface, risk_preference

    @prompt(template="filter.j2", output_type=Filtered)
    async def filter_document(self, document: str, *, generated: Filtered) -> str:
        return generated.content.strip()

    @prompt(template="categorize.j2", output_type=Categories)
    async def categorize(self, content: str, *, generated: Categories) -> list[str]:
        if len(generated.categories) != len(set(generated.categories)):
            raise ValueError("generated categories must be distinct")
        return generated.categories

    @prompt(template="generate.j2", output_type=Proposals)
    async def generate(
        self, content: str, category: str, existing: list[dict], *, generated: Proposals
    ) -> list[Candidate]:
        if any(not item.content.strip() for item in generated.candidates):
            raise ValueError("generated candidate content must not be blank")
        return [Candidate(category=category, **item.model_dump()) for item in generated.candidates]

    @prompt(template="confidence.j2", output_type=Assessment)
    async def confidence(
        self, candidate: Candidate, evaluation: Evaluation, context: str, *, generated: Assessment
    ) -> Assessment:
        return generated

    @prompt(template="risk.j2", output_type=Assessment)
    async def risk(
        self, candidate: Candidate, evaluation: Evaluation, context: str, *, generated: Assessment
    ) -> Assessment:
        return generated

    async def build_factory(self, documents: Sequence[str], factory: Sequence[Candidate]) -> None:
        # Exact content deduplication preserves executable whitespace and insertion order.
        seeds = {(item.category, item.content): item for item in factory}
        self.factory = list(seeds.values())
        for document in documents:
            content = await self.filter_document(document, provider=self._recording)
            if not content:
                continue
            categories = await self.categorize(content, provider=self._recording)
            for category in categories:
                existing = [item.model_dump() for item in seeds.values()]
                candidates = await self.generate(
                    content, category, existing, provider=self._recording
                )
                for candidate in candidates:
                    seeds.setdefault((candidate.category, candidate.content), candidate)
                self.factory = list(seeds.values())

    async def select(
        self, context: str, threshold: float, confidence_weight: float, risk_weight: float
    ) -> tuple[Candidate, ...]:
        winners: dict[str, JudgedCandidate] = {}
        for candidate in self.factory:
            self.evaluations += 1
            try:
                evaluation = await self.evaluate(candidate, context)
            except CandidateRejected as exc:
                self.rejections.append(Rejection(candidate, str(exc)))
                continue
            if not math.isfinite(evaluation.score):
                raise ValueError("measured score must be finite")
            confidence = await self.confidence(
                candidate, evaluation, context, provider=self._recording
            )
            risk = await self.risk(candidate, evaluation, context, provider=self._recording)
            score = confidence_weight * confidence.score + risk_weight * risk.score
            judged = JudgedCandidate(candidate, evaluation, confidence, risk, score)
            self.assessments.append(judged)
            if score > threshold:
                previous = winners.get(candidate.category)
                if previous is None or score > previous.score:
                    winners[candidate.category] = judged
        return tuple(item.candidate for item in winners.values())

    async def run(
        self,
        documents: Sequence[str],
        *,
        context: str = "",
        factory: Sequence[Candidate] = (),
        threshold: float = 0.5,
        confidence_weight: float = 0.6,
        risk_weight: float = 0.4,
        epochs: int = 1000,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        regularization: float = 0.001,
        seed: int = 0,
    ) -> Result:
        self.generations: list[Generation] = []
        self.assessments: list[JudgedCandidate] = []
        self.rejections: list[Rejection] = []
        self.evaluations = 0
        self._recording = _RecordingProvider(self.provider, self.generations)
        await self.build_factory(documents, factory)
        selected = await self.select(context, threshold, confidence_weight, risk_weight)
        model = None
        if selected:
            data = await self.prepare(selected)
            model = fit_combiner(
                data,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                regularization=regularization,
                seed=seed,
            )
        return Result(
            tuple(self.factory),
            selected,
            tuple(self.assessments),
            tuple(self.rejections),
            model,
            self.evaluations,
            len(self.generations),
            tuple(self.generations),
        )
