"""Compile ideas into candidates, enhance them genetically, and review measured results."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Stage = Literal["successes", "categories", "subcategories", "fields", "literature"]


class Candidate(BaseModel, extra="forbid", frozen=True):
    content: str
    explanation: Text


class Idea(BaseModel, extra="forbid", frozen=True):
    idea: Text
    directions: Text
    crossover_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    mutation_rate: float = Field(ge=0, le=1, allow_inf_nan=False)


class Seeds(BaseModel, extra="forbid"):
    candidates: list[Candidate] = Field(min_length=1)


class Choice(BaseModel, extra="forbid"):
    id: Text


class Analysis(BaseModel, extra="forbid", frozen=True):
    summary: Text
    next_direction: Text


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Query:
    stage: Stage
    text: str
    path: tuple[str, ...] = ()
    limit: int = 8


@dataclass(frozen=True)
class Evaluation:
    score: float
    feedback: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    qualified: bool = True


@dataclass(frozen=True)
class Individual:
    content: str
    explanation: str
    evaluation: Evaluation


@dataclass(frozen=True)
class Round:
    idea: Idea
    seeds: tuple[Individual, ...]
    population: tuple[Individual, ...]
    analysis: Analysis


@dataclass(frozen=True)
class Failure:
    stage: str
    content: str
    error: str


@dataclass(frozen=True)
class Result:
    best: Individual | None
    population: tuple[Individual, ...]
    rounds: tuple[Round, ...]
    calls: int
    evaluations: int
    failures: tuple[Failure, ...]
    stop_reason: Literal["rounds", "feedback", "budget"]


class CandidateRejected(ValueError):
    """The evaluator or variation operator rejected this candidate's domain validity."""


class BudgetExceeded(Exception):
    """A caller-owned model or evaluation cap has been reached."""


Mutate = Callable[[str, str, random.Random], Awaitable[str]]
Crossover = Callable[[str, str, str, random.Random], Awaitable[str]]
Retrieve = Callable[[Query], Awaitable[Sequence[Document]]]


class AlphaGPT:
    """Own one task's interactive or autonomous discovery run.

    Callbacks supply the domain grammar and evaluation; this class owns tournaments,
    variation scheduling, elitist selection, budgets, and explicit review history.
    Use a fresh instance per run. Scores must be stable throughout that run.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        *,
        mutate: Mutate,
        crossover: Crossover,
        retrieve: Retrieve | None = None,
        redundant: Callable[[Individual, Individual], bool] | None = None,
        seed_count: int = 10,
        population_size: int = 10,
        generations: int = 10,
        maximize: bool = True,
        seed: int = 0,
        max_calls: int = 100,
        max_evaluations: int = 1000,
        repair_attempts: int = 1,
        retrieval_limit: int = 8,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.mutate, self.crossover = mutate, crossover
        self.retrieve, self.redundant = retrieve, redundant
        self.seed_count, self.population_size = seed_count, population_size
        self.search_generations, self.maximize = generations, maximize
        self.max_calls, self.max_evaluations = max_calls, max_evaluations
        self.repair_attempts, self.retrieval_limit = repair_attempts, retrieval_limit
        self.rng = random.Random(seed)
        self.calls: list[dict] = []
        self.evaluations = 0
        self.failures: list[Failure] = []
        self.population: list[Individual] = []
        self.rounds: list[Round] = []
        self.generations: list[tuple[Individual, ...]] = []
        self.archive: dict[str, Individual | None] = {}
        self.rejections: dict[str, str] = {}
        self.retrievals: list[tuple[Query, tuple[Document, ...]]] = []

    @property
    def score_direction(self) -> str:
        return "Higher" if self.maximize else "Lower"

    @prompt(template="polish.j2", output_type=Idea)
    async def polish(
        self, idea: str, feedback: str, knowledge: Sequence[Document], *, generated: Idea
    ) -> Idea:
        return generated

    @prompt(template="initialize.j2", output_type=Seeds)
    async def initialize(
        self, idea: Idea, knowledge: Sequence[Document], *, generated: Seeds
    ) -> list[Candidate]:
        if len(generated.candidates) != self.seed_count:
            raise ValueError("generated seed count does not match seed_count")
        return generated.candidates

    @prompt(template="repair.j2", output_type=Candidate)
    async def repair(
        self, candidate: Candidate, error: str, idea: Idea, *, generated: Candidate
    ) -> Candidate:
        return generated

    @prompt(template="select_category.j2", output_type=Choice)
    async def select_category(
        self, options: Sequence[Document], memory: Sequence[Document], *, generated: Choice
    ) -> str:
        if generated.id not in {document.id for document in options}:
            raise ValueError("generated unknown category ID")
        return generated.id

    @prompt(template="select_subcategory.j2", output_type=Choice)
    async def select_subcategory(
        self,
        category: str,
        options: Sequence[Document],
        memory: Sequence[Document],
        *,
        generated: Choice,
    ) -> str:
        if generated.id not in {document.id for document in options}:
            raise ValueError("generated unknown subcategory ID")
        return generated.id

    @prompt(template="discover.j2", output_type=Candidate)
    async def discover(
        self,
        knowledge: Sequence[Document],
        memory: Sequence[Document],
        feedback: str,
        *,
        generated: Candidate,
    ) -> str:
        if not generated.content.strip():
            raise ValueError("generated idea is blank")
        return generated.content

    @prompt(template="review.j2", output_type=Analysis)
    async def review(
        self,
        idea: Idea,
        seeds: Sequence[Individual],
        population: Sequence[Individual],
        failures: Sequence[Failure],
        *,
        generated: Analysis,
    ) -> Analysis:
        return generated

    async def _retrieve(self, stage: Stage, text: str, path: tuple[str, ...] = ()):
        query = Query(stage, text, path, self.retrieval_limit)
        documents = () if self.retrieve is None else tuple(await self.retrieve(query))
        documents = documents[: self.retrieval_limit]
        self.retrievals.append((query, documents))
        return documents

    async def _discover_idea(self, feedback):
        if self.retrieve is None:
            raise ValueError("autonomous discovery requires a hierarchical retrieve callback")
        memory = await self._retrieve("successes", self.task)
        categories = await self._retrieve("categories", self.task + "\n" + feedback)
        category = await self._invoke(self.select_category, categories, memory)
        options = await self._retrieve("subcategories", self.task, (category,))
        subcategory = await self._invoke(self.select_subcategory, category, options, memory)
        fields = await self._retrieve("fields", self.task, (category, subcategory))
        idea = await self._invoke(self.discover, fields, memory, feedback)
        return idea, fields + memory

    async def _ideate(self, idea, feedback):
        knowledge = ()
        if idea is None:
            idea, knowledge = await self._discover_idea(feedback)
        else:
            knowledge = await self._retrieve("successes", idea)
            knowledge += await self._retrieve("fields", idea)
        knowledge += await self._retrieve("literature", idea)
        polished = await self._invoke(self.polish, idea, feedback, knowledge)
        return polished, knowledge

    def _rank(self, individual):
        return -individual.evaluation.score if self.maximize else individual.evaluation.score

    def _select(self, candidates):
        unique = {candidate.content: candidate for candidate in candidates}
        selected = []
        for candidate in sorted(unique.values(), key=self._rank):
            if self.redundant is not None and any(
                self.redundant(candidate, other) for other in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= self.population_size:
                break
        self.population = selected

    async def _assess(self, candidate: Candidate, stage: str) -> Individual | None:
        content = candidate.content
        if content in self.archive:
            return self.archive[content]
        if self.evaluations >= self.max_evaluations:
            raise BudgetExceeded
        try:
            if not content.strip():
                raise CandidateRejected("candidate is blank")
            self.evaluations += 1
            evaluation = await self.evaluate(content)
            if not math.isfinite(evaluation.score) or any(
                not math.isfinite(value) for value in evaluation.metrics.values()
            ):
                raise CandidateRejected("evaluation returned a nonfinite measurement")
            if not evaluation.qualified:
                raise CandidateRejected(evaluation.feedback or "candidate did not qualify")
        except CandidateRejected as error:
            self.failures.append(Failure(stage, content, str(error)))
            self.rejections[content] = str(error)
            self.archive[content] = None
            return None
        individual = Individual(content, candidate.explanation, evaluation)
        self.archive[content] = individual
        return individual

    async def _seed(self, idea, knowledge):
        candidates = await self._invoke(self.initialize, idea, knowledge)
        seeds = []
        for candidate in candidates:
            individual = await self._assess(candidate, "seed")
            for _ in range(self.repair_attempts):
                if individual is not None:
                    break
                candidate = await self._invoke(
                    self.repair, candidate, self.rejections[candidate.content], idea
                )
                individual = await self._assess(candidate, "repair")
            if individual is not None:
                seeds.append(individual)
                self._select(self.population + [individual])
        return tuple(seeds)

    async def _enhance(self, idea):
        for _ in range(self.search_generations):
            parents = tuple(self.population)
            if not parents:
                break
            for _ in range(self.population_size):
                left, right = [
                    min(self.rng.choices(parents, k=2), key=self._rank) for _ in range(2)
                ]
                content = left.content
                try:
                    if self.rng.random() < idea.crossover_rate:
                        content = await self.crossover(
                            content, right.content, idea.directions, self.rng
                        )
                    if self.rng.random() < idea.mutation_rate:
                        content = await self.mutate(content, idea.directions, self.rng)
                except CandidateRejected as error:
                    self.failures.append(Failure("variation", content, str(error)))
                    continue
                candidate = Candidate(content=content, explanation="Genetic search offspring")
                individual = await self._assess(candidate, "search")
                if individual is not None:
                    self._select(self.population + [individual])
            self.generations.append(tuple(self.population))

    async def run(
        self,
        idea: str | None = None,
        *,
        rounds: int = 3,
        feedback: Callable[[Round], Awaitable[str | None]] | None = None,
    ) -> Result:
        """Mine an idea interactively, or discover ideas autonomously when idea is None.

        Interactive runs stop after review without a feedback callback. Returning
        None from feedback stops; text guides the next round under the same task.
        Budget exhaustion returns partial results. Other errors propagate, retaining
        raw calls and partial state on the instance; there are no transport retries.
        """
        direction = ""
        reason = "rounds"
        try:
            for index in range(rounds):
                failure_start = len(self.failures)
                polished, knowledge = await self._ideate(idea, direction)
                seeds = await self._seed(polished, knowledge)
                await self._enhance(polished)
                analysis = await self._invoke(
                    self.review,
                    polished,
                    seeds,
                    tuple(self.population),
                    tuple(self.failures[failure_start:]),
                )
                report = Round(polished, seeds, tuple(self.population), analysis)
                self.rounds.append(report)
                if index + 1 == rounds:
                    break
                if feedback is not None:
                    direction = await feedback(report)
                    if direction is None:
                        reason = "feedback"
                        break
                elif idea is not None:
                    reason = "feedback"
                    break
                else:
                    direction = analysis.next_direction
        except BudgetExceeded:
            reason = "budget"
        return Result(
            self.population[0] if self.population else None,
            tuple(self.population),
            tuple(self.rounds),
            len(self.calls),
            self.evaluations,
            tuple(self.failures),
            reason,
        )

    async def _invoke(self, operation, *args):
        if len(self.calls) >= self.max_calls:
            raise BudgetExceeded
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Retain raw output before Slick parses it; no mutable Session is shared."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("Alpha-GPT prompt operations do not accept tool requests")
        return response, requests
