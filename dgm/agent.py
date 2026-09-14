"""Evolve executable agents through self-modification and an open-ended archive."""

import math
import random
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints, ValidationError
from slick import Provider, prompt

SEED_AGENT = """async def agent(instruction, generate):
    return await generate(instruction)
"""

Generate = Callable[[str], Awaitable[str]]
Execute = Callable[[str, str, Generate], Awaitable[str]]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Proposal(BaseModel, extra="forbid"):
    improvement_proposal: Text
    implementation_suggestion: Text
    problem_description: Text


class ExecutionError(Exception):
    """The executor reports generated-program failure or an exhausted budget."""


class CandidateRejected(ValueError):
    """A generated candidate cannot enter the archive."""


@dataclass(frozen=True)
class Config:
    iterations: int = 80
    batch_size: int = 1
    max_generations: int = 16
    mode: Literal["dgm", "no_self_improve", "no_open_ended"] = "dgm"


@dataclass(frozen=True)
class Evaluation:
    score: float
    can_self_modify: bool
    feedback: str = ""


@dataclass(frozen=True)
class Candidate:
    id: int
    parent_id: int | None
    source: str
    evaluation: Evaluation


@dataclass
class Attempt:
    id: int
    parent_id: int
    modifier_id: int
    proposal: Proposal | None = None
    source: str | None = None
    evaluation: Evaluation | None = None
    accepted: bool = False
    error: str | None = None


@dataclass(frozen=True)
class Generation:
    attempt_id: int
    operation: str
    prompt: str
    response: str


@dataclass
class Result:
    archive: list[Candidate] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    generations: list[Generation] = field(default_factory=list)
    evaluation_calls: int = 0

    @property
    def best(self) -> Candidate:
        return max(self.archive, key=lambda candidate: candidate.evaluation.score)


class _RecordingProvider:
    """Keep raw responses even when Slick parsing or postprocessing rejects them."""

    def __init__(self, provider, records, attempt_id, operation):
        self.provider, self.records = provider, records
        self.attempt_id, self.operation = attempt_id, operation

    async def acall(self, context, *, tools=None, tool_results=None):
        response, calls = await self.provider.acall(context, tools=tools, tool_results=tool_results)
        self.records.append(Generation(self.attempt_id, self.operation, context, response))
        if calls:
            raise ExecutionError("The text generation endpoint does not execute tool requests")
        return response, calls


class _AgentProvider:
    """Route a modification prompt through the selected executable, not a fixed FM."""

    def __init__(self, execute: Execute, source: str, generate: Generate):
        self.execute, self.source, self.generate = execute, source, generate

    async def acall(self, context, *, tools=None, tool_results=None):
        return await self.execute(self.source, context, self.generate), []


class DGM:
    """Own one search; execute and evaluate must enforce caller-owned isolation.

    execute(source, instruction, generate) runs async agent(instruction, generate).
    evaluate(source) measures downstream quality and continued self-modification.
    Scores are higher-is-better values in [0, 1], matching upstream selection.
    Explicit candidate/execution failures are recorded; infrastructure errors escape.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        *,
        execute: Execute,
        seed: int = 0,
    ):
        self.task, self.provider = task, provider
        self.evaluate, self.execute = evaluate, execute
        self.rng = random.Random(seed)
        self.result = Result()

    @prompt(template="diagnose.j2", output_type=Proposal)
    async def diagnose(self, parent: Candidate, *, generated: Proposal) -> Proposal:
        return generated

    @prompt(template="modify.j2")
    async def modify(self, parent: Candidate, proposal: Proposal, *, generated: str) -> str:
        self.check_source(generated)
        if generated == parent.source:
            raise CandidateRejected("Self-modification returned unchanged source")
        return generated

    @prompt(template="generate.j2")
    async def generate(self, instruction: str, *, generated: str) -> str:
        return generated

    @staticmethod
    def check_source(source: str) -> None:
        if not source.strip():
            raise CandidateRejected("Empty agent source")
        try:
            compile(source, "<dgm-agent>", "exec")
        except (SyntaxError, ValueError) as exc:
            raise CandidateRejected(f"Agent does not compile: {exc}") from exc

    async def assess(self, source: str) -> Evaluation:
        self.result.evaluation_calls += 1
        evaluation = await self.evaluate(source)
        if not math.isfinite(evaluation.score) or not 0 <= evaluation.score <= 1:
            raise CandidateRejected("Measured score must be finite and in [0, 1]")
        return evaluation

    async def initialize(self, source: str) -> None:
        self.check_source(source)
        evaluation = await self.assess(source)
        if not evaluation.can_self_modify:
            raise CandidateRejected("Initial agent cannot self-modify")
        self.result.archive.append(Candidate(0, None, source, evaluation))

    def select_parents(self, count: int, mode: str) -> list[Candidate]:
        archive = self.result.archive
        if mode == "no_open_ended":
            return [archive[-1]] * count
        children = Counter(candidate.parent_id for candidate in archive)
        # Adapted from DGM_outer.choose_selfimproves, score_child_prop (see SOURCES.md).
        weights = [
            1
            / (1 + math.exp(-10 * (candidate.evaluation.score - 0.5)))
            / (1 + children[candidate.id])
            for candidate in archive
        ]
        return self.rng.choices(archive, weights=weights, k=count)

    async def branch(self, parent: Candidate, config: Config) -> Candidate | None:
        modifier = self.result.archive[0] if config.mode == "no_self_improve" else parent
        attempt = Attempt(len(self.result.attempts) + 1, parent.id, modifier.id)
        self.result.attempts.append(attempt)
        records = self.result.generations
        diagnosis_provider = _RecordingProvider(self.provider, records, attempt.id, "diagnose")
        generation_provider = _RecordingProvider(self.provider, records, attempt.id, "generate")
        calls = 0

        async def generate(instruction: str) -> str:
            nonlocal calls
            if calls >= config.max_generations:
                raise ExecutionError("Self-modification generation budget exhausted")
            calls += 1
            return await self.generate(instruction, provider=generation_provider)

        executor = _AgentProvider(self.execute, modifier.source, generate)
        modification_provider = _RecordingProvider(executor, records, attempt.id, "modify")
        try:
            try:
                attempt.proposal = await self.diagnose(parent, provider=diagnosis_provider)
            except ValidationError as exc:
                raise CandidateRejected(f"Invalid diagnosis: {exc}") from exc
            attempt.source = await self.modify(
                parent, attempt.proposal, provider=modification_provider
            )
            attempt.evaluation = await self.assess(attempt.source)
            if not attempt.evaluation.can_self_modify:
                raise CandidateRejected("Agent lost self-modification capability")
        except (CandidateRejected, ExecutionError) as exc:
            attempt.error = str(exc)
            return None
        attempt.accepted = True
        return Candidate(attempt.id, parent.id, attempt.source, attempt.evaluation)

    async def run(self, initial: str = SEED_AGENT, *, config: Config = Config()) -> Result:
        await self.initialize(initial)
        batch_size = 1 if config.mode == "no_open_ended" else config.batch_size
        for start in range(0, config.iterations, batch_size):
            parents = self.select_parents(min(batch_size, config.iterations - start), config.mode)
            # ponytail: serial workers; add isolated parallel workers when throughput matters.
            for parent in parents:
                child = await self.branch(parent, config)
                if child is not None:
                    self.result.archive.append(child)
        return self.result
