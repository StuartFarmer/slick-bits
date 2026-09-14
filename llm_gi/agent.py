"""Sample LLM edits or accumulate strict improvements with Gin-style local search."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Target:
    """A replaceable region, using Python string offsets into the current artifact."""

    start: int
    end: int


@dataclass(frozen=True)
class Evaluation:
    """Caller-measured fitness (lower wins) and compilation/correctness outcomes."""

    fitness: float | None = None
    compiled: bool = True
    passed: bool = True
    detail: str = ""


@dataclass(frozen=True)
class Candidate:
    content: str
    fitness: float


class CandidateRejected(Exception):
    """An expected candidate validation or evaluation failure; consume the attempt."""


def whole_artifact(content: str) -> Sequence[Target]:
    return [Target(0, len(content))]


def nonblank(content: str) -> None:
    if not content.strip():
        raise CandidateRejected("empty content")


def identity(content: str) -> str:
    return content


class GeneticImprovement:
    """Own independent mutations, measured selection, and per-run audit records.

    The caller owns target discovery, syntax checks, classic edits, provider setup,
    and isolated evaluation. No generated content is executed by this class.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        *,
        context: str = "",
        language: str = "text",
        requirements: str = "",
        example_before: str = "",
        example_after: str = "",
        targets: Callable[[str], Sequence[Target]] = whole_artifact,
        validate: Callable[[str], None] = nonblank,
        fingerprint: Callable[[str], str] = identity,
        classic: Mapping[str, Callable[[str, random.Random], str]] | None = None,
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.context = context
        self.language = language
        self.requirements = requirements
        self.example_before = example_before
        self.example_after = example_after
        self.targets = targets
        self.validate = validate
        self.fingerprint = fingerprint
        self.classic = {} if classic is None else classic
        self.attempts: list[dict] = []
        self.evaluations = 0
        self.baseline: Candidate | None = None
        self.best: Candidate | None = None

    @prompt(template="simple.j2")
    async def simple(self, content: str, *, generated: str) -> str:
        """Request five variations without additional task or formatting context."""
        return generated

    @prompt(template="medium.j2")
    async def medium(self, content: str, *, generated: str) -> str:
        """Request five variations with caller context and a labeled fence contract."""
        return generated

    @prompt(template="detailed.j2")
    async def detailed(self, content: str, *, generated: str) -> str:
        """Add a fixed caller-supplied example of a useful change."""
        return generated

    async def run(
        self,
        initial: str,
        *,
        mode: Literal["local", "random"] = "local",
        operator: str = "MEDIUM",
        budget: int | None = None,
        seed: int = 123,
    ) -> Candidate | None:
        """Return the best passing candidate, resetting state on each run.

        Local budgets include one baseline slot (default 100 total). Random
        sampling uses every slot for a fresh mutation of initial (default 1000),
        with no baseline; it returns None if no candidate passes. Invalid edits
        and no-ops consume slots. Duplicates are evaluated again. Lower wins.
        """
        local = {"local": True, "random": False}[mode]
        budget = (100 if local else 1000) if budget is None else budget
        self.attempts = []
        self.evaluations = 0
        self.baseline = self.best = None
        self.rng = random.Random(seed)
        self.original_key = self.fingerprint(initial)
        self.seen = {self.original_key}
        if local:
            await self._measure_baseline(initial)
        for _ in range(budget - int(local)):
            parent = self.best.content if local else initial
            candidate = await self._attempt(parent, operator)
            if candidate is not None:
                self._select(candidate, self.attempts[-1])
        return self.best

    async def _measure_baseline(self, initial: str) -> None:
        record = {"operator": "BASELINE", "content": initial, "accepted": False}
        self.attempts.append(record)
        self.baseline = self.best = await self._assess(initial, record)
        if self.baseline is None:
            raise CandidateRejected(f"baseline failed: {record['status']}")
        record.update(status="baseline", accepted=True, improvement=0.0)

    async def _attempt(self, parent: str, operator: str) -> Candidate | None:
        record = {"operator": operator, "parent": parent, "accepted": False}
        self.attempts.append(record)
        try:
            content = await self._mutate(parent, operator, record)
            record.update(content=content, valid=True)
            key = self.fingerprint(content)
            record.update(fingerprint=key, unique=key not in self.seen)
            self.seen.add(key)
            if key in (self.original_key, self.fingerprint(parent)):
                record["status"] = "no_op"
                return None
            return await self._assess(content, record)
        except CandidateRejected as exc:
            record.update(status="invalid", error=str(exc))
            return None
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise

    async def _mutate(self, parent: str, operator: str, record: dict) -> str:
        methods = {"SIMPLE": self.simple, "MEDIUM": self.medium, "DETAILED": self.detailed}
        if operator not in methods:
            content = self.classic[operator](parent, self.rng)
            nonblank(content)
            self.validate(content)
            return content

        targets = self.targets(parent)
        if not targets:
            raise CandidateRejected("no eligible targets")
        target = self.rng.choice(targets)
        record["target"] = target
        response = await methods[operator](
            parent[target.start : target.end], provider=self.provider
        )
        record["raw_response"] = response
        record["invalid_suggestions"] = []
        # Retain Gin's labeled-fence boundary and first usable suggestion policy.
        pattern = r"```" + re.escape(self.language) + r"[ \t]*\r?\n(.*?)```"
        for match in re.finditer(pattern, response, re.DOTALL):
            replacement = match.group(1)
            content = parent[: target.start] + replacement + parent[target.end :]
            try:
                nonblank(replacement)
                self.validate(content)
            except CandidateRejected as exc:
                record["invalid_suggestions"].append(str(exc))
                continue
            return content
        raise CandidateRejected("no valid labeled suggestion")

    async def _assess(self, content: str, record: dict) -> Candidate | None:
        self.evaluations += 1
        try:
            result = await self.evaluate(content)
        except CandidateRejected as exc:
            record.update(status="rejected", error=str(exc))
            return None
        except TimeoutError as exc:
            record.update(status="timeout", error=str(exc))
            return None
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        record["evaluation"] = result
        if not result.compiled:
            record["status"] = "compile_failed"
        elif not result.passed:
            record["status"] = "test_failed"
        elif result.fitness is None or not math.isfinite(result.fitness):
            record["status"] = "nonfinite"
        else:
            record.update(status="passed", fitness=result.fitness)
            if self.baseline is not None:
                record["improvement"] = self.baseline.fitness - result.fitness
            return Candidate(content, result.fitness)
        return None

    def _select(self, candidate: Candidate, record: dict) -> None:
        if self.best is None or candidate.fitness < self.best.fitness:
            self.best = candidate
            record["accepted"] = True
