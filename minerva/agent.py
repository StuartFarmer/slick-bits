"""Minerva inference: independent solutions, final-answer grouping, majority vote."""

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Example:
    problem: str
    solution: str
    answer: str


@dataclass(frozen=True)
class Sample:
    response: str
    answer: str | None
    key: str | None
    error: str | None = None


@dataclass(frozen=True)
class Vote:
    answer: str
    key: str
    sample_indices: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.sample_indices)


@dataclass(frozen=True)
class Result:
    samples: tuple[Sample, ...]
    votes: tuple[Vote, ...]
    selected: tuple[Vote, ...]
    correctness: tuple[bool, ...] | None
    calls: int

    @property
    def answer(self) -> str | None:
        return self.selected[0].answer if self.selected else None

    @property
    def output(self) -> str | None:
        return self.samples[self.selected[0].sample_indices[0]].response if self.selected else None

    @property
    def pass_at_k(self) -> bool | None:
        return any(self.correctness) if self.correctness is not None else None

    @property
    def maj_at_n(self) -> bool | None:
        if self.correctness is None:
            return None
        return any(self.correctness[vote.sample_indices[0]] for vote in self.selected)


def extract_final_answer(response: str) -> str | None:
    """Read the last paper-format answer; never guess from solution prose.

    Require the closing marker so truncated outputs cannot cast partial votes.
    Keep mathematical notation, units, case, internal whitespace, and punctuation.
    """
    starts = list(
        re.finditer(r"(?im)^[ \t]*Final answer:[ \t]*The final answer is[ \t]*", response)
    )
    if not starts:
        return None
    tail = response[starts[-1].end() :]
    end = re.search(r"\.\s+I hope it is correct\.?(?=\s|$)", tail, re.IGNORECASE)
    if end is None:
        return None
    return tail[: end.start()].strip() or None


class Minerva:
    """Own fixed-budget sampling and frequency selection for an arbitrary task.

    Inject a stateless provider configured for stochastic sampling (or greedy
    decoding for k=1). Optional evaluation is posthoc and never guides selection.
    Set Slick's process-global template root once in application startup.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[bool]] | None = None,
        *,
        examples: Sequence[Example] = (),
        extract: Callable[[str], str | None] = extract_final_answer,
        normalize: Callable[[str], str] = str.strip,
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.examples = tuple(examples)
        self.extract = extract
        self.normalize = normalize
        self.samples: list[Sample] = []
        self.calls: list[dict] = []

    @prompt(template="solve.j2")
    async def solve(self, *, generated: str) -> Sample:
        """Extract one text solution; an invalid answer remains an explicit sample."""
        answer = self.extract(generated)
        if answer is None or not answer.strip():
            return Sample(generated, None, None, "missing or empty final answer")
        key = self.normalize(answer)
        if not key.strip():
            return Sample(generated, answer, None, "empty normalized answer")
        return Sample(generated, answer, key)

    async def run(self, *, k: int = 16, n: int = 1) -> Result:
        """Draw k solutions, rank answer frequencies, and select the top n groups.

        Ties retain first occurrence; invalid samples consume budget but do not
        vote. All-invalid runs return an empty selection. No repair calls, shared
        history, tools, or early stopping. Exceptions retain records and propagate.
        """
        self.samples = []
        self.calls = []
        await self.sample_solutions(k)
        votes = self.rank_answers()
        selected = votes[:n]
        correctness = await self.grade_samples()
        return Result(tuple(self.samples), votes, selected, correctness, len(self.calls))

    async def sample_solutions(self, k: int) -> None:
        """Use the same prompt for every independent provider request."""
        for _ in range(k):
            record = {"operation": "solve"}
            self.calls.append(record)

            async def acall(context):
                record["prompt"] = context
                text, requests = await self.provider.acall(context)
                record["response"] = text
                if requests:
                    raise ValueError("Minerva requires text responses without tool requests")
                return text, requests

            try:
                sample = await self.solve(provider=SimpleNamespace(acall=acall))
                self.samples.append(sample)
                if sample.error:
                    record["error"] = sample.error
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise

    def rank_answers(self) -> tuple[Vote, ...]:
        """Group by caller-defined canonical keys; stable sorting breaks ties."""
        groups: dict[str, list[int]] = {}
        for index, sample in enumerate(self.samples):
            if sample.key is not None:
                groups.setdefault(sample.key, []).append(index)
        return tuple(
            Vote(self.samples[indices[0]].answer, key, tuple(indices))
            for key, indices in sorted(groups.items(), key=lambda item: -len(item[1]))
        )

    async def grade_samples(self) -> tuple[bool, ...] | None:
        """Grade extracted answers after selection; malformed samples are incorrect."""
        if self.evaluate is None:
            return None
        return tuple(
            [
                await self.evaluate(sample.answer) if sample.key is not None else False
                for sample in self.samples
            ]
        )
