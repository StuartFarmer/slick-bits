"""Ensemble coupled rationale/answer samples over fixed, shuffled, or resampled prompts."""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from random import Random
from typing import Literal

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Example:
    input: str
    rationale: str
    answer: str


@dataclass(frozen=True)
class Sample:
    response: str
    rationale: str | None
    answer: str | None
    error: str | None = None


@dataclass(frozen=True)
class Result:
    answer: str
    counts: dict[str, int]
    consistency: float
    samples: tuple[Sample, ...]


def parse_response(response: str) -> tuple[str, str]:
    """Read rationale and final answer before a subsequent Q:, stripping one final period."""
    response = response.partition("\nQ:")[0]
    rationale, marker, answer = response.rpartition("The answer is ")
    answer = answer.strip().removesuffix(".").strip()
    if not marker or not rationale.strip() or not answer:
        raise ValueError("missing rationale or final answer")
    return rationale.strip(), answer


class RationaleEnsemble:
    """Own independent generation, leave-one-out rationale pools, and plurality voting.

    Providers own decoding settings and transport retries. No Sessions are shared.
    Parsing/normalization ValueErrors reject a draw; other errors abort the phase.
    Call records accumulate across phases. Use one operation at a time per instance.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        examples: Sequence[Example] = (),
        normalize_answer: Callable[[str], str] = str.strip,
        parse_response: Callable[[str], tuple[str, str]] = parse_response,
        rng: Random | None = None,
    ):
        self.task = task
        self.provider = provider
        self.examples = tuple(examples)
        self.normalize_answer = normalize_answer
        self.parse_response = parse_response
        self.rng = Random() if rng is None else rng
        self.samples: list[Sample] = []
        self.calls: list[dict] = []

    @prompt(template="generate.j2")
    async def generate(self, input: str, examples: Sequence[Example], *, generated: str) -> str:
        """Generate coupled rationale and answer as text in the paper's delimiter format."""
        return generated

    @prompt(template="sample_rationale.j2")
    async def sample_rationale(
        self, input: str, examples: Sequence[Example], *, generated: str
    ) -> str:
        """Predict a held-out exemplar without revealing its rationale or known answer."""
        return generated

    async def prepare_rationales(
        self, *, samples_per_example: int = 1024, provider: Provider | None = None
    ) -> tuple[tuple[str, ...], ...]:
        """Return reusable pools in exemplar order, retaining only answer-correct draws.

        Each exemplar gets exactly samples_per_example attempts. Empty pools raise
        after sampling; no repair, gold-answer hints, or human-rationale fallback.
        The optional provider supports stochastic preparation with greedy inference.
        """
        provider = self.provider if provider is None else provider
        pools = []
        for index, example in enumerate(self.examples):
            others = self.examples[:index] + self.examples[index + 1 :]
            expected = self.normalize_answer(example.answer)
            accepted = []
            for _ in range(samples_per_example):
                sample = await self.draw(self.sample_rationale, example.input, others, provider)
                if sample.answer is None:
                    continue
                if sample.answer != expected:
                    self.calls[-1]["rejection"] = "answer disagrees with exemplar ground truth"
                    continue
                accepted.append(sample.rationale)
            pools.append(tuple(accepted))
        empty = [i for i, pool in enumerate(pools) if not pool]
        if empty:
            raise ValueError(f"no accepted rationales for exemplars {empty}")
        return tuple(pools)

    async def run(
        self,
        input: str,
        *,
        method: Literal["self_consistency", "prompt_order", "input_rationale"] = "self_consistency",
        samples: int = 40,
        rationale_pools: Sequence[Sequence[str]] = (),
        replace_all: bool = False,
    ) -> Result:
        """Generate a fixed number of outputs and vote; reset output samples for this run.

        Input-rationale mode replaces one uniformly chosen exemplar per draw, or
        all exemplars with replace_all=True. Pools must correspond to these examples.
        Invalid outputs consume a draw; ties choose the first valid answer seen.
        """
        self.samples = []
        await self.sample_outputs(input, method, samples, rationale_pools, replace_all)
        return self.aggregate()

    async def sample_outputs(self, input, method, count, rationale_pools, replace_all) -> None:
        # ponytail: sequential calls; add bounded concurrency if model latency dominates.
        for _ in range(count):
            examples = self.select_examples(method, rationale_pools, replace_all)
            sample = await self.draw(self.generate, input, examples, self.provider)
            self.samples.append(sample)

    def select_examples(self, method, rationale_pools, replace_all) -> tuple[Example, ...]:
        examples = list(self.examples)
        if method == "prompt_order":
            self.rng.shuffle(examples)
        elif method == "input_rationale":
            if not examples or not rationale_pools or any(not pool for pool in rationale_pools):
                raise ValueError(
                    "input-rationale ensembling needs an accepted rationale pool per exemplar"
                )
            indices = range(len(examples)) if replace_all else [self.rng.randrange(len(examples))]
            for index in indices:
                examples[index] = replace(
                    examples[index], rationale=self.rng.choice(rationale_pools[index])
                )
        elif method != "self_consistency":
            raise ValueError(f"unknown ensemble method: {method}")
        return tuple(examples)

    def aggregate(self) -> Result:
        """Count every accepted draw, preserving repeated answers and repeated rationales."""
        counts = Counter(s.answer for s in self.samples if s.answer is not None)
        if not counts:
            raise ValueError("no valid answers among sampled responses")
        answer, votes = counts.most_common(1)[0]
        return Result(answer, dict(counts), votes / len(self.samples), tuple(self.samples))

    async def draw(self, operation, input, examples, provider) -> Sample:
        """Record raw output before extraction; only generated-content failures reject a draw."""
        record = {"operation": operation.__name__}
        self.calls.append(record)
        self._generation_provider = provider
        try:
            response = await operation(input, examples, provider=self)
            try:
                rationale, answer = self.parse_response(response)
                answer = self.normalize_answer(answer)
                if not rationale.strip() or not answer.strip():
                    raise ValueError("empty rationale or normalized answer")
            except ValueError as exc:
                record["rejection"] = str(exc)
                return Sample(response, None, None, str(exc))
            record["answer"] = answer
            return Sample(response, rationale, answer)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Proxy a single independent provider call, retaining text even on tool rejection."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self._generation_provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("rationale ensembles require text responses without tool requests")
        return response, requests
