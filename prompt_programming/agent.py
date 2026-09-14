"""Compose natural-language prompts and insert fragments at likely continuation boundaries."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

from slick import Prompt, prompt
from slick.prompts import PromptError
from slick.providers import Provider

Evaluator = Callable[[str], Awaitable[float]]
SuffixScorer = Callable[[str, str], Awaitable[float]]
TokenBoundaries = Callable[[str], Sequence[int]]
Mode = Literal["metaprompt", "direct", "demonstration", "proxy"]


@dataclass(frozen=True)
class Result:
    answer: str
    fills: tuple[str, ...]
    transcript: str
    score: float | None
    calls: int


def nonblank(text: str) -> str:
    if not text.strip():
        raise ValueError("generated text is blank")
    return text


class PromptProgrammer:
    """Run task specification or a multipart metaprompt with explicit continuations.

    Configure Slick's template root before use. Providers own decoding limits and
    transport retries. Optional counterfactual callbacks supply character offsets
    at model-token boundaries and log P(suffix | prefix), respectively. No Session,
    dataset, generated-code execution, or implicit model judge is used.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Evaluator | None = None,
        *,
        token_boundaries: TokenBoundaries | None = None,
        score_suffix: SuffixScorer | None = None,
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.token_boundaries = token_boundaries
        self.score_suffix = score_suffix
        self.calls: list[dict] = []

    @prompt(template="direct.j2")
    async def direct(self, *, generated: str) -> str:
        """Specify the task directly and generate an answer."""
        return nonblank(generated)

    @prompt(template="demonstrate.j2")
    async def demonstrate(
        self,
        examples: Sequence[tuple[str, str]],
        input_label: str,
        output_label: str,
        *,
        generated: str,
    ) -> str:
        """Use the paper's simple-colon format with zero or more demonstrations."""
        return nonblank(generated)

    @prompt(template="proxy.j2")
    async def by_proxy(self, proxy: str, *, generated: str) -> str:
        """Use a caller-selected character or archetype as a task signifier."""
        return nonblank(generated)

    @prompt(template="fill.j2")
    async def fill(self, context: str, *, generated: str) -> str:
        """Generate a prose slot without appending output-format instructions."""
        return nonblank(generated)

    @prompt(template="answer.j2")
    async def answer(self, context: str, *, generated: str) -> str:
        """Complete the final fragment; preserve exact artifact whitespace."""
        return nonblank(generated)

    async def run(
        self,
        *,
        mode: Mode = "metaprompt",
        fragments: tuple[str, ...] | None = None,
        examples: Sequence[tuple[str, str]] = (),
        input_label: str = "Input",
        output_label: str = "Output",
        proxy: str = "a patient teacher",
        max_calls: int = 512,
    ) -> Result:
        """Reset records, generate an answer, and optionally evaluate it once.

        Each fragment precedes one generated slot; the last slot is the answer.
        Fragments include their own whitespace. Default: serializing seed then
        injected verdict. Without scoring callbacks, each complete provider
        response fills a slot. max_calls counts generations, suffix scores, and
        evaluation attempts. Exhaustion and errors propagate; records survive.
        """
        self.calls = []
        self.max_calls = max_calls
        fills: tuple[str, ...] = ()
        if mode == "metaprompt":
            if fragments is None:
                fragments = (Prompt("seed.j2")(), "\n" + Prompt("verdict.j2")())
            context = self.task + "\n" + fragments[0]
            context, fills = await self._fill_slots(context, fragments[1:])
            answer = await self._generate(self.answer, context, verbatim=context)
        else:
            operation, args = {
                "direct": (self.direct, ()),
                "demonstration": (self.demonstrate, (examples, input_label, output_label)),
                "proxy": (self.by_proxy, (proxy,)),
            }[mode]
            answer = await self._generate(operation, *args)
        transcript = self.calls[-1]["prompt"] + answer
        score = await self._assess(answer)
        return Result(answer, fills, transcript, score, len(self.calls))

    async def _fill_slots(
        self, context: str, fragments: tuple[str, ...]
    ) -> tuple[str, tuple[str, ...]]:
        fills = []
        for fragment in fragments:
            generated = await self._generate(self.fill, context, verbatim=context)
            selected = generated
            if self.score_suffix is not None:
                selected = await self._select_prefix(context, generated, fragment)
            fills.append(selected)
            context += selected + fragment
        return context, tuple(fills)

    async def _select_prefix(self, context: str, generated: str, suffix: str) -> str:
        # Author's substring_logprobs procedure: score every candidate boundary,
        # then select globally. Stopping at the first decrease can miss the peak.
        best_prefix, best_score = None, -math.inf
        for offset in self.token_boundaries(generated):
            prefix = generated[:offset]
            record = {"operation": "score_suffix", "prefix": context + prefix, "suffix": suffix}

            async def score():
                value = float(await self.score_suffix(context + prefix, suffix))
                record["score"] = value
                if math.isnan(value) or value > 0:
                    raise ValueError("suffix score must be a nonpositive log probability")
                return value

            value = await self._invoke(score, record)
            if value > best_score:
                best_prefix, best_score = prefix, value
        if best_prefix is None:
            raise ValueError("no boundary has a possible suffix continuation")
        return best_prefix

    async def _assess(self, answer: str) -> float | None:
        if self.evaluate is None:
            return None
        record = {"operation": "evaluate", "answer": answer}

        async def evaluate():
            score = float(await self.evaluate(answer))
            record["score"] = score
            if not math.isfinite(score):
                raise ValueError("evaluation score must be finite")
            return score

        return await self._invoke(evaluate, record)

    async def _generate(self, operation, *args, verbatim: str | None = None) -> str:
        record = {"operation": operation.__name__}

        async def acall(rendered):
            # Slick 0.3 strips rendered text. Continuation/scoring prefixes must
            # retain their exact whitespace, including partial-word boundaries.
            context = rendered if verbatim is None else verbatim
            record["prompt"] = context
            response, requests = await self.provider.acall(context)
            record["response"] = response
            if requests:
                record["tool_requests"] = requests
                raise PromptError("prompt programming requires text continuations")
            return response, requests

        return await self._invoke(
            lambda: operation(*args, provider=SimpleNamespace(acall=acall)), record
        )

    async def _invoke(self, operation: Callable[[], Awaitable], record: dict):
        if len(self.calls) >= self.max_calls:
            raise RuntimeError("call budget exhausted")
        self.calls.append(record)
        try:
            return await operation()
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
