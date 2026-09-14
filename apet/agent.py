"""Reformulate arbitrary prompts with APET, then answer in an independent call."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import isfinite
from types import SimpleNamespace

from slick import Prompt, prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Result:
    original_prompt: str
    optimized_prompt: str
    answer: str
    original_answer: str | None
    original_score: float | None
    optimized_score: float | None
    calls: int


def nonblank(generated: str) -> str:
    """Reject empty generations without changing artifact whitespace."""
    if not generated.strip():
        raise ValueError("generated text is blank")
    return generated


class APET:
    """Own one zero-shot prompt rewrite and independent answer generation.

    task is the complete original prompt. Optional input_text identifies source
    material to restore verbatim if the rewrite omits it, as in the official code.
    Evaluation scores answers, never guides rewriting or selects a winner.
    Configure Slick's template root once at application startup.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]] | None = None,
        *,
        input_text: str = "",
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.input_text = input_text
        self.calls: list[dict] = []

    @prompt(template="reformulate.j2")
    async def reformulate(self, *, generated: str) -> str:
        """Apply the toolbox and restore protected input missing from the rewrite."""
        nonblank(generated)
        if self.input_text not in generated:
            generated += "\n\n" + Prompt("input.j2")(input_text=self.input_text)
        return generated

    @prompt(template="respond.j2")
    async def respond(self, sample_prompt: str, *, generated: str) -> str:
        """Answer only the supplied original or optimized prompt."""
        return nonblank(generated)

    async def optimize(self) -> str:
        """Make one rewrite call, reset records, and return the reusable prompt."""
        self.calls = []
        return await self._call(self.reformulate)

    async def run(self, *, compare: bool = False) -> Result:
        """Rewrite, optionally answer the baseline, then answer the rewrite.

        Makes two calls, or three with compare=True, before optional evaluation.
        Returns the optimized answer even if its score is worse. Calls are
        stateless; errors propagate without retries and retain raw call records.
        Use one run/optimize operation at a time per instance.
        """
        optimized = await self.optimize()
        original_answer = await self._call(self.respond, self.task) if compare else None
        answer = await self._call(self.respond, optimized)
        original_score = await self._score(original_answer)
        optimized_score = await self._score(answer)
        return Result(
            self.task,
            optimized,
            answer,
            original_answer,
            original_score,
            optimized_score,
            len(self.calls),
        )

    async def _score(self, answer: str | None) -> float | None:
        if self.evaluate is None or answer is None:
            return None
        score = await self.evaluate(answer)
        if not isfinite(score):
            raise ValueError("evaluation score must be finite")
        return score

    async def _call(self, operation, *args) -> str:
        record = {"operation": operation.__name__}
        self.calls.append(record)

        async def acall(context):
            record["prompt"] = context
            text, requests = await self.provider.acall(context)
            record["response"] = text
            if requests:
                raise ValueError("APET requires text responses, not tool requests")
            return text, requests

        try:
            return await operation(*args, provider=SimpleNamespace(acall=acall))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
