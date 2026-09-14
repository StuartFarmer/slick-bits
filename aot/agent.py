"""Explore alternatives within one generation using Algorithm of Thoughts."""

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Example:
    input: str
    search: str
    answer: str


@dataclass(frozen=True)
class Result:
    output: str
    response: str
    warm_start: str
    calls: int
    evaluation: float | None


def final_answer(response: str) -> str:
    """Extract the artifact after the first standalone answer: line, preserving bytes."""
    marker = re.search(r"(?im)^answer:[ \t]*\r?\n", response)
    if marker is None or not response[marker.end() :].strip():
        raise ValueError("generated response requires an answer: line and nonblank artifact")
    return response[marker.end() :]


class AoT:
    """Own in-context examples, optional preparation, and one uninterrupted search.

    The provider owns decoding settings and transport retries. The optional async
    evaluator assesses the final artifact once; it does not guide or repeat search.
    Configure Slick's template root at application startup, outside this class.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]] | None = None,
        *,
        examples: Sequence[Example] = (),
        warmup_instructions: str = "",
        plans: int = 5,
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.examples = tuple(examples)
        self.warmup_instructions = warmup_instructions
        self.plans = plans
        self.calls: list[dict] = []
        self.output: str | None = None
        self.warm_start = ""
        self.evaluation_error: str | None = None

    @prompt(template="prepare.j2")
    async def prepare(self, input: str, *, generated: str) -> str:
        """Propose compatible starting candidates in one optional warm-up call."""
        if not generated.strip():
            raise ValueError("generated warm-up is blank")
        return generated

    @prompt(template="dfs.j2")
    async def solve_dfs(self, input: str, warm_start: str, *, generated: str) -> str:
        """Demonstrate subtree exploration, pruning, and backtracking in context."""
        return final_answer(generated)

    @prompt(template="bfs.j2")
    async def solve_bfs(self, input: str, warm_start: str, *, generated: str) -> str:
        """Demonstrate level-wise exploration within the same model generation."""
        return final_answer(generated)

    @prompt(template="plans.j2")
    async def solve_plans(self, input: str, warm_start: str, *, generated: str) -> str:
        """Generate plans, select, produce, and refine within one response."""
        return final_answer(generated)

    async def run(
        self,
        input: str,
        *,
        strategy: Literal["dfs", "bfs", "plans"] = "plans",
        warmup: bool = False,
        warm_start: str | None = None,
    ) -> Result:
        """Use one search call, plus preparation only when requested without a start.

        Runs reset state; use one run at a time per instance. Failures propagate
        without retries or repair, retaining raw calls and any completed artifact.
        A generated answer is not a guarantee of correctness or search completeness.
        """
        self.calls = []
        self.output = None
        self.warm_start = "" if warm_start is None else warm_start
        self.evaluation_error = None
        solve = {"dfs": self.solve_dfs, "bfs": self.solve_bfs, "plans": self.solve_plans}[strategy]
        if warmup and warm_start is None:
            self.warm_start = await self._invoke(self.prepare, input)
        self.output = await self._invoke(solve, input, self.warm_start)
        evaluation = await self.assess(self.output)
        return Result(
            self.output, self.calls[-1]["response"], self.warm_start, len(self.calls), evaluation
        )

    async def assess(self, output: str) -> float | None:
        """Evaluate only the final artifact, retaining evaluation failures separately."""
        if self.evaluate is None:
            return None
        try:
            score = await self.evaluate(output)
            if not isfinite(score):
                raise ValueError("evaluation must be finite")
            return score
        except Exception as exc:
            self.evaluation_error = f"{type(exc).__name__}: {exc}"
            raise

    async def _invoke(self, operation, *args) -> str:
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Retain raw text before Slick postprocessing, including rejected answers."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("AoT requires text generation without tool requests")
        return response, requests
