"""Select an evolved reasoning instruction for one problem, rewrite, and answer it."""

import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class GeneratedText(BaseModel, extra="forbid"):
    text: Text


class Selection(BaseModel, extra="forbid"):
    index: Annotated[int, Field(strict=True, ge=0)]


class EoT:
    """Use model selection, not fitness optimization; evaluate only the final answer.

    The evaluator receives the final answer and may close over the problem.
    Parsing, provider, and evaluator errors propagate without retries.
    """

    def __init__(self, task: str, provider: Provider, evaluate: Callable[[str], Awaitable[float]]):
        self.task, self.provider, self.evaluate = task, provider, evaluate

    @prompt(template="crossover.j2", output_type=GeneratedText)
    async def crossover(self, parents: Sequence[str], *, generated: GeneratedText) -> str:
        return generated.text

    @prompt(template="mutate.j2", output_type=GeneratedText)
    async def mutate(self, instruction: str, *, generated: GeneratedText) -> str:
        return generated.text

    @prompt(template="select.j2", output_type=Selection)
    async def select(self, problem: str, candidates: Sequence[str], *, generated: Selection) -> int:
        if generated.index >= len(candidates):
            raise ValueError(f"selected index outside candidate pool: {generated.index}")
        return generated.index

    @prompt(template="rewrite.j2", output_type=GeneratedText)
    async def rewrite(self, problem: str, instruction: str, *, generated: GeneratedText) -> str:
        return generated.text

    @prompt(template="solve.j2", output_type=GeneratedText)
    async def solve(
        self, original: str, rewritten: str, instruction: str, *, generated: GeneratedText
    ) -> str:
        return generated.text

    @prompt(template="extract.j2", output_type=GeneratedText)
    async def extract(self, solution: str, answer_format: str, *, generated: GeneratedText) -> str:
        return generated.text

    async def _evolve(self, initial):
        crossed = await self.crossover(initial, provider=self.provider)
        mutated = await self.mutate(crossed, provider=self.provider)
        return [*initial, crossed, mutated]

    async def _answer(self, problem, instruction, answer_format):
        rewritten = await self.rewrite(problem, instruction, provider=self.provider)
        solution = await self.solve(problem, rewritten, instruction, provider=self.provider)
        answer = await self.extract(solution, answer_format, provider=self.provider)
        return rewritten, solution, answer

    async def run(
        self,
        problem: str,
        *,
        initial: Sequence[str] = (
            "Let's think step by step.",
            "Understand the problem, devise a plan, then carry it out step by step.",
        ),
        answer_format: str = "Return the final answer without commentary.",
    ) -> dict:
        candidates = await self._evolve(initial)
        index = await self.select(problem, candidates, provider=self.provider)
        rewritten, solution, answer = await self._answer(problem, candidates[index], answer_format)
        score = float(await self.evaluate(answer))
        if not math.isfinite(score):
            raise ValueError("measured score must be finite")
        return {
            "instruction": candidates[index],
            "candidates": candidates,
            "rewritten": rewritten,
            "solution": solution,
            "answer": answer,
            "score": score,
            "evaluations": 1,
        }
