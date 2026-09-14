"""Bootstrap rationale generation through answer filtering and base-model fine-tuning."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Example:
    input: str
    answer: str


@dataclass(frozen=True)
class Demonstration:
    input: str
    rationale: str
    answer: str


class Solution(BaseModel, extra="forbid", frozen=True):
    rationale: str
    answer: str


@dataclass(frozen=True)
class TrainingExample:
    example: Example
    prompt: str
    completion: str
    source: Literal["generate", "rationalize"]


@dataclass(frozen=True)
class Iteration:
    number: int
    generated: int
    rationalized: int


@dataclass(frozen=True)
class Result:
    provider: Provider
    history: tuple[Iteration, ...]
    stop_reason: Literal["budget", "no_training_examples"]


Evaluate = Callable[[str, str, str], Awaitable[bool]]
FineTune = Callable[[Provider, tuple[TrainingExample, ...], int], Awaitable[Provider]]


def checked_solution(generated: Solution) -> Solution:
    """Reject blank fields while preserving whitespace in nonblank artifacts."""
    if not generated.rationale.strip() or not generated.answer.strip():
        raise ValueError("Generated rationale and answer must both be nonblank")
    return generated


class STaR:
    """Own Algorithm 1's generation, rationalization, filtering, and training loop.

    evaluate(input, answer, expected) checks only final-answer correctness.
    fine_tune(base_provider, training, iteration) must train a fresh copy of the
    ORIGINAL checkpoint, leaving that checkpoint intact, and return its provider.
    Iterations are one-based; the callback owns the training schedule and backend.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Evaluate,
        fine_tune: FineTune,
        *,
        demonstrations: Sequence[Demonstration] = (),
    ):
        self.task = task
        self.base_provider = provider
        self.provider = provider
        self.evaluate = evaluate
        self.fine_tune = fine_tune
        self.demonstrations = tuple(demonstrations)
        self.training: tuple[TrainingExample, ...] = ()
        self.history: list[Iteration] = []
        self.calls: list[dict] = []

    @prompt(template="generate.j2", output_type=Solution)
    async def generate(self, input: str, *, generated: Solution) -> Solution:
        """Generate a rationale followed by an answer without a target hint."""
        return checked_solution(generated)

    @prompt(template="rationalize.j2", output_type=Solution)
    async def rationalize(self, input: str, answer: str, *, generated: Solution) -> Solution:
        """Generate a new rationale conditioned on the known correct answer."""
        return checked_solution(generated)

    async def run(
        self,
        examples: Sequence[Example],
        *,
        iterations: int = 4,
        rationalization: bool = True,
    ) -> Result:
        """Return the last trained provider, or stop when no examples are accepted.

        Each round replaces the training set. No hidden retries, accumulated
        training data, or automatic stopping based on training-set accuracy.
        State resets per run; use one run at a time on an instance.
        """
        self.provider = self.base_provider
        self.training, self.history, self.calls = (), [], []
        examples = tuple(examples)
        reason = "budget"
        for iteration in range(1, iterations + 1):
            self.training = ()
            generated, failed = await self._generate(examples)
            rationalized = await self._rationalize(failed) if rationalization else []
            self.training = tuple(generated + rationalized)
            if not self.training:
                reason = "no_training_examples"
                break
            # Algorithm 1, line 7: always M, never the previous round's M_n.
            self.provider = await self.fine_tune(self.base_provider, self.training, iteration)
            self.history.append(Iteration(iteration, len(generated), len(rationalized)))
        return Result(self.provider, tuple(self.history), reason)

    async def _generate(
        self, examples: Sequence[Example]
    ) -> tuple[list[TrainingExample], list[Example]]:
        accepted, failed = [], []
        for example in examples:
            row = await self._sample(example, "generate")
            if row is None:
                failed.append(example)
            else:
                accepted.append(row)
        return accepted, failed

    async def _rationalize(self, failed: Sequence[Example]) -> list[TrainingExample]:
        accepted = []
        for example in failed:
            row = await self._sample(example, "rationalize")
            if row is not None:
                accepted.append(row)
        return accepted

    async def _sample(
        self, example: Example, source: Literal["generate", "rationalize"]
    ) -> TrainingExample | None:
        record = {"iteration": len(self.history) + 1, "operation": source, "input": example.input}
        self.calls.append(record)
        try:
            if source == "generate":
                solution = await self.generate(example.input, provider=self)
            else:
                solution = await self.rationalize(example.input, example.answer, provider=self)
            correct = await self.evaluate(example.input, solution.answer, example.answer)
            record["correct"] = correct
            if not correct:
                return None
            # Re-render the unhinted prompt, including unhinted demonstrations.
            context = await STaR.generate.render(self, example.input)
            completion = solution.model_copy(update={"answer": example.answer}).model_dump_json()
            return TrainingExample(example, context, completion, source)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Save raw output before Slick parses it; calls are sequential and stateless."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("STaR requires rationale/answer responses, not tool requests")
        return response, requests
