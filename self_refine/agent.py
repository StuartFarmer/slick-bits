"""Generate, critique, and refine text with the same model and accumulated feedback."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Step:
    output: str
    feedback: str


@dataclass(frozen=True)
class Result:
    output: str
    history: tuple[Step, ...]
    stop_reason: Literal["feedback", "budget"]
    calls: int


def feedback_says_stop(feedback: str, iteration: int) -> bool:
    """Recognize only a complete stopping response, ignoring surrounding whitespace."""
    return feedback.strip() == "NO_FEEDBACK"


def nonblank(generated: str) -> str:
    """Reject empty model responses without changing artifact whitespace."""
    if not generated.strip():
        raise ValueError("generated response is blank")
    return generated


class SelfRefine:
    """Own Algorithm 1's generation, feedback, refinement, and explicit history.

    Task instructions and optional prompt prefixes define the problem and few-shot
    examples. A synchronous stop(feedback, iteration) replaces the default marker
    check. Model configuration and transport retries belong to the caller.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        generation_prompt: str = "",
        feedback_prompt: str = "",
        refinement_prompt: str = "",
        stop: Callable[[str, int], bool] = feedback_says_stop,
    ):
        self.task = task
        self.provider = provider
        self.generation_prompt = generation_prompt
        self.feedback_prompt = feedback_prompt
        self.refinement_prompt = refinement_prompt
        self.stop = stop
        self.output: str | None = None
        self.history: list[Step] = []
        self.calls: list[dict] = []

    @prompt(template="generate.j2")
    async def generate(self, input: str, *, generated: str) -> str:
        """Generate y_0 from the task and input (Equation 1)."""
        return nonblank(generated)

    @prompt(template="feedback.j2")
    async def feedback(self, input: str, output: str, *, generated: str) -> str:
        """Critique only the current output against the input (Equation 2)."""
        return nonblank(generated)

    @prompt(template="refine.j2")
    async def refine(self, input: str, history: Sequence[Step], *, generated: str) -> str:
        """Refine using every previous output and feedback pair (Equation 4)."""
        return nonblank(generated)

    async def run(
        self,
        input: str,
        *,
        max_refinements: int = 4,
        initial_output: str | None = None,
    ) -> Result:
        """Return the last output after feedback stops the loop or the cap is reached.

        A cap of N permits N revisions and N+1 feedback calls, including feedback
        on the final output. Initial generation costs one more call unless supplied.
        Zero permits feedback but no revision. Failures propagate without retries;
        partial output, history, and raw call records remain on the instance.
        Runs reset state; use one run at a time per instance.
        """
        self.output = initial_output
        self.history = []
        self.calls = []
        if self.output is None:
            self.output = await self._invoke(self.generate, input)
        iteration = 0
        while True:
            feedback = await self._invoke(self.feedback, input, self.output)
            self.history.append(Step(self.output, feedback))
            if self.stop(feedback, iteration):
                reason = "feedback"
                break
            if iteration >= max_refinements:
                reason = "budget"
                break
            self.output = await self._invoke(self.refine, input, tuple(self.history))
            iteration += 1
        return Result(self.output, tuple(self.history), reason, len(self.calls))

    async def _invoke(self, operation, *args) -> str:
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Forward to the same provider, saving raw text before output validation."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("SELF-REFINE prompt operations require text, not tool requests")
        return response, requests
