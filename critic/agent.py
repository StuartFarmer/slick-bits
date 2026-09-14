"""Generate, verify with external tools, and correct using CRITIC Algorithm 1."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, StrictBool, StringConstraints
from slick import Session, prompt
from slick.providers import Provider
from slick.tools import Tool


class Critique(BaseModel, extra="forbid"):
    correct: StrictBool
    feedback: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@dataclass(frozen=True)
class Step:
    output: str
    critique: Critique
    evidence: tuple[dict, ...]


@dataclass(frozen=True)
class Result:
    output: str
    history: tuple[Step, ...]
    stop_reason: Literal["verified", "unchanged", "budget"]
    calls: int


def nonblank(generated: str) -> str:
    """Validate generated artifacts without modifying their whitespace."""
    if not generated.strip():
        raise ValueError("generated output is blank")
    return generated


class CRITIC:
    """Own task instructions, external tools, generation boundaries, and run state.

    Tools are ordinary annotated functions or Slick Tools. The caller owns tool
    permissions, isolation, deadlines, and provider configuration. Each verification
    gets a fresh Session; generation and correction are independent text calls.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        tools: Sequence[Tool | Callable],
        *,
        criteria: str = "Check that the output satisfies the task and input constraints.",
        examples: str = "",
    ):
        self.task = task
        self.provider = provider
        self.tools = list(tools)
        self.criteria = criteria
        self.examples = examples
        self.output: str | None = None
        self.history: list[Step] = []
        self.calls: list[dict] = []
        self.verification_session: Session | None = None

    @prompt(template="generate.j2")
    async def generate(self, input: str, *, generated: str) -> str:
        """Produce the initial output from the task, demonstrations, and input."""
        return nonblank(generated)

    @prompt(template="critique.j2", output_type=Critique, max_turns=8)
    async def critique(self, input: str, output: str, *, generated: Critique) -> Critique:
        """Produce a critique after at most eight provider turns of tool interaction."""
        return generated

    @prompt(template="revise.j2")
    async def revise(
        self,
        input: str,
        output: str,
        critique: Critique,
        evidence: Sequence[dict],
        *,
        generated: str,
    ) -> str:
        """Correct the current output using its critique and actual tool results."""
        return nonblank(generated)

    async def run(
        self,
        input: str,
        *,
        max_iterations: int = 3,
        initial_output: str | None = None,
        unchanged_patience: int | None = None,
    ) -> Result:
        """Return y_n, or stop when a critique accepts y_i (Algorithm 1).

        N permits N verifications and N corrections; y_n is not verified again.
        Zero returns the initial output. Optional patience counts consecutive
        byte-identical corrections, which do not establish correctness.
        Failures propagate without retries, retaining partial state and raw calls.
        Runs reset state; use one active run per instance.
        """
        self.output, self.history, self.calls = initial_output, [], []
        self.verification_session = None
        if self.output is None:
            self.output = await self._invoke(self.generate, input)
        unchanged = 0
        reason = "budget"
        for _ in range(max_iterations):
            step = await self.verify(input, self.output)
            self.history.append(step)
            if step.critique.correct:
                reason = "verified"
                break
            self.output = await self._invoke(
                self.revise, input, step.output, step.critique, step.evidence
            )
            unchanged = unchanged + 1 if self.output == step.output else 0
            if unchanged_patience is not None and unchanged >= unchanged_patience:
                reason = "unchanged"
                break
        return Result(self.output, tuple(self.history), reason, len(self.calls))

    async def verify(self, input: str, output: str) -> Step:
        """Keep model-selected tool requests and results together for correction."""
        self.verification_session = Session(provider=self, tools=self.tools)
        critique = await self._invoke(
            self.critique, input, output, session=self.verification_session
        )
        evidence = tuple(
            work["result"]
            for exchange in self.verification_session.history
            for work in exchange["work"]
            if work["result"] is not None
        )
        if critique.correct and not any(
            not result.get("is_error", False) and result["content"].strip() for result in evidence
        ):
            error = "Cannot accept correctness without successful external tool feedback"
            self.calls[-1]["error"] = error
            raise ValueError(error)
        return Step(output, critique, evidence)

    async def _invoke(self, operation, *args, session=None):
        self._operation = operation.__name__
        previous_calls = len(self.calls)
        try:
            if session is not None:
                return await operation(*args, session=session)
            return await operation(*args, provider=self)
        except Exception as exc:
            if len(self.calls) > previous_calls:
                self.calls[-1]["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Log every raw response before Slick parses it, including failed calls."""
        record = {"operation": self._operation, "prompt": context}
        self.calls.append(record)
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record.update(response=response, requests=requests)
        if requests and tools is None:
            raise ValueError("Tools are only available during verification")
        return response, requests

    @property
    def aturn(self):
        """Keep native provider continuation; let Session handle portable providers."""
        return self._aturn if getattr(self.provider, "aturn", None) is not None else None

    async def _aturn(self, context, **kwargs):
        record = {"operation": self._operation, "prompt": context}
        self.calls.append(record)
        response = await self.provider.aturn(context, **kwargs)
        record.update(response=response["text"], requests=response["requests"])
        return response
