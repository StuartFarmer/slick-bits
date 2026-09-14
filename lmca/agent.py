"""Accept every validated mutation, repairing failures before switching providers."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Validation:
    """A canonical program on success, or None and a repair reason on rejection."""

    program: str | None
    reason: str = ""


@dataclass(frozen=True)
class Result:
    programs: tuple[str, ...]
    stop_reason: Literal["budget", "validation_exhausted"]
    calls: int


class LMCA:
    """Run the paper's neutral chain with caller-owned syntax/type validation.

    The validator returns Validation; exceptions indicate operational failures,
    not candidate rejection. No fitness, novelty filter, or execution is added.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        validate: Callable[[str], Awaitable[Validation]],
        *,
        constraints: str = "",
        instruction: str = "Produce a mutated program.",
        fallback: Provider | None = None,
    ):
        self.task = task
        self.provider = provider
        self.validate = validate
        self.constraints = constraints
        self.instruction = instruction
        self.fallback = fallback
        self.programs: list[str] = []
        self.calls: list[dict] = []

    @prompt(template="mutate.j2")
    async def mutate(self, parent: str, *, generated: str) -> Validation:
        """Produce one variant and validate its representation, without selection."""
        return await self._validate(generated)

    @prompt(template="repair.j2")
    async def repair(self, candidate: str, reason: str, *, generated: str) -> Validation:
        """Repair the last invalid candidate using only constraints and failure feedback."""
        return await self._validate(generated)

    async def _validate(self, generated: str) -> Validation:
        if not generated.strip():
            return Validation(None, "The generated program is blank.")
        return await self.validate(generated)

    async def run(self, initial: str, *, steps: int = 300, retries: int = 5) -> Result:
        """Return the seed plus accepted mutations, including revisits and self-loops.

        retries means additional attempts: each provider gets 1 + retries calls.
        Fallback repairs the primary's last failure. Each new step starts with
        primary mutation again. Exhaustion ends the chain without a fake step.
        Runs reset state; do not run one instance concurrently.
        """
        self.programs, self.calls = [initial], []
        reason = "budget"
        for _ in range(steps):
            candidate = await self._next_mutation(self.programs[-1], retries)
            if candidate is None:
                reason = "validation_exhausted"
                break
            self.programs.append(candidate)
        return Result(tuple(self.programs), reason, len(self.calls))

    async def _next_mutation(self, parent: str, retries: int) -> str | None:
        providers = [("primary", self.provider)]
        if self.fallback is not None:
            providers.append(("fallback", self.fallback))
        failure = None
        candidate = ""
        for stage, provider in providers:
            for attempt in range(retries + 1):
                if failure is None:
                    result = await self._invoke(provider, stage, attempt, self.mutate, parent)
                else:
                    result = await self._invoke(
                        provider, stage, attempt, self.repair, candidate, failure
                    )
                if result.program is not None:
                    return result.program
                candidate = self.calls[-1]["response"]
                failure = result.reason or "The candidate failed validation."
        return None

    async def _invoke(self, provider, stage, attempt, operation, *args) -> Validation:
        record = {
            "step": len(self.programs),
            "stage": stage,
            "attempt": attempt + 1,
            "operation": operation.__name__,
        }
        self.calls.append(record)

        async def acall(context):
            record["prompt"] = context
            response, requests = await provider.acall(context)
            record["response"] = response
            if requests:
                raise ValueError("LMCA requires program text, not tool requests")
            return response, requests

        try:
            result = await operation(*args, provider=SimpleNamespace(acall=acall))
            record["accepted"] = result.program is not None
            record["reason"] = result.reason
            return result
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
