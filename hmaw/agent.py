"""Optimize a query through CEO and Manager instructions, then generate its answer.

Adapted from liuyvchi/HMAW, Copyright (c) 2024 ycliu; see LICENSE and README.md.
"""

from dataclasses import dataclass
from types import SimpleNamespace

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Result:
    answer: str
    ceo_instructions: str
    manager_instructions: str
    optimized_prompt: str
    calls: int


def nonblank(generated: str) -> str:
    """Reject blank generated text without altering artifact whitespace."""
    if not generated.strip():
        raise ValueError("generated text is blank")
    return generated


class HMAW:
    """Own the three-stage workflow and its raw call records.

    Configure Slick's template root at application startup. Each run resets
    records; use one run at a time per instance. Errors propagate immediately
    with partial records retained. Callers own transport retries and evaluation.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        ceo_provider: Provider | None = None,
        manager_provider: Provider | None = None,
    ):
        self.task = task
        self.provider = provider
        self.ceo_provider = provider if ceo_provider is None else ceo_provider
        self.manager_provider = provider if manager_provider is None else manager_provider
        self.calls: list[dict] = []

    @prompt(template="ceo.j2")
    async def instruct_manager(self, *, generated: str) -> str:
        """Generate high-level instructions for the Manager."""
        return nonblank(generated)

    @prompt(template="manager.j2")
    async def instruct_worker(self, ceo_instructions: str, *, generated: str) -> str:
        """Turn CEO guidance and the original query into Worker instructions."""
        return nonblank(generated)

    @prompt(template="worker.j2")
    async def respond(self, manager_instructions: str, *, generated: str) -> str:
        """Answer the original query using the Manager's instructions."""
        return nonblank(generated)

    async def run(self) -> Result:
        """Make exactly three sequential text generations, without shared history."""
        self.calls = []
        ceo = await self._call(self.instruct_manager, self.ceo_provider)
        manager = await self._call(self.instruct_worker, self.manager_provider, ceo)
        answer = await self._call(self.respond, self.provider, manager)
        return Result(answer, ceo, manager, self.calls[-1]["prompt"], len(self.calls))

    async def _call(self, operation, source: Provider, *args) -> str:
        record = {"operation": operation.__name__}
        self.calls.append(record)

        async def acall(context):
            record["prompt"] = context
            text, requests = await source.acall(context)
            record["response"] = text
            if requests:
                raise ValueError("HMAW requires text responses, not tool requests")
            return text, requests

        try:
            return await operation(*args, provider=SimpleNamespace(acall=acall))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
