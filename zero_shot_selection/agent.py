"""Synthesize independent parent selectors, check them, then measure them without feedback."""

import ast
import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace

from pydantic import BaseModel, Field, ValidationError
from slick import prompt
from slick.providers import Provider


class CandidateRejected(ValueError):
    """The generated selector failed syntax, interface, or isolated functional checks."""


class Operator(BaseModel, extra="forbid", frozen=True):
    source: str = Field(min_length=1)


@dataclass
class Result:
    source: str
    attempt: int
    metrics: dict[str, float] | None = None


class ZeroShotSelection:
    """Own zero-shot sampling; callers own populations, execution isolation, and benchmarks.

    ``validate(source)`` must test executable behavior in caller-owned isolation,
    returning None on success or raising CandidateRejected on candidate failure.
    It must clean up workers on cancellation. ``evaluate(source)`` optionally runs
    the fixed search experiment and returns named finite measurements. Neither
    callback's results enter the prompt. Use a stateless provider, not a Session.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        validate: Callable[[str], Awaitable[None]],
        evaluate: Callable[[str], Awaitable[Mapping[str, float]]] | None = None,
        *,
        validation_timeout: float = 10.0,
    ):
        self.task = task
        self.provider = provider
        self.validate = validate
        self.evaluate = evaluate
        self.validation_timeout = validation_timeout
        self.attempts: list[dict] = []
        self.operators: list[Result] = []

    @prompt(template="synthesize.j2", output_type=Operator)
    async def synthesize(self, *, generated: Operator) -> Operator:
        """Parse and check the generated entry point without executing source code."""
        try:
            tree = ast.parse(generated.source)
            compile(tree, "<generated-selector>", "exec")
        except (SyntaxError, ValueError) as exc:
            raise CandidateRejected(f"invalid Python: {exc}") from exc
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "custom_selection"
        ]
        if len(functions) != 1:
            raise CandidateRejected("define one synchronous custom_selection function")
        function = functions[0]
        arguments = function.args
        names = [arg.arg for arg in arguments.posonlyargs + arguments.args]
        if (
            names != ["population", "k", "status"]
            or arguments.vararg
            or arguments.kwarg
            or arguments.kwonlyargs
            or function.decorator_list
        ):
            raise CandidateRejected("expected custom_selection(population, k=100, status={})")
        return generated

    async def run(self, count: int = 10, max_attempts: int = 100) -> list[Result]:
        """Collect valid operators up to a finite budget, then benchmark in sample order.

        Invalid generated output and functional timeouts consume attempts. Provider
        and infrastructure errors abort. Exhaustion raises with partial operators
        retained; no partial batch is benchmarked. Benchmark failures never trigger
        regeneration. Each run resets records; use one run at a time per instance.
        """
        self.attempts, self.operators = [], []
        await self._collect(count, max_attempts)
        await self._benchmark()
        return self.operators.copy()

    async def _collect(self, count: int, max_attempts: int) -> None:
        for _ in range(max_attempts):
            if len(self.operators) >= count:
                break
            record = {"attempt": len(self.attempts) + 1, "phase": "generate"}
            self.attempts.append(record)

            async def acall(context):
                record["prompt"] = context
                text, requests = await self.provider.acall(context)
                record["response"] = text
                if requests:
                    raise CandidateRejected("expected source code, not tool requests")
                return text, requests

            try:
                operator = await self.synthesize(provider=SimpleNamespace(acall=acall))
            except (ValidationError, CandidateRejected) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                continue
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            record.update(source=operator.source, phase="validate")
            try:
                await asyncio.wait_for(self.validate(operator.source), self.validation_timeout)
            except (CandidateRejected, asyncio.TimeoutError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                continue
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            record["phase"] = "accepted"
            self.operators.append(Result(operator.source, record["attempt"]))
        if len(self.operators) < count:
            raise RuntimeError(f"collected {len(self.operators)} of {count} valid operators")

    async def _benchmark(self) -> None:
        if self.evaluate is None:
            return
        for operator in self.operators:
            record = self.attempts[operator.attempt - 1]
            record["phase"] = "evaluate"
            try:
                metrics = dict(await self.evaluate(operator.source))
                if not metrics or not all(math.isfinite(value) for value in metrics.values()):
                    raise ValueError("evaluation must return nonempty, finite measurements")
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            operator.metrics = metrics
            record.update(phase="evaluated", metrics=metrics)
