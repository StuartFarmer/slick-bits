"""Formulate optimization models, generate solver code, and repair failed executions/tests."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from slick import prompt
from slick.providers import Provider


class Artifact(BaseModel, extra="forbid"):
    text: str = Field(min_length=1)


@dataclass(frozen=True)
class Execution:
    status: Literal["passed", "execution_error", "test_failure", "invalid_test"]
    output: Any = None
    feedback: str = ""


class OptiMUS:
    """Use a structured problem description; actual instance data stays in the executor."""

    def __init__(
        self,
        provider: Provider,
        execute: Callable[[str, str, Any], Awaitable[Execution]],
        *,
        interface: str,
    ):
        self.provider, self.execute, self.interface = provider, execute, interface

    @prompt(template="formulate.j2", output_type=Artifact)
    async def formulate(self, problem: str, *, generated: Artifact) -> str:
        return generated.text

    @prompt(template="code.j2", output_type=Artifact)
    async def code(self, problem: str, formulation: str, *, generated: Artifact) -> str:
        return generated.text

    @prompt(template="tests.j2", output_type=Artifact)
    async def tests(self, problem: str, *, generated: Artifact) -> str:
        return generated.text

    @prompt(template="fix_execution.j2", output_type=Artifact)
    async def fix_execution(
        self, problem: str, formulation: str, code: str, feedback: str, *, generated: Artifact
    ) -> str:
        return generated.text

    @prompt(template="fix_logic.j2", output_type=Artifact)
    async def fix_logic(
        self, problem: str, formulation: str, code: str, feedback: str, *, generated: Artifact
    ) -> str:
        return generated.text

    @prompt(template="rephrase.j2", output_type=Artifact)
    async def rephrase(self, problem: str, *, generated: Artifact) -> str:
        return generated.text

    async def solve(self, problem, data, test_code, repairs):
        formulation = await self.formulate(problem, provider=self.provider)
        code = await self.code(problem, formulation, provider=self.provider)
        tests = (
            test_code
            if test_code is not None
            else await self.tests(problem, provider=self.provider)
        )
        attempts = []
        for iteration in range(repairs + 1):
            result = await self.execute(code, tests, data)
            if result.status not in {"passed", "execution_error", "test_failure", "invalid_test"}:
                raise ValueError("executor returned an unknown status")
            attempts.append({"code": code, "execution": result})
            if result.status in {"passed", "invalid_test"} or iteration == repairs:
                break
            repair = self.fix_execution if result.status == "execution_error" else self.fix_logic
            code = await repair(problem, formulation, code, result.feedback, provider=self.provider)
        return {
            "problem": problem,
            "formulation": formulation,
            "tests": tests,
            "code": code,
            "execution": result,
            "attempts": attempts,
        }

    async def run(
        self,
        problem: str,
        data: Any,
        *,
        test_code: str | None = None,
        repairs: int = 3,
        augmentations: int = 0,
    ) -> dict:
        attempts = []
        current = problem
        for index in range(augmentations + 1):
            if index:
                current = await self.rephrase(current, provider=self.provider)
            result = await self.solve(current, data, test_code, repairs)
            attempts.append(result)
            if result["execution"].status in {"passed", "invalid_test"}:
                break
        return {
            "result": result,
            "variants": attempts,
            "execution_calls": sum(len(v["attempts"]) for v in attempts),
        }
