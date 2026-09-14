"""Coordinate isolated expert calls using Suzgun and Kalai's meta-prompting loop.

Adapted from suzgunmirac/meta-prompting (MIT); see README.md and LICENSE.
Copyright (c) 2023 Mirac Suzgun
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from slick import prompt
from slick.prompts import render
from slick.providers import Provider

EXPERT = re.compile(r'\b(Expert [^\r\n:"]+):\s*"""(.*?)"""', re.DOTALL)
FINAL = re.compile(r'>> FINAL ANSWER:\s*"""(.*?)"""', re.DOTALL)
CODE_BLOCK = re.compile(r"^```([^\r\n`]*)\r?\n(.*?)^```[ \t]*(?:\r?\n|$)", re.M | re.S)


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Result:
    answer: str | None
    rounds: int
    calls: int
    stop_reason: Literal["final_answer", "round_limit"]
    history: tuple[Message, ...]


class MetaPromptingScaffolding:
    """Own one sequential conductor and its fresh-context expert calls.

    Configure Slick's template root at startup and supply a stateless provider.
    Both roles use that provider; only the conductor sees the full history.
    Each run resets records. Do not run concurrently on the same instance.
    Optional execute_python(code) owns isolation, timeouts, and output limits.
    Provider/executor exceptions propagate without retries, retaining records.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        execute_python: Callable[[str], Awaitable[str]] | None = None,
    ):
        self.task = task
        self.provider = provider
        self.execute_python = execute_python
        self.system_template = "system.j2" if execute_python is None else "system_python.j2"
        self.history: list[Message] = []
        self.calls: list[dict] = []
        self.executions: list[dict] = []

    @prompt(template="conduct.j2")
    async def conduct(self, round: int, reminder: str, *, generated: str) -> str:
        """Choose the next expert instruction or delimit a final answer."""
        return generated

    @prompt(template="consult.j2")
    async def consult(self, name: str, instruction: str, *, generated: str) -> str:
        """Answer using only the conductor's delegated instruction."""
        return generated

    @prompt(template="program.j2")
    async def program(self, name: str, instruction: str, *, generated: str) -> str:
        """Generate Python and optionally request its execution."""
        return generated

    async def run(self, *, max_rounds: int = 15) -> Result:
        """Run Algorithm 1; exhaustion returns answer=None, never an invented answer.

        A round costs one conductor call and at most one expert call. Malformed
        replies consume a round. Experts take priority over final answers; only
        the first matching block of each type is used. Verification is prompted,
        not enforced or evaluated. Final answer whitespace is preserved.
        """
        self.history = [Message("user", render("initialize.j2", task=self.task))]
        self.calls, self.executions = [], []
        answer, rounds = None, 0
        for rounds in range(1, max_rounds + 1):
            reminder = render("last_round.j2") if rounds == max_rounds else ""
            output = await self._invoke(self.conduct, rounds, reminder)
            self.history.append(Message("assistant", output))
            expert = EXPERT.search(output)
            final = FINAL.search(output)
            if expert and expert[2].strip():
                await self._consult_expert(expert[1].strip(), expert[2].strip())
            elif final and final[1].strip():
                answer = final[1]
                break
            else:
                self.history.append(Message("user", render("error.j2")))
        return Result(
            answer,
            rounds,
            len(self.calls),
            "final_answer" if answer is not None else "round_limit",
            tuple(self.history),
        )

    async def _consult_expert(self, name: str, instruction: str):
        use_python = name == "Expert Python" and self.execute_python is not None
        operation = self.program if use_python else self.consult
        output = await self._invoke(operation, name, instruction)
        if use_python and "Please run this code!" in output:
            output = await self._execute(output)
        self.history.append(Message("user", render("feedback.j2", name=name, output=output)))

    async def _execute(self, output: str) -> str:
        # Match upstream's last fenced block before the explicit execution marker.
        blocks = [
            code
            for language, code in CODE_BLOCK.findall(output.split("Please run this code!", 1)[0])
            if language.strip() in ("", "python")
        ]
        if not blocks or not blocks[-1].strip():
            return output + "\n\n" + render("code_error.j2")
        code = blocks[-1].strip()
        record = {"code": code}
        self.executions.append(record)
        try:
            result = await self.execute_python(code)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        record["output"] = result
        return output + "\n\n" + render("execution.j2", code=code, output=result)

    async def _invoke(self, operation, *args):
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Retain raw responses before parsing; native tool requests are unsupported."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("Meta-prompting requires text, not native tool requests")
        return response, requests
