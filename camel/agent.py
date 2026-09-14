"""Solve arbitrary tasks with CAMEL's cooperative user/assistant role-playing loop.

Adapted from camel-ai/camel v0.1.0 (Apache-2.0); see NOTICE and LICENSE.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Speaker = Literal["user", "assistant"]
StopReason = Literal[
    "task_done", "user_no_instruct", "assistant_instruct", "message_limit", "token_limit"
]
DONE = "<CAMEL_TASK_DONE>"
INSTRUCTION = re.compile(r"^[ \t]*Instruction:[ \t]*\S", re.MULTILINE)


class TokenLimitError(Exception):
    """A provider adapter can raise this for context overflow or output truncation."""


class Choice(BaseModel, extra="forbid", frozen=True):
    option: int = Field(ge=1, strict=True)
    explanation: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@dataclass(frozen=True)
class Message:
    speaker: Speaker
    content: str


@dataclass(frozen=True)
class Selection:
    speaker: Speaker
    proposals: tuple[str, ...]
    choice: Choice


@dataclass(frozen=True)
class Result:
    task: str
    messages: tuple[Message, ...]
    stop_reason: StopReason
    solution: str | None
    selections: tuple[Selection, ...]
    calls: int


Reviewer = Callable[[Speaker, tuple[str, ...], tuple[Message, ...]], Awaitable[Choice]]


class CAMEL:
    """Own a sequential role-playing conversation and optional critic selection.

    Supply stateless Slick providers. History is rendered explicitly so both
    sides retain every selected instruction/solution and no rejected branch.
    Providers own decoding settings and transport retries. Runs reset state;
    use one run at a time per instance. Generated content is never executed.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        assistant_role: str,
        user_role: str,
        user_provider: Provider | None = None,
        specifier_provider: Provider | None = None,
        critic: Provider | None = None,
        review: Reviewer | None = None,
        critic_role: str = "Critic",
        criteria: str = "improving the task performance",
        candidates: int = 3,
        count_tokens: Callable[[str], int] | None = None,
        token_limit: int | None = None,
    ):
        self.idea = task
        self.task = task
        self.assistant_role = assistant_role
        self.user_role = user_role
        self.critic_role = critic_role
        self.criteria = criteria
        self.providers = {
            "assistant": provider,
            "user": provider if user_provider is None else user_provider,
            "specifier": provider if specifier_provider is None else specifier_provider,
            "critic": critic,
        }
        self.review = review
        self.candidates = candidates if critic is not None or review is not None else 1
        self.count_tokens = count_tokens
        self.token_limit = token_limit
        self.messages: list[Message] = []
        self.selections: list[Selection] = []
        self.calls: list[dict] = []
        self.reviews: list[dict] = []
        self.stop_reason: StopReason | None = None

    @prompt(template="specify.j2")
    async def specify(self, word_limit: int, *, generated: str) -> str:
        """Specify the idea once; the upstream word limit is a prompt constraint."""
        if not generated.strip():
            raise ValueError("specified task is blank")
        return generated.strip()

    @prompt(template="instruct.j2")
    async def instruct(self, *, generated: str) -> str:
        """Generate one instruction or the task-done token as unparsed text."""
        return generated

    @prompt(template="solve.j2")
    async def solve(self, *, generated: str) -> str:
        """Generate the assistant's solution, preserving the paper's delimiters."""
        return generated

    @prompt(template="select.j2", output_type=Choice)
    async def select(
        self, speaker: Speaker, proposals: tuple[str, ...], *, generated: Choice
    ) -> Choice:
        """Select an existing proposal; invalid choices fail without repair calls."""
        if generated.option > len(proposals):
            raise ValueError("critic option is outside the proposals")
        return generated

    @prompt(template="extract.j2")
    async def extract(self, *, generated: str) -> str:
        """Extract the conversation's full solution using Appendix H's prompt."""
        if not generated.strip():
            raise ValueError("extracted solution is blank")
        return generated

    async def run(
        self,
        *,
        specify_task: bool = True,
        word_limit: int = 50,
        max_messages: int = 40,
        extract: bool = False,
    ) -> Result:
        """Specify, alternate roles, and optionally extract a full solution.

        max_messages counts selected user AND assistant messages, including a
        terminating message; specification, alternatives, and critic calls are
        additional. A task-done token is an agent judgment, not evaluated success.
        Token exhaustion returns partial work. Other failures propagate with
        raw records retained; extraction errors also propagate.
        """
        self.task = self.idea
        self.messages, self.selections, self.calls, self.reviews = [], [], [], []
        self.stop_reason = None
        try:
            if specify_task:
                self.task = await self._invoke(self.specify, "specifier", word_limit)
            self.stop_reason = await self._converse(max_messages)
        except TokenLimitError:
            self.stop_reason = "token_limit"
        solution = None
        if extract and any(message.speaker == "assistant" for message in self.messages):
            solution = await self._invoke(self.extract, "assistant")
        return Result(
            self.task,
            tuple(self.messages),
            self.stop_reason,
            solution,
            tuple(self.selections),
            sum(record["attempted"] for record in self.calls),
        )

    async def _converse(self, max_messages: int) -> StopReason:
        missed_instructions = 0
        while len(self.messages) < max_messages:
            instruction = await self._propose("user")
            self.messages.append(Message("user", instruction))
            if instruction.strip() == DONE:
                return "task_done"
            # ponytail: delimiter heuristic; semantic role detection needs a separate evaluator.
            missed_instructions = 0 if INSTRUCTION.search(instruction) else missed_instructions + 1
            if missed_instructions >= 3:
                return "user_no_instruct"
            if len(self.messages) >= max_messages:
                break
            solution = await self._propose("assistant")
            self.messages.append(Message("assistant", solution))
            if INSTRUCTION.search(solution):
                return "assistant_instruct"
        return "message_limit"

    async def _propose(self, speaker: Speaker) -> str:
        operation = self.instruct if speaker == "user" else self.solve
        proposals = []
        for _ in range(self.candidates):
            proposals.append(await self._invoke(operation, speaker))
        options = tuple(proposals)
        if len(options) == 1:
            return options[0]
        if self.review is None:
            choice = await self._invoke(self.select, "critic", speaker, options)
        else:
            record = {"speaker": speaker, "proposals": options}
            self.reviews.append(record)
            try:
                choice = await self.review(speaker, options, tuple(self.messages))
                record["choice"] = choice
                choice = Choice.model_validate(choice.model_dump())
                if choice.option > len(options):
                    raise ValueError("critic option is outside the proposals")
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
        self.selections.append(Selection(speaker, options, choice))
        return options[choice.option - 1]

    async def _invoke(self, operation, role: str, *args):
        record = {"operation": operation.__name__, "role": role, "attempted": False}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Retain raw responses before Slick parsing and check the exact prompt budget."""
        record = self.calls[-1]
        record["prompt"] = context
        if self.token_limit is not None:
            record["tokens"] = self.count_tokens(context)
            if record["tokens"] >= self.token_limit:
                raise TokenLimitError("rendered prompt reached token_limit")
        record["attempted"] = True
        response, requests = await self.providers[record["role"]].acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("CAMEL role-playing expects text, not native tool requests")
        return response, requests
