"""Score reasoning methods, select the best, and execute it on the original input."""

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from pydantic import BaseModel, Field
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Method:
    name: str
    description: str
    execute: Callable[[str], Awaitable[str]] | None = None


class Suitability(BaseModel, extra="forbid"):
    score: int = Field(strict=True, ge=1, le=7)


class Vote(BaseModel, extra="forbid"):
    index: int = Field(strict=True, ge=0)


@dataclass(frozen=True)
class Assessment:
    method: str
    score: int


@dataclass(frozen=True)
class Result:
    output: str
    method: str
    assessments: tuple[Assessment, ...]
    calls: int


def nonblank(generated: str) -> str:
    if not generated.strip():
        raise ValueError("generated response is blank")
    return generated


class MRP:
    """Own Algorithm 1's per-method scoring and Appendix A.2 execution flows.

    A custom Method either applies its description as a prompt or delegates to an
    async execute(input) callback. Callbacks own their context and infrastructure.
    Configure Slick's template root at application startup, outside this class.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        methods: Sequence[Method] | None = None,
        proposals: int = 3,
    ):
        self.task = task
        self.provider = provider
        self.proposals = proposals
        self.methods = self._default_methods() if methods is None else tuple(methods)
        self.assessments: list[Assessment] = []
        self.selected: Method | None = None
        self.output: str | None = None
        self.calls: list[dict] = []
        self.execution_error: str | None = None

    def _default_methods(self) -> tuple[Method, ...]:
        executors = {
            "cot": partial(self._invoke, self.chain_of_thought),
            "tot": self.tree_of_thoughts,
            "analogical": partial(self._invoke, self.analogical),
            "self_refine": self.self_refine,
            "spp": partial(self._invoke, self.collaborate),
            "step_back": self.step_back,
            "simtom": self.simtom,
        }
        pool = json.loads(Path(__file__).with_name("methods.json").read_text())
        return tuple(Method(m["name"], m["description"], executors[m["id"]]) for m in pool)

    @prompt(template="assess_method.j2", output_type=Suitability)
    async def assess_method(self, input: str, method: Method, *, generated: Suitability) -> int:
        return generated.score

    @prompt(template="apply.j2")
    async def apply(self, input: str, method: Method, *, generated: str) -> str:
        return nonblank(generated)

    @prompt(template="chain_of_thought.j2")
    async def chain_of_thought(self, input: str, *, generated: str) -> str:
        return nonblank(generated)

    @prompt(template="propose.j2")
    async def propose(self, input: str, *, generated: str) -> str:
        return nonblank(generated)

    @prompt(template="vote.j2", output_type=Vote)
    async def vote(self, input: str, choices: Sequence[str], *, generated: Vote) -> int:
        if generated.index >= len(choices):
            raise ValueError("vote index is outside the supplied choices")
        return generated.index

    @prompt(template="analogical.j2")
    async def analogical(self, input: str, *, generated: str) -> str:
        return nonblank(generated)

    @prompt(template="draft.j2")
    async def draft(self, input: str, *, generated: str) -> str:
        return nonblank(generated)

    @prompt(template="revise.j2")
    async def revise(self, input: str, draft: str, *, generated: str) -> str:
        return nonblank(generated)

    @prompt(template="collaborate.j2")
    async def collaborate(self, input: str, *, generated: str) -> str:
        return nonblank(generated)

    @prompt(template="abstract.j2")
    async def abstract(self, input: str, *, generated: str) -> str:
        return nonblank(generated)

    @prompt(template="solve_with_principles.j2")
    async def solve_with_principles(self, input: str, principles: str, *, generated: str) -> str:
        return nonblank(generated)

    @prompt(template="perspective.j2")
    async def perspective(self, input: str, *, generated: str) -> str:
        return nonblank(generated)

    @prompt(template="solve_with_facts.j2")
    async def solve_with_facts(self, input: str, facts: str, *, generated: str) -> str:
        return nonblank(generated)

    async def tree_of_thoughts(self, input: str) -> str:
        # MRP Figure 8 uses whole-answer proposals followed by a vote.
        choices = [await self._invoke(self.propose, input) for _ in range(self.proposals)]
        index = await self._invoke(self.vote, input, choices)
        return choices[index]

    async def self_refine(self, input: str) -> str:
        draft = await self._invoke(self.draft, input)
        return await self._invoke(self.revise, input, draft)

    async def step_back(self, input: str) -> str:
        principles = await self._invoke(self.abstract, input)
        return await self._invoke(self.solve_with_principles, input, principles)

    async def simtom(self, input: str) -> str:
        facts = await self._invoke(self.perspective, input)
        return await self._invoke(self.solve_with_facts, input, facts)

    async def select(self, input: str) -> Method:
        for method in self.methods:
            score = await self._invoke(self.assess_method, input, method)
            self.assessments.append(Assessment(method.name, score))
        index = max(range(len(self.methods)), key=lambda i: self.assessments[i].score)
        return self.methods[index]

    async def execute(self, input: str, method: Method) -> str:
        try:
            if method.execute is None:
                return await self._invoke(self.apply, input, method)
            return nonblank(await method.execute(input))
        except Exception as exc:
            self.execution_error = f"{type(exc).__name__}: {exc}"
            raise

    async def run(self, input: str) -> Result:
        """Score every method once, execute the first maximum, and return raw text.

        Runs reset records; use one run at a time per instance. Failures propagate
        with partial assessments and raw calls retained. No retries, repair, or
        fallback method is implicit. Callback-internal calls are not counted.
        """
        self.assessments = []
        self.selected = None
        self.output = None
        self.calls = []
        self.execution_error = None
        self.selected = await self.select(input)
        self.output = await self.execute(input, self.selected)
        return Result(self.output, self.selected.name, tuple(self.assessments), len(self.calls))

    async def _invoke(self, operation, *args):
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Retain provider text before Slick parsing and generated-output checks."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("MRP prompt operations require text, not tool requests")
        return response, requests
