"""Execute prompting programs with modular handlers and frame-local answer references."""

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from pydantic import JsonValue
from slick import prompt
from slick.providers import Provider

Handler = Callable[[str], Awaitable[JsonValue]]
REFERENCE = re.compile(r'"#([0-9]+)"|#([0-9]+)')
OPERATORS = {
    "select": "select",
    "foreach": "foreach",
    "project_values": "foreach",
    "foreach_merge": "foreach_merge",
    "project_values_flat.unique": "foreach_merge",
    "project_values_flat_unique": "foreach_merge",
}


@dataclass(frozen=True)
class Program:
    """Few-shot examples for one decomposer, including recursive programs."""

    examples: str = ""


@dataclass(frozen=True)
class Step:
    handler: str
    question: str = ""
    operator: str = "select"


@dataclass(frozen=True)
class Execution:
    program: str
    depth: int
    index: int
    step: Step
    questions: tuple[str, ...]
    answer: JsonValue


@dataclass(frozen=True)
class Result:
    answer: JsonValue
    steps: tuple[Execution, ...]
    calls: int


def parse_step(raw: str) -> Step:
    """Accept one paper-style instruction; never accept a model-invented answer."""
    text = re.sub(r"^(?:QS|Q[0-9]+):\s*", "", raw.strip())
    if text == "[EOQ]":
        return Step("EOQ")
    match = re.fullmatch(
        r"(?:\(([^)]+)\)\s*|\[(foreach|foreach_merge)\]\s*)?"
        r"\[([A-Za-z_][A-Za-z_0-9-]*)\]\s+(.+)",
        text,
        flags=re.DOTALL,
    )
    if match is None or re.search(
        r"\n\s*(?:A:|#[0-9]+:|QS:|Q[0-9]+:|\[[A-Za-z_][A-Za-z_0-9-]*\]|\([a-z_.]+\))",
        text,
    ):
        raise ValueError(f"expected one handler question or [EOQ]; response={raw!r}")
    operator, alias, handler, question = match.groups()
    operation = operator or alias or "select"
    if operation not in OPERATORS or handler == "EOQ" or not question.strip():
        raise ValueError(f"invalid instruction; response={raw!r}")
    return Step(handler, question.strip(), OPERATORS[operation])


def substitute(question: str, answers: list[JsonValue], overrides: dict | None = None) -> str:
    """Substitute complete reference tokens once, preserving JSON quoting."""

    def replace(match):
        index = int(match.group(1) or match.group(2))
        if not 1 <= index <= len(answers):
            raise ValueError(f"unknown answer reference #{index}")
        value = (overrides or {}).get(index, answers[index - 1])
        if match.group(1) or not isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        return value

    return REFERENCE.sub(replace, question)


class Decomp:
    """Generate one sub-question, execute it, and feed its answer back until EOQ.

    Callbacks receive resolved text and return JSON-compatible values. Programs
    and callbacks share a namespace; a callback overrides a program of the same
    name. Failures propagate without retries. Partial steps, raw decompositions,
    and the consumed call budget remain available on the instance after failure.
    Use one run at a time per instance; every run resets its execution records.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        handlers: Mapping[str, Handler],
        programs: Mapping[str, Program] | None = None,
    ):
        self.task = task
        self.provider = provider
        self.handlers = dict(handlers)
        self.programs = {"decomp": Program(), **(programs or {})}
        self.steps: list[Execution] = []
        self.generations: list[str] = []
        self.calls = 0

    @prompt(template="decompose.j2")
    async def decompose(
        self, question: str, examples: str, history: str, *, generated: str
    ) -> Step:
        """Record raw generation before checking its textual instruction."""
        self.generations.append(generated)
        step = parse_step(generated)
        if step.handler != "EOQ" and step.handler not in self.handlers | self.programs:
            raise ValueError(f"unknown handler {step.handler!r}; response={generated!r}")
        return step

    def _consume_call(self):
        if self.calls >= self.max_calls:
            raise RuntimeError(f"DECOMP call budget exhausted ({self.max_calls})")
        self.calls += 1

    async def _dispatch(self, handler: str, question: str, depth: int) -> JsonValue:
        if handler in self.handlers:
            self._consume_call()
            return await self.handlers[handler](question)
        return await self._execute(question, handler, depth + 1)

    async def _answer(self, step: Step, answers: list[JsonValue], depth: int):
        if step.operator == "select":
            question = substitute(step.question, answers)
            return (question,), await self._dispatch(step.handler, question, depth)
        indices = list(dict.fromkeys(int(a or b) for a, b in REFERENCE.findall(step.question)))
        # Validate all references before invoking any handler.
        substitute(step.question, answers)
        lists = [index for index in indices if isinstance(answers[index - 1], list)]
        if len(lists) != 1:
            raise ValueError("foreach requires exactly one referenced list")
        index = lists[0]
        questions, results = [], []
        for item in answers[index - 1]:
            question = substitute(step.question, answers, {index: item})
            questions.append(question)
            results.append(await self._dispatch(step.handler, question, depth))
        if step.operator == "foreach_merge":
            merged, seen = [], set()
            for result in results:
                if not isinstance(result, list):
                    raise ValueError("foreach_merge handlers must return lists")
                for item in result:
                    key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)
            results = merged
        return tuple(questions), results

    async def _execute(self, question: str, program: str, depth: int) -> JsonValue:
        if depth > self.max_depth:
            raise RuntimeError(f"DECOMP recursion depth exceeded ({self.max_depth})")
        examples = self.programs[program].examples
        answers, history = [], []
        while True:
            self._consume_call()
            step = await self.decompose(
                question, examples, "\n".join(history), provider=self.provider
            )
            if step.handler == "EOQ":
                if not answers:
                    raise ValueError("[EOQ] requires a previous answer")
                return answers[-1]
            questions, answer = await self._answer(step, answers, depth)
            serialized = json.dumps(answer, ensure_ascii=False, allow_nan=False)
            answers.append(answer)
            index = len(answers)
            self.steps.append(Execution(program, depth, index, step, questions, answer))
            history.append(
                f"Q{index}: ({step.operator}) [{step.handler}] {step.question}\n"
                f"#{index}: {serialized}"
            )

    async def run(
        self, question: str, *, program: str = "decomp", max_calls: int = 256, max_depth: int = 16
    ) -> Result:
        """Run with local reference scopes and a shared budget across the entire tree.

        Each decomposer generation (including EOQ) and leaf dispatch consumes
        one call, including failed attempts. Callbacks own their internal costs.
        The root has depth zero; max_depth=0 disables nested decomposition.
        """
        self.calls = 0
        self.steps = []
        self.generations = []
        self.max_calls, self.max_depth = max_calls, max_depth
        answer = await self._execute(question, program, 0)
        return Result(answer, tuple(self.steps), self.calls)
