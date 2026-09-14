"""Execute first, then recursively decompose failed tasks with ADaPT AND/OR plans.

Adapted from archiki/ADaPT's plan_and_run controllers; see LICENSE.upstream.
"""

import ast
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Step:
    action: str
    observation: str


@dataclass(frozen=True)
class Execution:
    task: str
    depth: int
    completed: bool
    steps: tuple[Step, ...]
    reason: Literal["completed", "failed", "step_limit"]


@dataclass(frozen=True)
class Result:
    completed: bool
    steps: tuple[Step, ...]
    executions: tuple[Execution, ...]
    calls: int


@dataclass(frozen=True)
class Plan:
    tasks: dict[int, str]
    logic: ast.expr


@dataclass(frozen=True)
class Decision:
    kind: Literal["action", "think", "completed", "failed"]
    content: str


def parse_plan(raw: str) -> Plan:
    """Parse the paper's numbered steps and boolean order without evaluating code.

    Every declared step must occur exactly once. AND binds tighter than OR;
    parentheses override precedence. Reject malformed plans rather than silently
    treating them as conjunctions, as the reference implementation does.
    """
    parts = re.split(r"(?im)^Execution Order:[ \t]*", raw)
    if len(parts) != 2:
        raise ValueError("plan requires exactly one Execution Order")
    tasks = {}
    for line in parts[0].splitlines():
        line = line.strip()
        if not line.startswith("Step"):
            continue
        match = re.fullmatch(r"Step ([1-9][0-9]*):[ \t]*(\S.*)", line)
        if match is None:
            raise ValueError(f"invalid plan step: {line!r}")
        index, task = int(match[1]), match[2]
        if index in tasks:
            raise ValueError(f"duplicate step {index}")
        tasks[index] = task
    expression = parts[1].strip()
    if not tasks or not re.fullmatch(r"(?:Step[ \t]+[1-9][0-9]*|AND|OR|[()\s])+", expression):
        raise ValueError("plan requires tasks and an AND/OR expression")
    expression = re.sub(r"Step[ \t]+([1-9][0-9]*)", r"s\1", expression)
    expression = expression.replace("AND", " and ").replace("OR", " or ")
    try:
        logic = ast.parse(f"({expression})", mode="eval").body
    except SyntaxError as exc:
        raise ValueError("invalid execution order") from exc
    references = []
    for node in ast.walk(logic):
        if isinstance(node, ast.Name):
            match = re.fullmatch(r"s([1-9][0-9]*)", node.id)
            if match is None:
                raise ValueError("invalid step reference")
            references.append(int(match[1]))
        elif not isinstance(node, (ast.BoolOp, ast.And, ast.Or, ast.Load)):
            raise ValueError("execution order only accepts steps, AND, OR, and parentheses")
    if set(references) != set(tasks) or len(references) != len(tasks):
        raise ValueError("execution order must reference every step exactly once")
    return Plan(tasks, logic)


class ADAPT:
    """Own sequential, independent Slick calls and successful action checkpoints.

    evaluate(history, action) restores the environment to history, applies one
    action, and returns its textual observation. Pure evaluators may instead
    compute from the supplied history. The caller owns isolation and restoration;
    replaying irreversible actions is not a valid implementation of this contract.
    Success is the executor's heuristic, not an external reward. Use one run at a
    time per instance. Generation/evaluation errors propagate without retry.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[tuple[Step, ...], str], Awaitable[str]],
        *,
        actions: str = "A textual reasoning step, environment command, or candidate answer",
        observation: str = "",
        executor_examples: str = "",
        planner_examples: str = "",
        planner_provider: Provider | None = None,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.planner_provider = provider if planner_provider is None else planner_provider
        self.actions, self.observation = actions, observation
        self.executor_examples, self.planner_examples = executor_examples, planner_examples
        self.calls: list[dict] = []
        self.evaluations: list[dict] = []
        self.executions: list[Execution] = []

    @prompt(template="execute.j2")
    async def next_action(
        self, task: str, checkpoint: tuple[Step, ...], history: list[str], *, generated: str
    ) -> Decision:
        """Interpret one textual ReAct turn; only explicit thought markers finish."""
        text = generated.strip().removeprefix("> ")
        kind, separator, content = text.partition(":")
        if separator and kind.lower() in {"think", "action"}:
            kind, content = kind.lower(), content.strip()
        else:
            kind, content = "action", text
        if not content.strip():
            raise ValueError("empty executor response")
        if kind == "think":
            marker = re.fullmatch(r"Task (completed|failed)[!.]?", content, flags=re.IGNORECASE)
            if marker:
                kind = marker[1].lower()
        return Decision(kind, content)

    @prompt(template="plan.j2")
    async def plan(
        self,
        task: str,
        checkpoint: tuple[Step, ...],
        failure: str,
        last_observation: str,
        *,
        generated: str,
    ) -> Plan:
        """Decompose only a failed task, retaining the paper's textual plan format."""
        return parse_plan(generated)

    async def run(
        self, *, max_depth: int = 3, max_executor_steps: int = 20, max_calls: int = 256
    ) -> Result:
        """Run ADaPT with root depth one; max_depth=1 runs only the executor.

        Executor turns include thoughts and completion markers. All planner and
        executor generations share max_calls; exhaustion raises RuntimeError.
        A run resets records, but never resets the caller's environment. Returned
        steps are the accepted checkpoint, including partial progress on failure.
        """
        self.max_depth, self.max_executor_steps, self.max_calls = (
            max_depth,
            max_executor_steps,
            max_calls,
        )
        self.calls, self.evaluations, self.executions = [], [], []
        completed, steps = await self._adapt(self.task, 1, ())
        return Result(completed, steps, tuple(self.executions), len(self.calls))

    async def _adapt(self, task: str, depth: int, checkpoint: tuple[Step, ...]):
        if depth > self.max_depth:
            return False, checkpoint
        execution = await self._execute(task, depth, checkpoint)
        if execution.completed:
            return True, checkpoint + execution.steps
        if depth >= self.max_depth:
            return False, checkpoint
        last_observation = (
            execution.steps[-1].observation
            if execution.steps
            else checkpoint[-1].observation
            if checkpoint
            else self.observation
        )
        plan = await self._invoke(self.plan, task, checkpoint, execution.reason, last_observation)
        return await self._compose(plan.logic, plan.tasks, depth + 1, checkpoint)

    async def _compose(self, node, tasks, depth, checkpoint):
        if isinstance(node, ast.Name):
            return await self._adapt(tasks[int(node.id[1:])], depth, checkpoint)
        conjunction = isinstance(node.op, ast.And)
        current = checkpoint
        for child in node.values:
            completed, steps = await self._compose(child, tasks, depth, current)
            if conjunction:
                current = steps
                if not completed:
                    return False, current
            elif completed:
                return True, steps
            # OR alternatives retain the original checkpoint even after partial success.
        return conjunction, current

    async def _execute(self, task, depth, checkpoint) -> Execution:
        # ponytail: full checkpoint history grows with the budget; summarize if context limits matter.
        history, steps = [], ()
        reason = "step_limit"
        for _ in range(self.max_executor_steps):
            decision = await self._invoke(self.next_action, task, checkpoint, history)
            if decision.kind in {"completed", "failed"}:
                reason = decision.kind
                break
            if decision.kind == "think":
                history.append(f"Think: {decision.content}")
                continue
            observation = await self._evaluate(checkpoint + steps, decision.content)
            steps += (Step(decision.content, observation),)
            history.append(f"Action: {decision.content}\nObservation: {observation}")
        execution = Execution(task, depth, reason == "completed", steps, reason)
        self.executions.append(execution)
        return execution

    async def _evaluate(self, history, action):
        record = {"history": history, "action": action}
        self.evaluations.append(record)
        try:
            observation = await self.evaluate(history, action)
            record["observation"] = observation
            return observation
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _invoke(self, operation, *args):
        if len(self.calls) >= self.max_calls:
            raise RuntimeError(f"ADaPT call budget exhausted ({self.max_calls})")
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Capture raw responses before Slick postprocessing can reject them."""
        record = self.calls[-1]
        provider = self.planner_provider if record["operation"] == "plan" else self.provider
        record["prompt"] = context
        response, requests = await provider.acall(context, tools=tools, tool_results=tool_results)
        record["response"] = response
        if requests:
            raise ValueError("ADaPT uses textual actions, not provider tool requests")
        return response, requests
