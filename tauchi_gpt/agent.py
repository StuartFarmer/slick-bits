"""Run TAUCHI's retrieval, execution, reflection, task creation, and prioritization loop."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

from .memory import Chunk, Document, Embed, LocalMemory

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TaskID = Annotated[int, Field(strict=True, gt=0)]


class TaskList(BaseModel, extra="forbid", frozen=True):
    tasks: list[Text]


class TaskOrder(BaseModel, extra="forbid", frozen=True):
    task_ids: list[TaskID]


class Citation(BaseModel, extra="forbid", frozen=True):
    chunk_id: Text
    quote: Text


class Draft(BaseModel, extra="forbid", frozen=True):
    content: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class Task:
    id: int
    name: str


@dataclass(frozen=True)
class Reflection:
    draft: Draft
    critique: str


@dataclass(frozen=True)
class Step:
    task: Task
    output: Draft
    context: tuple[Chunk, ...]
    reflections: tuple[Reflection, ...]


@dataclass(frozen=True)
class Evaluation:
    goal_complete: bool
    feedback: str = ""


@dataclass(frozen=True)
class Result:
    steps: tuple[Step, ...]
    pending: tuple[Task, ...]
    evaluations: tuple[Evaluation, ...]
    stop_reason: Literal["completed", "exhausted", "budget"]
    calls: int

    @property
    def output(self) -> str:
        """The latest task artifact; all earlier artifacts remain in steps."""
        return self.steps[-1].output.content if self.steps else ""


Evaluate = Callable[[str, tuple[Step, ...]], Awaitable[Evaluation]]


def checked_draft(generated: Draft, context: Sequence[Chunk]) -> Draft:
    """Check passage provenance, without claiming semantic support for the answer."""
    if not generated.content.strip():
        raise ValueError("generated content is blank")
    sources = {chunk.id: chunk for chunk in context}
    for citation in generated.citations:
        if (
            citation.chunk_id not in sources
            or citation.quote not in sources[citation.chunk_id].text
        ):
            raise ValueError("citation must quote a retrieved passage exactly")
    return generated


class TauchiGPT:
    """Own one sequential task loop with explicit memory and stateless prompt calls.

    evaluate(goal, steps) checks overall completion and supplies planning feedback.
    It also owns any domain-specific execution and isolation. No tools, files,
    network clients, model choices, retries, or persistent sessions are created here.
    """

    def __init__(self, task: str, provider: Provider, embed: Embed, evaluate: Evaluate):
        self.task = task
        self.provider = provider
        self.embed = embed
        self.evaluate = evaluate
        self.pending: list[Task] = []
        self.steps: list[Step] = []
        self.evaluations: list[Evaluation] = []
        self.calls: list[dict] = []
        self.memory = LocalMemory(embed)
        self.seen: set[str] = set()
        self.next_id = 1

    @prompt(template="initialize.j2", output_type=TaskList)
    async def initialize(self, *, generated: TaskList) -> TaskList:
        """Decompose an arbitrary objective into actionable initial tasks."""
        return generated

    @prompt(template="execute.j2", output_type=Draft)
    async def execute(
        self, task: Task, context: Sequence[Chunk], recent: Sequence[Step], *, generated: Draft
    ) -> Draft:
        """Produce one task artifact with references to retrieved passages."""
        return checked_draft(generated, context)

    @prompt(template="reflect.j2")
    async def reflect(
        self, task: Task, context: Sequence[Chunk], output: Draft, *, generated: str
    ) -> str:
        """Critique the current artifact without asking for hidden reasoning traces."""
        if not generated.strip():
            raise ValueError("generated critique is blank")
        return generated

    @prompt(template="revise.j2", output_type=Draft)
    async def revise(
        self,
        task: Task,
        context: Sequence[Chunk],
        output: Draft,
        critique: str,
        *,
        generated: Draft,
    ) -> Draft:
        """Revise an artifact from explicit feedback and the same retrieved evidence."""
        return checked_draft(generated, context)

    @prompt(template="create_tasks.j2", output_type=TaskList)
    async def create_tasks(
        self, latest: Step, context: Sequence[Chunk], feedback: str, *, generated: TaskList
    ) -> TaskList:
        """Create follow-up tasks from the stored result and current objective."""
        return generated

    @prompt(template="prioritize.j2", output_type=TaskOrder)
    async def prioritize(self, *, generated: TaskOrder) -> list[Task]:
        """Accept only an exact permutation of the current stable task IDs."""
        tasks = {task.id: task for task in self.pending}
        if len(generated.task_ids) != len(tasks) or set(generated.task_ids) != set(tasks):
            raise ValueError("priority order must contain every pending task ID exactly once")
        return [tasks[task_id] for task_id in generated.task_ids]

    async def run(
        self,
        documents: Sequence[Document] = (),
        *,
        initial_tasks: Sequence[str] | None = None,
        max_steps: int = 10,
        reflection_cycles: int = 0,
        top_k: int = 5,
        chunk_size: int = 2000,
        overlap: int = 200,
    ) -> Result:
        """Reset run state and return completed, exhausted, or budget-limited work.

        Zero reflection cycles gives direct execution; four or five gives V1's
        fixed critique/revision cycles composed with V2's local retrieval loop.
        Exceptions propagate without retry; raw responses and partial state remain.
        Do not run the same instance concurrently.
        """
        self.pending, self.steps, self.evaluations, self.calls = [], [], [], []
        self.seen, self.next_id = set(), 1
        self.memory = LocalMemory(self.embed, chunk_size=chunk_size, overlap=overlap)
        if initial_tasks is not None:
            self._add_tasks(initial_tasks)
        if max_steps <= 0:
            return self._result("budget")
        for index, document in enumerate(documents):
            await self.memory.add(document, f"document:{index}", "document")
        if initial_tasks is None:
            self._add_tasks((await self._invoke(self.initialize)).tasks)
        for _ in range(max_steps):
            if not self.pending:
                return self._result("exhausted")
            step = await self._execute_task(reflection_cycles, top_k)
            evaluation = await self.evaluate(self.task, tuple(self.steps))
            self.evaluations.append(evaluation)
            await self.memory.add(
                Document(f"task:{step.task.id}", step.output.content),
                f"result:{step.task.id}",
                "result",
            )
            if evaluation.goal_complete:
                return self._result("completed")
            await self._replan(step, evaluation.feedback, top_k)
        return self._result("budget" if self.pending else "exhausted")

    async def _execute_task(self, cycles: int, top_k: int) -> Step:
        task = self.pending[0]
        context = await self.memory.retrieve(f"{self.task}\n{task.name}", top_k)
        output = await self._invoke(self.execute, task, context, tuple(self.steps[-3:]))
        reflections = []
        for _ in range(cycles):
            critique = await self._invoke(self.reflect, task, context, output)
            reflections.append(Reflection(output, critique))
            output = await self._invoke(self.revise, task, context, output, critique)
        step = Step(task, output, context, tuple(reflections))
        self.steps.append(step)
        self.pending.pop(0)
        return step

    async def _replan(self, latest: Step, feedback: str, top_k: int):
        context = await self.memory.retrieve(self.task, top_k)
        tasks = await self._invoke(self.create_tasks, latest, context, feedback)
        self._add_tasks(tasks.tasks)
        if len(self.pending) > 1:
            self.pending = await self._invoke(self.prioritize)

    def _add_tasks(self, names: Sequence[str]):
        for name in names:
            # ponytail: exact text deduplication; add semantic matching if paraphrases recur.
            key = " ".join(name.split()).casefold()
            if key not in self.seen:
                self.pending.append(Task(self.next_id, name))
                self.seen.add(key)
                self.next_id += 1

    def _result(self, reason) -> Result:
        return Result(
            tuple(self.steps), tuple(self.pending), tuple(self.evaluations), reason, len(self.calls)
        )

    async def _invoke(self, operation, *args):
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Retain the raw response before Slick parses or validates generated data."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("TAUCHI prompt operations do not accept tool requests")
        return response, requests
