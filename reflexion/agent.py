"""Retry arbitrary tasks using evaluated trajectories and bounded verbal memory."""

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Trajectory:
    output: str
    trace: str = ""


@dataclass(frozen=True)
class Evaluation:
    passed: bool
    feedback: str = ""
    reward: float | None = None


@dataclass(frozen=True)
class Trial:
    trajectory: Trajectory
    evaluation: Evaluation


@dataclass(frozen=True)
class Result:
    trajectory: Trajectory | None
    trials: tuple[Trial, ...]
    memory: tuple[str, ...]
    stop_reason: Literal["success", "budget"]
    calls: int


Actor = Callable[[str, str, tuple[str, ...]], Awaitable[Trajectory]]
Evaluator = Callable[[Trajectory], Awaitable[Evaluation]]


def nonblank(generated: str) -> str:
    """Check model text without altering whitespace in candidate artifacts."""
    if not generated.strip():
        raise ValueError("generated response is blank")
    return generated


class Reflexion:
    """Own the actor/evaluator/reflection loop for one task at a time.

    The optional actor(task, input, memory) produces a complete episode; it owns
    environment reset, action limits, tools, and short-term trajectory history.
    The async evaluator supplies success independently from its optional reward.
    Providers must not carry implicit history between prompt operations.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Evaluator,
        *,
        actor: Actor | None = None,
        reflection_provider: Provider | None = None,
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.actor = actor
        self.reflection_provider = provider if reflection_provider is None else reflection_provider
        self.trajectory: Trajectory | None = None
        self.trials: list[Trial] = []
        self.memory: deque[str] = deque(maxlen=3)
        self.calls: list[dict] = []

    @prompt(template="generate.j2")
    async def generate(self, input: str, memory: tuple[str, ...], *, generated: str) -> str:
        """Produce the initial text trajectory."""
        return nonblank(generated)

    @prompt(template="retry.j2")
    async def retry(self, input: str, memory: tuple[str, ...], *, generated: str) -> str:
        """Attempt the task again using distilled experience."""
        return nonblank(generated)

    @prompt(template="reflect.j2")
    async def reflect(
        self,
        input: str,
        trajectory: Trajectory,
        evaluation: Evaluation,
        memory: tuple[str, ...],
        *,
        generated: str,
    ) -> str:
        """Distill a failed trajectory and feedback into a plan for the next trial."""
        return nonblank(generated)

    async def run(self, input: str, *, max_trials: int = 4, memory_size: int = 3) -> Result:
        """Return the last evaluated trajectory, stopping on success or the trial cap.

        The initial trial counts toward max_trials. Zero makes no calls. Reflect
        only after failures with another trial available. Zero memory_size discards
        reflections. Runs reset state; use one run at a time per instance.
        Errors propagate without retries, preserving partial state and raw model
        call records. Calls counts only model requests made by this agent, excluding
        work inside callbacks and provider transport retries.
        """
        self.trajectory = None
        self.trials = []
        self.memory = deque(maxlen=memory_size)
        self.calls = []
        reason = "budget"
        for trial_index in range(max_trials):
            self.trajectory = await self.act(input, trial_index)
            evaluation = await self.evaluate(self.trajectory)
            self.trials.append(Trial(self.trajectory, evaluation))
            if evaluation.passed:
                reason = "success"
                break
            if trial_index + 1 < max_trials:
                reflection = await self._invoke(
                    self.reflect, input, self.trajectory, evaluation, tuple(self.memory)
                )
                self.memory.append(reflection)
        return Result(
            self.trajectory, tuple(self.trials), tuple(self.memory), reason, len(self.calls)
        )

    async def act(self, input: str, trial_index: int) -> Trajectory:
        """Start an independent text generation or caller-owned environment episode."""
        memory = tuple(self.memory)
        if self.actor is not None:
            return await self.actor(self.task, input, memory)
        operation = self.generate if trial_index == 0 else self.retry
        return Trajectory(await self._invoke(operation, input, memory))

    async def _invoke(self, operation, *args) -> str:
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Save raw model text before Slick postprocessing can reject it."""
        record = self.calls[-1]
        provider = self.reflection_provider if record["operation"] == "reflect" else self.provider
        record["prompt"] = context
        response, requests = await provider.acall(context, tools=tools, tool_results=tool_results)
        record["response"] = response
        if requests:
            raise ValueError("Reflexion prompt operations require text, not tool requests")
        return response, requests
