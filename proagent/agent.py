"""Plan cooperative skills, check them, and revise beliefs from observed behavior.

Adapted from PKU-Alignment/ProAgent's planner, recent-K memory, and action loop.
See NOTICE for the pinned official source and the task-independent adaptations.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, JsonValue, StringConstraints, ValidationError
from slick import prompt
from slick.providers import Provider

State = TypeVar("State")
Action = TypeVar("Action")
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str


class SkillCall(BaseModel, extra="forbid", frozen=True):
    name: Text
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class Plan(BaseModel, extra="forbid", frozen=True):
    analysis: Text
    intentions: dict[str, Text]
    skill: SkillCall


class InvalidPlan(ValueError):
    """A generated plan names an unavailable skill or the wrong teammates."""


class SkillFailure(Exception):
    """Controller cannot produce an action; no environment action was executed."""


@dataclass(frozen=True)
class Behavior:
    """Observed teammate action at an environment step, not an inferred intention."""

    step: int
    teammate: str
    action: str


@dataclass(frozen=True)
class Observation:
    step: int
    description: str
    behaviors: tuple[Behavior, ...] = ()
    terminal: bool = False


@dataclass(frozen=True)
class Verification:
    status: Literal["ready", "complete", "invalid"]
    feedback: str = ""


@dataclass
class MemoryEntry:
    step: int
    state: str
    plan: Plan
    status: str = "planned"
    feedback: str = ""
    actions: int = 0


@dataclass(frozen=True)
class Decision(Generic[Action]):
    status: Literal["action", "terminal", "exhausted"]
    action: Action | None = None
    skill: SkillCall | None = None


@dataclass(frozen=True)
class Result(Generic[State, Action]):
    state: State
    decisions: tuple[Decision[Action], ...]
    stop_reason: Literal["terminal", "exhausted", "budget"]


class ProAgent(Generic[State, Action]):
    """Own one decentralized agent's episode and independent Slick prompt calls.

    Grounding, authoritative verification, low-level control, and stepping belong
    to the caller. A controller returns one action without executing it. The
    verifier checks completion and preconditions against each fresh state.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        skills: Sequence[Skill],
        teammates: Sequence[str],
        ground: Callable[[State], Observation],
        verify: Callable[[State, SkillCall], Awaitable[Verification]],
        control: Callable[[State, SkillCall], Awaitable[Action]],
        step: Callable[[State, Action], Awaitable[State]],
        *,
        role: str = "agent",
        knowledge: str = "",
        examples: str = "",
        recent_k: int = 1,
        belief_revision: bool = True,
        max_plan_attempts: int = 5,
        verification_rounds: Literal[1, 3] = 1,
    ):
        self.task, self.provider, self.role = task, provider, role
        self.skills, self.teammates = tuple(skills), tuple(teammates)
        self.knowledge, self.examples = knowledge, examples
        self.ground, self.verify, self.control, self.step = ground, verify, control, step
        self.recent_k, self.belief_revision = recent_k, belief_revision
        self.max_plan_attempts, self.verification_rounds = max_plan_attempts, verification_rounds
        self.memory: list[MemoryEntry] = []
        self.behaviors: list[Behavior] = []
        self._seen_behaviors: set[Behavior] = set()
        self.current: MemoryEntry | None = None
        self.calls: list[dict] = []
        self.attempts: list[dict] = []
        self.events: list[dict] = []

    @prompt(template="plan.j2", output_type=Plan)
    async def plan(
        self,
        observation: Observation,
        history: list[dict],
        beliefs: dict,
        *,
        generated: Plan,
    ) -> Plan:
        """Infer teammates' next intentions and choose one complementary skill."""
        return self._check_plan(generated)

    @prompt(template="replan.j2", output_type=Plan)
    async def replan(
        self,
        observation: Observation,
        history: list[dict],
        beliefs: dict,
        feedback: str,
        *,
        generated: Plan,
    ) -> Plan:
        """Revise a rejected plan using authoritative feedback and failure analysis."""
        return self._check_plan(generated)

    @prompt(template="analyze_failure.j2")
    async def analyze_failure(
        self,
        observation: Observation,
        failed: dict,
        *,
        generated: str,
    ) -> str:
        """Explain the unsatisfied skill preconditions in one round."""
        return self._check_explanation(generated)

    @prompt(template="double_check.j2")
    async def double_check(
        self,
        observation: Observation,
        failed: dict,
        explanation: str,
        *,
        generated: str,
    ) -> str:
        """Check the initial explanation against the state and verifier evidence."""
        return self._check_explanation(generated)

    @prompt(template="conclude_failure.j2")
    async def conclude_failure(
        self,
        observation: Observation,
        failed: dict,
        explanation: str,
        review: str,
        *,
        generated: str,
    ) -> str:
        """Summarize the checked failure into guidance for replanning."""
        return self._check_explanation(generated)

    def _check_plan(self, generated: Plan) -> Plan:
        if generated.skill.name not in {skill.name for skill in self.skills}:
            raise InvalidPlan(f"Unknown skill: {generated.skill.name}")
        if set(generated.intentions) != set(self.teammates):
            raise InvalidPlan("Intentions must name exactly the configured teammates")
        return generated

    @staticmethod
    def _check_explanation(generated: str) -> str:
        if not generated.strip():
            raise InvalidPlan("Failure explanation is blank")
        return generated

    def observe(self, state: State) -> Observation:
        """Ground state and append newly observed behavior, including delayed events."""
        observation = self.ground(state)
        for behavior in observation.behaviors:
            if behavior not in self._seen_behaviors:
                self.behaviors.append(behavior)
                self._seen_behaviors.add(behavior)
        return observation

    def retrieve(self) -> tuple[list[dict], dict]:
        """Recent-K trajectories plus separate prediction/observation histories.

        Belief revision follows the official implementation: evidence is supplied
        to the next planner, without an additional model or invented truth labels.
        """
        recent = self.memory[-self.recent_k :] if self.recent_k else []
        history = [{**asdict(entry), "plan": entry.plan.model_dump()} for entry in recent]
        beliefs = {}
        if self.belief_revision:
            # ponytail: full belief history, add retrieval when episodes exceed context limits.
            beliefs = {
                "predictions": [
                    {
                        "step": entry.step,
                        "intentions": entry.plan.intentions,
                        "status": entry.status,
                    }
                    for entry in self.memory
                ],
                "observations": [asdict(behavior) for behavior in self.behaviors],
            }
        return history, beliefs

    async def run(self, initial: State, *, max_steps: int = 100) -> Result[State, Action]:
        """Run at most max_steps environment actions; use a fresh agent per episode.

        Planning retries never advance the environment. Fail closed on exhausted
        plans; no task-specific random movement or implicit fallback skill.
        """
        state, decisions, reason = initial, [], "budget"
        for _ in range(max_steps):
            decision = await self.act(state)
            decisions.append(decision)
            if decision.status != "action":
                reason = decision.status
                break
            state = await self.step(state, decision.action)
            if self.observe(state).terminal:
                reason = "terminal"
                break
        return Result(state, tuple(decisions), reason)

    async def act(self, state: State) -> Decision[Action]:
        """Return one action, continuing a valid skill or making bounded new plans.

        Malformed/unknown generated plans consume attempts. Callback, provider, and
        explainer errors propagate with partial records. Only SkillFailure from
        control requests replanning. Call serially; the caller applies the action.
        """
        observation = self.observe(state)
        if observation.terminal:
            return Decision("terminal")
        failed, feedback = None, ""
        if self.current is not None:
            previous = self.current
            decision = await self._continue_skill(state)
            if decision is not None:
                return decision
            if previous.status == "failed":
                failed = previous
        for attempt in range(self.max_plan_attempts):
            if failed is not None:
                feedback = await self._explain(observation, failed)
                failed = None
            record = {"step": observation.step, "attempt": attempt + 1}
            self.attempts.append(record)
            history, beliefs = self.retrieve()
            try:
                operation = self.replan if feedback else self.plan
                args = (
                    (observation, history, beliefs, feedback)
                    if feedback
                    else (observation, history, beliefs)
                )
                proposed = await self._invoke(operation, *args)
            except (ValidationError, InvalidPlan) as exc:
                if "response" not in self.calls[-1]:
                    raise  # Provider failures are not generated-plan rejections.
                feedback = f"{type(exc).__name__}: {exc}"
                record["error"] = feedback
                continue
            self.current = MemoryEntry(observation.step, observation.description, proposed)
            self.memory.append(self.current)
            record["entry"] = self.current
            entry = self.current
            decision = await self._continue_skill(state)
            if decision is not None:
                return decision
            if entry.status == "failed":
                failed = entry
            else:
                feedback = "Selected skill is already complete. Choose an unfinished useful skill."
        return Decision("exhausted")

    async def _continue_skill(self, state: State) -> Decision[Action] | None:
        entry = self.current
        event = {"skill": entry.plan.skill.model_dump(), "operation": "verify"}
        self.events.append(event)
        try:
            verdict = await self.verify(state, entry.plan.skill)
            event["result"] = verdict
            entry.feedback = verdict.feedback
            if verdict.status != "ready":
                entry.status = "complete" if verdict.status == "complete" else "failed"
                self.current = None
                return None
            event = {"skill": entry.plan.skill.model_dump(), "operation": "control"}
            self.events.append(event)
            try:
                action = await self.control(state, entry.plan.skill)
            except SkillFailure as exc:
                event["error"] = entry.feedback = str(exc)
                entry.status, self.current = "failed", None
                return None
            event["action"] = action
            entry.status = "active"
            entry.actions += 1
            return Decision("action", action, entry.plan.skill)
        except (Exception, asyncio.CancelledError) as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _explain(self, observation: Observation, entry: MemoryEntry) -> str:
        failed = {"skill": entry.plan.skill.model_dump(), "feedback": entry.feedback}
        explanation = await self._invoke(self.analyze_failure, observation, failed)
        if self.verification_rounds == 3:
            review = await self._invoke(self.double_check, observation, failed, explanation)
            explanation = await self._invoke(
                self.conclude_failure, observation, failed, explanation, review
            )
        return f"Verifier/controller feedback: {entry.feedback}\nFailure analysis: {explanation}"

    async def _invoke(self, operation, *args):
        record = {"operation": operation.__name__}
        self.calls.append(record)

        async def acall(context):
            record["prompt"] = context
            text, requests = await self.provider.acall(context)
            record["response"] = text
            if requests:
                raise InvalidPlan("ProAgent expects text responses, not tool requests")
            return text, requests

        try:
            return await operation(*args, provider=SimpleNamespace(acall=acall))
        except (Exception, asyncio.CancelledError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
