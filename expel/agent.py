"""Learn cross-task insights and retrieve successful experiences without weight updates.

Adapted from LeapLabTHU/ExpeL, commit e41ec9a; see README.md and LICENSE.upstream.
"""

import heapq
import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider


class Action(BaseModel, extra="forbid"):
    thought: str
    action: Annotated[str, StringConstraints(min_length=1)]


@dataclass(frozen=True)
class Outcome:
    observation: str
    done: bool = False
    succeeded: bool = False
    reward: float = 0.0
    feedback: str = ""


@dataclass(frozen=True)
class Experience:
    task: str
    trajectory: str
    succeeded: bool
    reward: float = 0.0
    feedback: str = ""
    steps: int = 0


@dataclass(frozen=True)
class Insight:
    text: str
    importance: int = 2


@dataclass(frozen=True)
class Evaluation:
    experiences: tuple[Experience, ...]

    @property
    def success_rate(self) -> float:
        return (
            sum(item.succeeded for item in self.experiences) / len(self.experiences)
            if self.experiences
            else 0.0
        )


Reset = Callable[[str], Awaitable[str]]
Step = Callable[[str], Awaitable[Outcome]]
Embed = Callable[[str], Awaitable[Sequence[float]]]


def update_insights(insights: tuple[Insight, ...], generated: str) -> tuple[Insight, ...]:
    """Apply a whole valid batch against the prompt's numbering, or raise unchanged."""
    if generated.strip() == "NONE":
        return insights
    lines = [line.strip() for line in generated.splitlines() if line.strip()]
    if not 1 <= len(lines) <= 4:
        raise ValueError("expected one to four insight operations, or NONE")
    updated = list(insights)
    touched = set()
    for line in lines:
        match = re.fullmatch(r"(ADD|EDIT|UPVOTE|DOWNVOTE) ([1-9][0-9]*): (\S.*)", line)
        if match is None:
            raise ValueError(f"invalid insight operation: {line}")
        operation, number, text = match.groups()
        index = int(number) - 1
        if operation == "ADD":
            if index != len(updated):
                raise ValueError("ADD must use the next unused rule number")
            updated.append(Insight(text))
            continue
        if index >= len(insights) or index in touched:
            raise ValueError("existing rule must occur in the prompt and be changed at most once")
        touched.add(index)
        previous = insights[index]
        if operation in {"UPVOTE", "DOWNVOTE"} and text != previous.text:
            raise ValueError("votes must repeat the referenced rule exactly")
        delta = -1 if operation == "DOWNVOTE" else 1
        updated[index] = Insight(text, previous.importance + delta)
    surviving = [item for item in updated if item.importance > 0]
    if len({item.text for item in surviving}) != len(surviving):
        raise ValueError("insights must not duplicate existing text")
    return tuple(sorted(surviving, key=lambda item: item.importance, reverse=True))


def nonblank(text: str) -> str:
    if not text.strip():
        raise ValueError("generated response is blank")
    return text


class ExpeL:
    """Own sequential ReAct episodes, cross-task experience, and insight extraction.

    reset(task) starts a fresh environment; step(action) executes and evaluates
    an action. Only the environment declares success. Callbacks own execution
    isolation. One active operation per instance; providers carry no implicit history.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        reset: Reset,
        step: Step,
        embed: Embed,
        *,
        manual_examples: tuple[Experience, ...] = (),
        insights: tuple[Insight, ...] = (),
        reflection_provider: Provider | None = None,
        insight_provider: Provider | None = None,
    ):
        self.task = task
        self.provider = provider
        self.reset = reset
        self.step = step
        self.embed = embed
        self.manual_examples = manual_examples
        self.initial_insights = insights
        self.reflection_provider = provider if reflection_provider is None else reflection_provider
        self.insight_provider = provider if insight_provider is None else insight_provider
        self.pool: list[Experience] = list(manual_examples)
        self.insights = insights
        self.embeddings: dict[str, tuple[float, ...]] = {}
        self.calls: list[dict] = []
        self.trajectory = ""
        self.evaluations: list[Experience] = []

    @prompt(template="act.j2", output_type=Action)
    async def act(
        self,
        task: str,
        trajectory: str,
        examples: tuple[Experience, ...],
        guidance: tuple[str, ...],
        reflections: tuple[str, ...],
        remaining_steps: int,
        *,
        generated: Action,
    ) -> Action:
        """Choose one action; observations and success must come from the environment."""
        nonblank(generated.action)
        return generated

    @prompt(template="reflect.j2")
    async def reflect(
        self,
        failure: Experience,
        reflections: tuple[str, ...],
        *,
        generated: str,
    ) -> str:
        """Produce a corrective plan for the next attempt of this task."""
        return nonblank(generated)

    @prompt(template="compare.j2")
    async def compare(
        self,
        success: Experience,
        failure: Experience,
        *,
        generated: str,
    ) -> tuple[Insight, ...]:
        """Extract rules by contrasting successful and failed trials of one task."""
        return update_insights(self.insights, generated)

    @prompt(template="summarize.j2")
    async def summarize(
        self,
        successes: tuple[Experience, ...],
        *,
        generated: str,
    ) -> tuple[Insight, ...]:
        """Extract common practices from successes on distinct tasks."""
        return update_insights(self.insights, generated)

    @prompt(template="transfer.j2")
    async def transfer(
        self,
        target_task: str,
        examples: tuple[Experience, ...],
        *,
        generated: str,
    ) -> str:
        """Ground source insights in target-domain demonstrations."""
        return nonblank(generated)

    async def run(
        self,
        training_tasks: Sequence[str],
        evaluation_tasks: Sequence[str],
        *,
        max_retries: int = 3,
        max_steps: int = 20,
        chunk_size: int = 8,
        k: int = 3,
        seed: int = 0,
    ) -> Evaluation:
        """Gather training trials, rebuild insights, then evaluate each task once."""
        await self.gather(training_tasks, max_retries=max_retries, max_steps=max_steps)
        await self.extract_insights(chunk_size=chunk_size, seed=seed)
        return await self.evaluate(evaluation_tasks, max_steps=max_steps, k=k)

    async def gather(
        self,
        tasks: Sequence[str],
        *,
        max_retries: int = 3,
        max_steps: int = 20,
    ) -> tuple[Experience, ...]:
        """Append every completed training trial; Z retries means at most Z+1 trials."""
        start = len(self.pool)
        for task in tasks:
            reflections: tuple[str, ...] = ()
            for trial in range(max_retries + 1):
                experience = await self.episode(
                    task, self.manual_examples, (), reflections, max_steps
                )
                self.pool.append(experience)
                if experience.succeeded:
                    break
                if trial < max_retries:
                    reflection = await self._invoke(self.reflect, experience, reflections)
                    reflections += (reflection,)
        return tuple(self.pool[start:])

    async def extract_insights(self, *, chunk_size: int = 8, seed: int = 0) -> tuple[Insight, ...]:
        """Rebuild from seed insights and the current pool, comparisons before successes."""
        self.insights = self.initial_insights
        by_task: dict[str, list[Experience]] = {}
        for experience in self.pool:
            by_task.setdefault(experience.task, []).append(experience)
        successes = []
        for experiences in by_task.values():
            passed = [item for item in experiences if item.succeeded]
            if not passed:
                continue
            successes.append(passed[0])
            for success in passed:
                for failure in experiences:
                    if not failure.succeeded:
                        self.insights = await self._invoke(self.compare, success, failure)
        random.Random(seed).shuffle(successes)
        for start in range(0, len(successes), chunk_size):
            chunk = tuple(successes[start : start + chunk_size])
            self.insights = await self._invoke(self.summarize, chunk)
        return self.insights

    async def retrieve(self, task: str, *, k: int = 3) -> tuple[Experience, ...]:
        """Return top-k distinct successful tasks by exact embedding inner product."""
        successes: dict[str, Experience] = {}
        for experience in self.pool:
            if experience.succeeded:
                successes.setdefault(experience.task, experience)
        if k == 0 or not successes:
            return ()
        for text in (task, *successes):
            if text not in self.embeddings:
                self.embeddings[text] = tuple(await self.embed(text))
        query = self.embeddings[task]

        def similarity(experience: Experience) -> float:
            score = sum(a * b for a, b in zip(query, self.embeddings[experience.task], strict=True))
            if not math.isfinite(score):
                raise ValueError("non-finite embedding similarity")
            return score

        # ponytail: linear scan of cached vectors; use a vector index when the pool is large.
        return tuple(heapq.nlargest(k, successes.values(), key=similarity))

    async def evaluate(
        self,
        tasks: Sequence[str],
        *,
        max_steps: int = 20,
        k: int = 3,
    ) -> Evaluation:
        """Attempt each task once without updating the pool or extracted insights.

        k=0 selects fixed manual demonstrations (also used for insight transfer).
        With k>0, retrieval uses only successes already present in the pool.
        """
        self.evaluations = []
        guidance = tuple(item.text for item in self.insights)
        for task in tasks:
            examples = self.manual_examples if k == 0 else await self.retrieve(task, k=k)
            self.evaluations.append(await self.episode(task, examples, guidance, (), max_steps))
        return Evaluation(tuple(self.evaluations))

    async def adapt_insights(self, target_task: str, examples: tuple[Experience, ...] = ()) -> str:
        """Return a transfer paragraph without modifying source memory or environments."""
        return await self._invoke(self.transfer, target_task, examples)

    async def episode(
        self,
        task: str,
        examples: tuple[Experience, ...],
        guidance: tuple[str, ...],
        reflections: tuple[str, ...],
        max_steps: int,
    ) -> Experience:
        """Reset the environment and interleave generated actions with real observations."""
        self.trajectory = ""
        self.trajectory = f"Observation 0: {await self.reset(task)}"
        outcome = Outcome("")
        steps = 0
        for index in range(max_steps):
            action = await self._invoke(
                self.act, task, self.trajectory, examples, guidance, reflections, max_steps - index
            )
            self.trajectory += (
                f"\nThought {index + 1}: {action.thought}\nAction {index + 1}: {action.action}"
            )
            outcome = await self.step(action.action)
            steps += 1
            self.trajectory += (
                f"\nObservation {steps}: {outcome.observation}\nReward {steps}: {outcome.reward}"
                f"\nFeedback {steps}: {outcome.feedback}"
            )
            if outcome.done or outcome.succeeded:
                break
        feedback = outcome.feedback
        if not (outcome.done or outcome.succeeded):
            feedback = (feedback + "\nStep budget exhausted.").strip()
        return Experience(task, self.trajectory, outcome.succeeded, outcome.reward, feedback, steps)

    async def _invoke(self, operation, *args):
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Record raw output before Slick parsing and rule validation."""
        record = self.calls[-1]
        operation = record["operation"]
        provider = self.provider
        if operation == "reflect":
            provider = self.reflection_provider
        elif operation in {"compare", "summarize", "transfer"}:
            provider = self.insight_provider
        record["prompt"] = context
        response, requests = await provider.acall(context, tools=tools, tool_results=tool_results)
        record["response"] = response
        if requests:
            raise ValueError("ExpeL prompt operations require text, not provider tool requests")
        return response, requests
