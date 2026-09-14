"""Optimize reusable instructions through generation, blind auditing, and textual feedback."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Audit(BaseModel, extra="forbid", frozen=True):
    score: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)
    critique: Text


class Gradient(BaseModel, extra="forbid", frozen=True):
    critique: Text
    members: tuple[Annotated[int, Field(ge=0, strict=True)], ...] = Field(min_length=1)


class Gradients(BaseModel, extra="forbid"):
    groups: list[Gradient] = Field(min_length=1)


@dataclass(frozen=True)
class Case:
    input: str
    context: str = ""
    reference: str = ""


@dataclass(frozen=True)
class Example:
    input: str
    output: str
    context: str = ""


@dataclass(frozen=True)
class Instruction:
    text: str
    examples: tuple[Example, ...] = ()


@dataclass(frozen=True)
class Observation:
    case: Case
    output: str
    audit: Audit


@dataclass(frozen=True)
class Batch:
    observations: tuple[Observation, ...]
    best: tuple[Observation, ...]

    @property
    def score(self) -> float:
        return fmean(item.audit.score for item in self.best)


@dataclass
class Trial:
    instruction: Instruction
    gradients: tuple[Gradient, ...] = ()
    train: Batch | None = None
    gold: Batch | None = None
    accepted: bool = False
    reason: str = "pending"


@dataclass(frozen=True)
class Result:
    instruction: Instruction
    train: Batch
    gold: Batch
    history: tuple[Trial, ...]
    stop_reason: Literal["threshold", "budget", "no_gradients"]
    calls: int
    evaluations: int


Evaluator = Callable[[Case, str], Awaitable[Audit]]
Reviewer = Callable[[Instruction, Instruction], Awaitable[bool]]


class MetaPrompting:
    """Own the Adversarial Trinity and its incumbent prompt.

    Supply either an async evaluator or a stateless auditor provider. Providers
    own decoding settings and transport retries. No generated artifact is executed.
    Failures propagate; raw calls, evaluations, and pending trials remain inspectable.
    """

    def __init__(
        self,
        task: str,
        generator: Provider,
        optimizer: Provider,
        *,
        rules: Sequence[str],
        auditor: Provider | None = None,
        evaluate: Evaluator | None = None,
        review: Reviewer | None = None,
    ):
        self.task = task
        self.rules = tuple(rules)
        self.providers = {"generator": generator, "auditor": auditor, "optimizer": optimizer}
        self.evaluate = evaluate
        self.review = review
        self.history: list[Trial] = []
        self.calls: list[dict] = []
        self.evaluations: list[dict] = []
        self.instruction: Instruction | None = None

    @prompt(template="generate.j2")
    async def generate(
        self, instruction: str, case: Case, examples: Sequence[Example], *, generated: str
    ) -> str:
        """Generate only the final artifact, preserving its whitespace."""
        if not generated.strip():
            raise ValueError("generated artifact is blank")
        return generated

    @prompt(template="audit.j2", output_type=Audit)
    async def audit(self, case: Case, output: str, *, generated: Audit) -> Audit:
        """Judge the artifact without the generating instruction or conversation."""
        return generated

    @prompt(template="aggregate.j2", output_type=Gradients)
    async def aggregate(
        self, failures: list[dict], *, generated: Gradients
    ) -> tuple[Gradient, ...]:
        """Require a complete partition so clustering cannot invent or omit reports."""
        members = [index for group in generated.groups for index in group.members]
        if sorted(members) != list(range(len(failures))):
            raise ValueError("gradient groups must cover every failure exactly once")
        return tuple(sorted(generated.groups, key=lambda group: -len(group.members)))

    @prompt(template="revise.j2")
    async def revise(
        self,
        instruction: Instruction,
        gradients: Sequence[Gradient],
        failures: list[dict],
        history: list[dict],
        *,
        generated: str,
    ) -> str:
        """Refactor reusable instructions using critiques and previous update decisions."""
        if not generated.strip():
            raise ValueError("revised instruction is blank")
        return generated.strip()

    async def run(
        self,
        initial_prompt: str,
        train: Sequence[Case],
        gold: Sequence[Case],
        *,
        best_of_n: int = 3,
        max_iterations: int = 5,
        threshold: float = 0.95,
        anchors: Sequence[Example] = (),
        max_examples: int = 4,
    ) -> Result:
        """Evaluate a baseline, then attempt at most max_iterations prompt updates.

        Train and gold are nonempty, disjoint batches. Gold is regression validation,
        never optimizer feedback or a source of demonstrations. Anchors are separate
        human-verified demonstrations, retained independently of max_examples.
        Runs reset state; use one run at a time per instance.
        """
        self.history, self.calls, self.evaluations = [], [], []
        self.instruction = Instruction(initial_prompt, tuple(anchors))
        incumbent = Trial(self.instruction)
        self.history.append(incumbent)
        await self._measure(incumbent, train, gold, best_of_n)
        incumbent.accepted, incumbent.reason = True, "baseline"
        for iteration in range(max_iterations + 1):
            if min(incumbent.train.score, incumbent.gold.score) >= threshold:
                reason = "threshold"
                break
            if iteration == max_iterations:
                reason = "budget"
                break
            proposal = await self._propose(incumbent, tuple(anchors), max_examples)
            if proposal is None:
                reason = "no_gradients"
                break
            if await self._consider(incumbent, proposal, train, gold, best_of_n):
                incumbent = proposal
                self.instruction = incumbent.instruction
        return Result(
            incumbent.instruction,
            incumbent.train,
            incumbent.gold,
            tuple(self.history),
            reason,
            len(self.calls),
            len(self.evaluations),
        )

    async def predict(self, instruction: Instruction, case: Case) -> str:
        """Generate once using an optimized instruction; no audit or reference leakage."""
        examples = [
            example
            for example in instruction.examples
            if (example.input, example.context) != (case.input, case.context)
        ]
        return await self._invoke(self.generate, "generator", instruction.text, case, examples)

    async def _batch(
        self, instruction: Instruction, cases: Sequence[Case], n: int, split: str
    ) -> Batch:
        observations, best = [], []
        for case in cases:
            candidates = []
            for _ in range(n):
                output = await self.predict(instruction, case)
                record = {"split": split, "case": case, "output": output}
                self.evaluations.append(record)
                try:
                    audit = (
                        await self.evaluate(case, output)
                        if self.evaluate is not None
                        else await self._invoke(self.audit, "auditor", case, output)
                    )
                    # Validate measured results too, including constructed/mutated models.
                    audit = Audit.model_validate(audit.model_dump())
                    record["audit"] = audit
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                candidates.append(Observation(case, output, audit))
            observations.extend(candidates)
            best.append(max(candidates, key=lambda item: item.audit.score))
        return Batch(tuple(observations), tuple(best))

    async def _measure(self, trial: Trial, train: Sequence[Case], gold: Sequence[Case], n: int):
        trial.train = await self._batch(trial.instruction, train, n, "train")
        trial.gold = await self._batch(trial.instruction, gold, n, "gold")

    async def _propose(
        self, incumbent: Trial, anchors: tuple[Example, ...], limit: int
    ) -> Trial | None:
        failures = [
            {
                "input": item.case.input,
                "context": item.case.context,
                "output": item.output,
                "loss": 1 - item.audit.score,
                "critique": item.audit.critique,
            }
            for item in incumbent.train.observations
            if item.audit.score < 1
        ]
        if not failures:
            return None
        gradients = await self._invoke(self.aggregate, "optimizer", failures)
        # Only training information and acceptance decisions cross into optimization.
        history = [
            {
                "instruction": trial.instruction.text,
                "accepted": trial.accepted,
                "reason": trial.reason,
            }
            for trial in self.history
        ]
        text = await self._invoke(
            self.revise, "optimizer", incumbent.instruction, gradients, failures, history
        )
        successes = [
            Example(item.case.input, item.output, item.case.context)
            for item in incumbent.train.best
            if item.audit.score == 1
        ]
        previous = [example for example in incumbent.instruction.examples if example not in anchors]
        examples = tuple(dict.fromkeys(successes + previous))[:limit]
        proposal = Trial(Instruction(text, tuple(dict.fromkeys(anchors + examples))), gradients)
        self.history.append(proposal)
        return proposal

    async def _consider(
        self, incumbent: Trial, proposal: Trial, train: Sequence[Case], gold: Sequence[Case], n: int
    ) -> bool:
        if self.review is not None and not await self.review(
            incumbent.instruction, proposal.instruction
        ):
            proposal.reason = "review_rejected"
            return False
        await self._measure(proposal, train, gold, n)
        if any(
            new.audit.score < old.audit.score
            for new, old in zip(proposal.gold.best, incumbent.gold.best)
        ):
            proposal.reason = "gold_regression"
        elif proposal.train.score < incumbent.train.score:
            proposal.reason = "train_regression"
        else:
            proposal.accepted, proposal.reason = True, "accepted"
        return proposal.accepted

    async def _invoke(self, operation, role: str, *args):
        record = {"operation": operation.__name__, "role": role}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Record raw output before Slick parses or checks it; calls are sequential."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.providers[record["role"]].acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("Meta-Prompting requires text responses, not tool requests")
        return response, requests
