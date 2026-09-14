"""Evolve reward programs using policy-training statistics and execution feedback."""

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean

from pydantic import BaseModel, Field
from slick import prompt
from slick.providers import Provider


class Program(BaseModel, extra="forbid"):
    code: str = Field(min_length=1)


@dataclass(frozen=True)
class TrainingResult:
    score: float = 0.0
    components: Mapping[str, Sequence[float]] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class Trial:
    code: str
    result: TrainingResult
    feedback: dict


class Eureka:
    """The callback compiles and trains each reward program in caller-owned isolation."""

    def __init__(
        self,
        task: str,
        provider: Provider,
        train: Callable[[str], Awaitable[TrainingResult]],
        *,
        interface: str,
    ):
        self.task, self.provider, self.train, self.interface = task, provider, train, interface

    @prompt(template="initialize.j2", output_type=Program)
    async def initialize(self, *, generated: Program) -> str:
        if not generated.code.strip():
            raise ValueError("empty reward program")
        return generated.code

    @prompt(template="revise.j2", output_type=Program)
    async def revise(self, code: str, feedback: dict, *, generated: Program) -> str:
        if not generated.code.strip():
            raise ValueError("empty reward program")
        return generated.code

    async def assess(self, code: str) -> Trial:
        result = await self.train(code)
        feedback = {"error": result.error}
        if not result.error:
            if not math.isfinite(result.score):
                raise ValueError("training score must be finite")
            feedback["task_score"] = result.score
            feedback["components"] = {}
            for name, values in result.components.items():
                if len(values) == 0 or not all(math.isfinite(x) for x in values):
                    raise ValueError("reward component statistics must be nonempty and finite")
                stride = max(len(values) // 10, 1)
                feedback["components"][name] = {
                    "sampled": list(values[::stride]),
                    "stride": stride,
                    "min": min(values),
                    "max": max(values),
                    "mean": fmean(values),
                }
        snapshot = TrainingResult(
            result.score,
            {name: tuple(values) for name, values in result.components.items()},
            result.error,
        )
        return Trial(code, snapshot, feedback)

    async def sample_batch(self, context, count):
        batch = []
        for _ in range(count):
            code = (
                await self.initialize(provider=self.provider)
                if context is None
                else await self.revise(context.code, context.feedback, provider=self.provider)
            )
            batch.append(await self.assess(code))
        return batch

    async def run(self, *, rounds: int = 5, candidates: int = 16) -> dict:
        best = context = None
        history = []
        for _ in range(rounds):
            batch = await self.sample_batch(context, candidates)
            history.extend(batch)
            successful = [trial for trial in batch if not trial.result.error]
            if successful:
                context = max(successful, key=lambda trial: trial.result.score)
                if best is None or context.result.score > best.result.score:
                    best = context
            elif candidates == 1:
                context = batch[0]
        return {"best": best, "trials": history, "training_calls": len(history)}
