"""Compile module demonstrations from successful teacher traces and validation search."""

import math
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class Example:
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]


@dataclass(frozen=True)
class Trace:
    module: str
    example: Example


@dataclass(frozen=True)
class Prediction:
    outputs: Mapping[str, Any]
    trace: Sequence[Trace] = ()


Program = dict[str, tuple[Example, ...]]
Execute = Callable[[Program, Mapping[str, Any], int], Awaitable[Prediction]]


class DSPyBootstrap:
    """Executors run fixed modules; compilation, acceptance and selection stay local."""

    def __init__(
        self,
        modules: Sequence[str],
        teacher: Execute,
        student: Execute,
        metric: Callable[[Example, Prediction], float],
        *,
        threshold: float = 1.0,
    ):
        self.modules, self.teacher, self.student = tuple(modules), teacher, student
        self.metric, self.threshold = metric, threshold

    def labeled(self, examples, count):
        rng = random.Random(0)
        return {
            module: tuple(rng.sample(list(examples), min(count, len(examples))))
            for module in self.modules
        }

    def score(self, example, prediction):
        score = self.metric(example, prediction)
        if not math.isfinite(score):
            raise ValueError("metric must be finite")
        return score

    async def bootstrap(self, train, count, labeled_count, rounds):
        teacher = self.labeled(train, labeled_count)
        traces, accepted, rng = defaultdict(list), set(), random.Random(0)
        for index, example in enumerate(train):
            if len(accepted) >= count:
                break
            # Remove the current label before invoking the teacher.
            visible = {
                name: tuple(d for d in demos if d != example) for name, demos in teacher.items()
            }
            for rollout in range(rounds):
                prediction = await self.teacher(visible, example.inputs, rollout)
                self.teacher_calls += 1
                if self.score(example, prediction) >= self.threshold:
                    grouped = defaultdict(list)
                    for step in prediction.trace:
                        if step.module in self.modules:
                            grouped[step.module].append(step.example)
                    for module, demos in grouped.items():
                        chosen = (
                            rng.choice(demos[:-1])
                            if len(demos) > 1 and rng.random() < 0.5
                            else demos[-1]
                        )
                        traces[module].append(chosen)
                    accepted.add(index)
                    break
        remaining = [ex for index, ex in enumerate(train) if index not in accepted]
        random.Random(0).shuffle(remaining)
        rng = random.Random(0)
        program = {}
        for module in self.modules:
            augmented = traces[module][:count]
            raw = rng.sample(remaining, min(max(labeled_count - len(augmented), 0), len(remaining)))
            program[module] = tuple(augmented + raw)
        return program

    async def validate(self, program, validation):
        scores = []
        for example in validation:
            prediction = await self.student(program, example.inputs, 0)
            self.validation_calls += 1
            scores.append(self.score(example, prediction))
        return fmean(scores)

    async def run(
        self,
        train: Sequence[Example],
        validation: Sequence[Example],
        *,
        candidates: int = 16,
        max_bootstrapped: int = 4,
        max_labeled: int = 16,
        bootstrap_rounds: int = 1,
    ) -> dict:
        self.teacher_calls = self.validation_calls = 0
        results = []
        for seed in range(-3, candidates):
            data = list(train)
            if seed == -3:
                program = {module: () for module in self.modules}
            elif seed == -2:
                program = self.labeled(data, max_labeled)
            else:
                count = max_bootstrapped
                if seed >= 0:
                    random.Random(seed).shuffle(data)
                    count = (
                        random.Random(seed).randint(1, max_bootstrapped) if max_bootstrapped else 0
                    )
                program = await self.bootstrap(data, count, max_labeled, bootstrap_rounds)
            results.append(
                {
                    "seed": seed,
                    "program": program,
                    "score": await self.validate(program, validation),
                }
            )
        return {
            "best": max(results, key=lambda item: item["score"]),
            "candidates": results,
            "teacher_calls": self.teacher_calls,
            "validation_calls": self.validation_calls,
        }
