"""Search SEED pipelines using cached validation and cost/effectiveness frontiers."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue


@dataclass(frozen=True)
class Example:
    input: str
    output: JsonValue


@dataclass(frozen=True)
class Prediction:
    """None outside this record means abstain; Prediction(None) is a JSON null answer."""

    value: JsonValue
    confidence: float = 1.0


Predictor = Callable[[Sequence[str]], Awaitable[Sequence[Prediction | None]]]
Evaluator = Callable[[Sequence[Example], Sequence[Prediction | None]], Awaitable[float]]


@dataclass(frozen=True)
class Module:
    """A frozen configuration; change key whenever its predictor or cost changes."""

    key: str
    family: str
    cost: float
    predict: Predictor
    threshold: float | None = None


@dataclass(frozen=True)
class Plan:
    modules: tuple[Module, ...]
    effectiveness: float
    cost: float


@dataclass(frozen=True)
class Optimization:
    best: Plan
    frontier: tuple[Plan, ...]
    evaluated: int


def checked_predictions(predictions, count):
    predictions = tuple(predictions)
    if len(predictions) != count:
        raise ValueError("predictor returned the wrong number of records")
    for prediction in predictions:
        if prediction is not None and not (
            math.isfinite(prediction.confidence) and 0 <= prediction.confidence <= 1
        ):
            raise ValueError("measured confidence must be finite and in [0, 1]")
    return predictions


def priority_order(modules, fallback):
    def priority(module):
        coverage = 1 - fallback[module.key]
        if module.cost == 0:
            return math.inf if coverage else 0.0
        return coverage / module.cost

    return tuple(sorted(modules, key=priority, reverse=True))


def execution_cost(modules, fallback):
    cost, reach = 0.0, 1.0
    for module in modules:
        cost += reach * module.cost
        reach *= fallback[module.key]
    return cost


def skyline(plans):
    """Keep cheapest per effectiveness, then remove weakly dominated plans (Alg. 2)."""
    ordered = sorted(plans, key=lambda p: (-p.effectiveness, p.cost))
    frontier, cheapest = [], math.inf
    for plan in ordered:
        if plan.cost < cheapest:
            frontier.append(plan)
            cheapest = plan.cost
    return frontier


class SeedOptimizer:
    """Own validation memoization and generic/specialized/exhaustive plan search.

    Evaluator returns a finite higher-is-better metric in [0, 1]. Module outputs
    are measured once per immutable key. LLM remains available as a safety fallback,
    including when its estimated reach is zero. Generic skyline pruning assumes
    dominance persists in descendants; exhaustive mode makes no such assumption.
    """

    def __init__(self, validation: Sequence[Example], evaluate: Evaluator):
        self.validation, self.evaluate = tuple(validation), evaluate
        self.outputs, self.fallback, self.scores = {}, {}, {}

    async def _measure(self, modules):
        inputs = [example.input for example in self.validation]
        for module in modules:
            if module.key not in self.outputs:
                outputs = checked_predictions(await module.predict(inputs), len(inputs))
                self.outputs[module.key] = outputs
                self.fallback[module.key] = sum(p is None for p in outputs) / len(inputs)

    async def _assess(self, modules):
        key = tuple(module.key for module in modules)
        if key not in self.scores:
            outputs = [None] * len(self.validation)
            for module in modules:
                for index, prediction in enumerate(self.outputs[module.key]):
                    if outputs[index] is None:
                        outputs[index] = prediction
            score = float(await self.evaluate(self.validation, outputs))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("measured effectiveness must be finite and in [0, 1]")
            self.scores[key] = score
        return Plan(tuple(modules), self.scores[key], execution_cost(modules, self.fallback))

    async def _specialized(self, modules, initial, gap, beam_size):
        frontier = skyline(initial)
        for family in ("CodeGen", "ModelGen", "CacheReuse"):
            candidates = list(frontier)
            for plan in frontier:
                for module in modules:
                    if module.family == family:
                        ordered = priority_order((*plan.modules, module), self.fallback)
                        candidates.append(await self._assess(ordered))
            best = max(p.effectiveness for p in candidates)
            frontier = skyline(p for p in candidates if p.effectiveness >= best - gap - 1e-12)
            if beam_size is not None:
                frontier = frontier[:beam_size]
        return frontier

    async def _generic(self, modules, initial, prune):
        layer, all_plans = initial, list(initial)
        families = {m.family for m in modules} - {"LLM"}
        for _ in families:
            groups, seen = {}, set()
            for plan in layer:
                used = frozenset(m.family for m in plan.modules)
                for module in modules:
                    if module.family in used:
                        continue
                    for position in range(len(plan.modules) + 1):
                        ordered = (*plan.modules[:position], module, *plan.modules[position:])
                        key = tuple(m.key for m in ordered)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidate = await self._assess(ordered)
                        groups.setdefault(used | {module.family}, []).append(candidate)
            layer = [p for group in groups.values() for p in (skyline(group) if prune else group)]
            all_plans.extend(layer)
        return skyline(all_plans)

    async def run(
        self,
        modules: Sequence[Module],
        *,
        gap: float = 0.05,
        mode: Literal["specialized", "generic", "exhaustive"] = "specialized",
        beam_size: int | None = 20,
    ) -> Optimization:
        await self._measure(modules)
        before = len(self.scores)
        initial = [await self._assess((m,)) for m in modules if m.family == "LLM"]
        if mode == "specialized":
            frontier = await self._specialized(modules, initial, gap, beam_size)
        else:
            frontier = await self._generic(modules, initial, prune=mode == "generic")
        best_score = max(p.effectiveness for p in frontier)
        feasible = [p for p in frontier if p.effectiveness >= best_score - gap - 1e-12]
        best = min(feasible, key=lambda p: (p.cost, -p.effectiveness, len(p.modules)))
        return Optimization(best, tuple(frontier), len(self.scores) - before)
