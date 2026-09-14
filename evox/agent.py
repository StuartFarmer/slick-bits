"""Co-evolve caller-scored artifacts and executable search strategies with Slick."""

import ast
import inspect
import math
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from statistics import fmean, pstdev
from typing import Annotated, Any, Literal

from pydantic import BaseModel, StrictInt, StringConstraints
from slick import Provider, prompt

from .initial_strategy import select

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StrategyRunner = Callable[[str, list[dict], dict, int], Awaitable[dict]]


class Operators(BaseModel, extra="forbid"):
    refine: Text
    diverge: Text


class StrategyDraft(BaseModel, extra="forbid"):
    code: str


class Selection(BaseModel, extra="forbid"):
    parent_id: StrictInt
    operator: Literal["free", "refine", "diverge"]
    inspiration_ids: list[StrictInt]


@dataclass(frozen=True)
class Config:
    iterations: int = 100
    window: int | None = None
    stagnation_threshold: float = 1e-6
    inspirations: int = 4
    strategy_attempts: int = 3
    maximize: bool = True
    seed: int = 0


@dataclass(frozen=True)
class Evaluation:
    score: float
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    id: int
    text: str
    score: float | None = None
    artifacts: dict = field(default_factory=dict)
    step: int = 0
    parent_id: int | None = None
    operator: str = "initial"
    strategy_id: int = 0
    error: str = ""


@dataclass(frozen=True)
class Strategy:
    id: int
    code: str


@dataclass(frozen=True)
class Window:
    strategy: Strategy
    start_state: dict
    end_state: dict
    steps: int
    improvement: float
    score: float


@dataclass
class StrategyAttempt:
    step: int
    code: str = ""
    accepted: bool = False
    error: str = ""


@dataclass(frozen=True)
class Generation:
    channel: str
    prompt: str
    response: str


@dataclass
class Result:
    best: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)
    strategies: list[Strategy] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    strategy_attempts: list[StrategyAttempt] = field(default_factory=list)
    selection_errors: list[dict] = field(default_factory=list)
    generations: list[Generation] = field(default_factory=list)
    steps: int = 0
    evaluation_calls: int = 0


class _RecordingProvider:
    """Capture raw responses before Slick parses or postprocesses them."""

    def __init__(self, provider: Provider, channel: str, records: list[Generation]):
        self.provider, self.channel, self.records = provider, channel, records

    async def acall(self, context, *, tools=None, tool_results=None):
        response, calls = await self.provider.acall(context, tools=tools, tool_results=tool_results)
        self.records.append(Generation(self.channel, context, response))
        return response, calls


def window_score(start: float, end: float, steps: int) -> float:
    """SkyDiscover LogWindowScorer, applied to higher-is-better quality.

    Adapted from skydiscover/optimize/search/evox/utils/search_scorer.py;
    see SOURCES.md and UPSTREAM_LICENSE for provenance and the Eq. 2 difference.
    """
    return (end - start) * (1 + math.log1p(max(0.0, start))) / math.sqrt(steps)


class EvoX:
    """Own one sequential run; callers supply task evaluation and strategy execution.

    Evaluator ValueError means candidate rejection. Strategy ValueError means
    invalid generated code/output. Other dependency errors and cancellation propagate.
    No sessions are shared: all generations use independent provider calls.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        *,
        run_strategy: StrategyRunner,
        config: Config | None = None,
        strategy_provider: Provider | None = None,
        operator_provider: Provider | None = None,
        initial_population: Sequence[tuple[str, Evaluation]] = (),
        initial_strategy: str | None = None,
        score_window: Callable[[float, float, int], float] = window_score,
    ):
        self.task, self.evaluate, self.run_strategy = task, evaluate, run_strategy
        self.config = config if config is not None else Config()
        self.window = (
            self.config.window
            if self.config.window is not None
            else max(1, self.config.iterations // 10)
        )
        self.score_window = score_window
        self.initial_population, self.initial_strategy = initial_population, initial_strategy
        self.result = Result()
        self.provider = _RecordingProvider(provider, "solution", self.result.generations)
        self.strategy_provider = _RecordingProvider(
            strategy_provider if strategy_provider is not None else provider,
            "strategy",
            self.result.generations,
        )
        self.operator_provider = _RecordingProvider(
            operator_provider if operator_provider is not None else provider,
            "operators",
            self.result.generations,
        )
        self.rng = random.Random(self.config.seed)
        self.active = Strategy(0, inspect.getsource(select))
        self.last_improvement = 0
        self.last_significant_quality: float | None = None

    @property
    def direction(self) -> str:
        return "maximize" if self.config.maximize else "minimize"

    @prompt(template="operators.j2", output_type=Operators)
    async def prepare_operators(self, *, generated: Operators) -> Operators:
        return generated

    @prompt(template="initialize.j2")
    async def initialize(self, *, generated: str) -> str:
        return generated

    @prompt(template="refine.j2")
    async def refine(
        self, parent: dict, inspirations: list[dict], guidance: str, *, generated: str
    ) -> str:
        return generated

    @prompt(template="diverge.j2")
    async def diverge(
        self, parent: dict, inspirations: list[dict], guidance: str, *, generated: str
    ) -> str:
        return generated

    @prompt(template="vary.j2")
    async def vary(self, parent: dict, inspirations: list[dict], *, generated: str) -> str:
        return generated

    @prompt(template="strategy.j2", output_type=StrategyDraft)
    async def mutate_strategy(
        self,
        parent: dict,
        inspirations: list[dict],
        state: dict,
        failures: list[str],
        *,
        generated: StrategyDraft,
    ) -> str:
        # Syntax/interface failures join runtime validation failures in the retry loop.
        return generated.code

    def quality(self, score: float) -> float:
        return score if self.config.maximize else -score

    def population(self, candidates: Sequence[Candidate] | None = None) -> list[dict]:
        candidates = self.result.candidates if candidates is None else candidates
        return [
            dict(asdict(c), quality=self.quality(c.score))
            for c in candidates
            if c.score is not None
        ]

    def descriptor(self, candidates: Sequence[Candidate] | None = None) -> dict:
        candidates = self.result.candidates if candidates is None else candidates
        population = self.population(candidates)
        scores = sorted(p["quality"] for p in population)
        recent = [c for c in candidates if c.step > 0][-self.window :]
        percentiles = {}
        for percentile in (10, 25, 50, 75, 90):
            if scores:
                position = (len(scores) - 1) * percentile / 100
                lower, upper = math.floor(position), math.ceil(position)
                percentiles[str(percentile)] = scores[lower] + (scores[upper] - scores[lower]) * (
                    position - lower
                )
        return {
            "size": len(population),
            "step": self.result.steps,
            "remaining": self.config.iterations - self.result.steps,
            "window": self.window,
            "inspirations": self.config.inspirations,
            "best": max(scores) if scores else None,
            "mean": fmean(scores) if scores else None,
            "spread": pstdev(scores) if scores else None,
            "percentiles": percentiles,
            "frontier": [
                {"id": p["id"], "quality": p["quality"]}
                for p in sorted(population, key=lambda p: p["quality"], reverse=True)[:5]
            ],
            "steps_since_improvement": self.result.steps - self.last_improvement,
            "parent_counts": dict(
                Counter(c.parent_id for c in candidates if c.parent_id is not None)
            ),
            "recent_parent_counts": dict(
                Counter(c.parent_id for c in recent if c.parent_id is not None)
            ),
            "operator_counts": dict(Counter(c.operator for c in recent)),
            "recent": [
                {
                    "id": c.id,
                    "score": c.score,
                    "parent_id": c.parent_id,
                    "operator": c.operator,
                    "error": c.error,
                }
                for c in recent
            ],
        }

    def accept_evaluation(self, candidate: Candidate, evaluation: Evaluation) -> None:
        if not math.isfinite(evaluation.score):
            raise ValueError("evaluator score must be finite")
        candidate.score, candidate.artifacts = evaluation.score, deepcopy(evaluation.artifacts)
        quality = self.quality(candidate.score)
        if self.result.best is None or quality > self.quality(self.result.best.score):
            self.result.best = candidate
        if (
            self.last_significant_quality is None
            or quality - self.last_significant_quality > self.config.stagnation_threshold
        ):
            self.last_significant_quality, self.last_improvement = quality, self.result.steps

    async def seed(self) -> None:
        for text, evaluation in self.initial_population:
            candidate = Candidate(len(self.result.candidates), text)
            self.accept_evaluation(candidate, evaluation)
            self.result.candidates.append(candidate)
        if self.initial_strategy is not None:
            await self.validate_strategy(self.initial_strategy)
            self.active = Strategy(0, self.initial_strategy)
        self.result.strategies.append(self.active)

    def check_selection(self, raw: dict, population: list[dict]) -> Selection:
        selection = Selection.model_validate(raw)
        ids = {p["id"] for p in population}
        inspirations = selection.inspiration_ids
        if selection.parent_id not in ids or not set(inspirations) <= ids:
            raise ValueError("strategy selected an unknown candidate ID")
        if (
            selection.parent_id in inspirations
            or len(set(inspirations)) != len(inspirations)
            or len(inspirations) > self.config.inspirations
        ):
            raise ValueError("inspirations must be distinct, exclude the parent, and fit the limit")
        return selection

    async def validate_strategy(self, code: str) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"strategy syntax: {exc}") from exc
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "select"
        ]
        if len(functions) != 1 or [a.arg for a in functions[0].args.args] != [
            "population",
            "state",
            "rng",
        ]:
            raise ValueError("strategy must define select(population, state, rng)")
        candidates = [c for c in self.result.candidates if c.score is not None] or [
            Candidate(0, "validation", score=0.0)
        ]
        # Validate on a singleton and the current population; validation consumes no search RNG.
        for snapshot in [candidates[:1], candidates]:
            population = self.population(snapshot)
            state = self.descriptor(snapshot)
            raw = await self.run_strategy(code, deepcopy(population), state, self.config.seed)
            self.check_selection(raw, population)

    async def select_context(self, population: list[dict]) -> Selection:
        state = self.descriptor()
        seed = self.rng.getrandbits(64)
        if self.active.id == 0 and self.initial_strategy is None:
            return self.check_selection(select(population, state, random.Random(seed)), population)
        try:
            raw = await self.run_strategy(self.active.code, deepcopy(population), state, seed)
            return self.check_selection(raw, population)
        except ValueError as exc:
            self.result.selection_errors.append(
                {"step": self.result.steps, "strategy_id": self.active.id, "error": str(exc)}
            )
            return self.check_selection(select(population, state, random.Random(seed)), population)

    async def evolve_solution(self) -> None:
        population = self.population()
        selection = None
        if population:
            selection = await self.select_context(population)
            by_id = {p["id"]: p for p in population}
            parent = by_id[selection.parent_id]
            inspirations = [by_id[i] for i in selection.inspiration_ids]
            if selection.operator == "refine":
                text = await self.refine(
                    parent, inspirations, self.operators.refine, provider=self.provider
                )
            elif selection.operator == "diverge":
                text = await self.diverge(
                    parent, inspirations, self.operators.diverge, provider=self.provider
                )
            else:
                text = await self.vary(parent, inspirations, provider=self.provider)
        else:
            text = await self.initialize(provider=self.provider)
        self.result.steps += 1
        candidate = Candidate(
            len(self.result.candidates),
            text,
            step=self.result.steps,
            parent_id=selection.parent_id if selection else None,
            operator=selection.operator if selection else "initial",
            strategy_id=self.active.id,
        )
        self.result.candidates.append(candidate)
        try:
            if not text.strip():
                raise ValueError("candidate is blank")
            self.result.evaluation_calls += 1
            evaluation = await self.evaluate(text)
            self.accept_evaluation(candidate, evaluation)
        except ValueError as exc:
            candidate.error = str(exc)

    async def evolve_window(self) -> Window:
        start_state = self.descriptor()
        steps = min(self.window, self.config.iterations - self.result.steps)
        start = start_state["best"]
        for _ in range(steps):
            await self.evolve_solution()
            if start is None and self.result.best is not None:
                start = self.quality(self.result.best.score)
        end_state = self.descriptor()
        end = end_state["best"]
        improvement = end - start if start is not None else 0.0
        score = self.score_window(start, end, steps) if start is not None else 0.0
        if not math.isfinite(score):
            raise ValueError("strategy window score must be finite")
        window = Window(self.active, start_state, end_state, steps, improvement, score)
        self.result.windows.append(window)
        return window

    def select_strategies(self, state: dict) -> tuple[dict, list[dict]]:
        ranked = sorted(self.result.windows, key=lambda w: w.score, reverse=True)
        # Rank-biased selection is stable across differently scaled task scores.
        parent = self.rng.choices(ranked, weights=[1 / (i + 1) for i in range(len(ranked))])[0]

        def distance(window):
            return sum(
                abs((window.start_state[key] or 0) - (state[key] or 0)) / (1 + abs(state[key] or 0))
                for key in ("size", "best", "spread", "steps_since_improvement")
            )

        similar = sorted(ranked, key=distance)
        inspirations = []
        seen = {parent.strategy.id}
        for window in [ranked[0], similar[0], *ranked, *similar]:
            if window.strategy.id not in seen:
                inspirations.append(asdict(window))
                seen.add(window.strategy.id)
            if len(inspirations) >= 2:
                break
        return asdict(parent), inspirations

    async def evolve_strategy(self) -> None:
        state = self.descriptor()
        parent, inspirations = self.select_strategies(state)
        failures = []
        for _ in range(self.config.strategy_attempts):
            attempt = StrategyAttempt(self.result.steps)
            self.result.strategy_attempts.append(attempt)
            try:
                attempt.code = await self.mutate_strategy(
                    parent,
                    inspirations,
                    state,
                    failures,
                    provider=self.strategy_provider,
                )
                await self.validate_strategy(attempt.code)
            except ValueError as exc:
                attempt.error = str(exc)
                failures.append(attempt.error)
                continue
            self.active = Strategy(len(self.result.strategies), attempt.code)
            self.result.strategies.append(self.active)
            attempt.accepted = True
            return

    async def run(self) -> Result:
        """Run Algorithm 1 windows; every generated solution attempt consumes one step."""
        await self.seed()
        if self.result.steps >= self.config.iterations:
            return self.result
        self.operators = await self.prepare_operators(provider=self.operator_provider)
        while self.result.steps < self.config.iterations:
            window = await self.evolve_window()
            if (
                window.improvement < self.config.stagnation_threshold
                and self.result.steps < self.config.iterations
                and self.result.best is not None
            ):
                await self.evolve_strategy()
        return self.result
