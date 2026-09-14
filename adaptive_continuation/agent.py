"""Control a running optimizer with bounded numeric actions and between-run reflection."""

import math
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StrictBool, ValidationError
from slick import prompt
from slick.providers import Provider
from slick.providers.base import ProviderError


@dataclass(frozen=True)
class Parameter:
    lower: float
    upper: float
    initial: float
    description: str = ""
    monotone: Literal["none", "increase", "decrease"] = "none"
    integer: bool = False

    def clamp(self, value: float) -> float:
        value = min(self.upper, max(self.lower, value))
        return round(value) if self.integer else value


@dataclass(frozen=True)
class Gate:
    """Cap a parameter when a measured metric strictly exceeds its threshold."""

    metric: str
    parameter: str
    threshold: float
    cap: float
    threshold_key: str | None = None


@dataclass(frozen=True)
class Measurement:
    """State and objective must describe the same evaluated solver iterate."""

    state: Any
    objective: float
    metrics: Mapping[str, float]
    feasible: bool = True


@dataclass(frozen=True)
class Snapshot:
    measurement: Measurement
    parameters: dict[str, float]
    iteration: int


Numeric = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class Action(BaseModel, extra="forbid"):
    parameters: dict[str, Numeric]
    restart: StrictBool
    note: str


class MetaUpdate(BaseModel, extra="forbid"):
    updates: dict[str, Numeric]
    note: str


class InvalidAction(ValueError):
    """A parsed model action violates the configured numeric interface."""


@dataclass(frozen=True)
class Result:
    final: Measurement
    best: Snapshot | None
    history: tuple[dict, ...]
    calls: tuple[dict, ...]
    main_evaluations: int
    tail_evaluations: int
    fallbacks: int
    tail_origin: str | None

    @property
    def primary_eligible(self) -> bool:
        return self.fallbacks == 0


def _check_numbers(values, names):
    if set(values) != set(names):
        raise InvalidAction("response must contain exactly the configured parameter names")
    if not all(math.isfinite(value) for value in values.values()):
        raise InvalidAction("generated numbers must be finite")


class AdaptiveContinuation:
    """Own observations, checkpoints, control calls, and meta-optimization.

    evaluate(state, parameters) performs one optimizer step and returns a measured
    iterate. The caller owns solver restoration, RNG state, execution isolation,
    and any external resources. States must support deepcopy (or be immutable
    checkpoint handles). One run at a time per instance; no mutable LLM session.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Any, dict[str, float]], Awaitable[Measurement]],
        *,
        parameters: Mapping[str, Parameter],
        gates: tuple[Gate, ...] = (),
        valid: Callable[[Measurement, dict[str, float]], bool] = lambda m, p: True,
        schedule: Callable[[int, int, dict], dict[str, float]] | None = None,
        guidance: str = "",
        maximize: bool = False,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.parameters, self.gates = dict(parameters), gates
        self.valid, self.schedule, self.guidance = valid, schedule, guidance
        self.maximize = maximize
        self.direction = "maximize" if maximize else "minimize"
        self.parameter_contract = {name: asdict(spec) for name, spec in parameters.items()}
        self.calls: list[dict] = []

    @prompt(template="control.j2", output_type=Action)
    async def control(self, observation: dict, *, generated: Action) -> Action:
        """Check generated parameter names and finite values before applying rails."""
        _check_numbers(generated.parameters, self.parameters)
        return generated

    @prompt(template="reflect.j2", output_type=MetaUpdate)
    async def reflect(
        self,
        summaries: list[dict],
        settings: dict,
        tunables: Mapping[str, Parameter],
        *,
        generated: MetaUpdate,
    ) -> dict:
        """Accept a bounded partial configuration update; never modify source code."""
        if not generated.updates.keys() <= tunables.keys():
            raise InvalidAction("unknown meta-parameter")
        _check_numbers(generated.updates, generated.updates)
        updates = {key: tunables[key].clamp(value) for key, value in generated.updates.items()}
        return {**settings, **updates}

    async def run(
        self,
        initial: Measurement,
        *,
        iterations: int = 300,
        settings: Mapping[str, float] | None = None,
        mode: Literal["llm", "fixed", "schedule"] = "llm",
        tail: Mapping[str, float] | None = None,
        tail_iterations: int = 40,
    ) -> Result:
        """Advance exactly iterations steps, then optionally finish from the best.

        Control follows every call_every completed steps that have a next step.
        Fixed mode has no calls, gates, or tail. Schedule mode has no state gates.
        With no valid snapshot, the tail starts from initial. It intentionally
        bypasses continuation gates and monotonicity. Missing fallback schedules
        mean hold the previous values. Failed model attempts are never retried.
        """
        self.settings = {"call_every": 5, **(settings or {})}
        self.calls, self.history = [], []
        self.main_evaluations = self.tail_evaluations = self.fallbacks = 0
        self.best, self.stagnation = None, 0
        self.current, self.initial = deepcopy(initial), deepcopy(initial)
        self.values = {key: spec.initial for key, spec in self.parameters.items()}
        self._record(0, "main")
        for iteration in range(1, iterations + 1):
            if mode == "schedule":
                self.values = self.schedule(iteration - 1, iterations, self.settings)
            if mode == "llm":
                self.values = self._rails(self.values, self.current)
            await self._step(iteration, "main")
            if (
                mode == "llm"
                and iteration < iterations
                and iteration % self.settings["call_every"] == 0
            ):
                await self._decide(iteration, iterations)
        tail_origin = (
            await self._finish(tail, tail_iterations, iterations) if mode != "fixed" else None
        )
        return Result(
            deepcopy(self.current),
            deepcopy(self.best),
            tuple(self.history),
            tuple(self.calls),
            self.main_evaluations,
            self.tail_evaluations,
            self.fallbacks,
            tail_origin,
        )

    async def _step(self, iteration, phase):
        if phase == "main":
            self.main_evaluations += 1
        else:
            self.tail_evaluations += 1
        self.current = await self.evaluate(self.current.state, self.values.copy())
        self._record(iteration, phase)

    def _record(self, iteration, phase):
        measured = self.current
        if not math.isfinite(measured.objective) or not all(
            math.isfinite(value) for value in measured.metrics.values()
        ):
            raise ValueError("measured objective and metrics must be finite")
        eligible = measured.feasible and self.valid(measured, self.values)
        improved = self.best is None or (
            measured.objective > self.best.measurement.objective
            if self.maximize
            else measured.objective < self.best.measurement.objective
        )
        if phase == "main":
            if eligible and improved:
                self.best = Snapshot(deepcopy(measured), self.values.copy(), iteration)
                self.stagnation = 0
            elif iteration > 0:
                self.stagnation += 1
        self.history.append(
            {
                "iteration": iteration,
                "phase": phase,
                "objective": measured.objective,
                "metrics": dict(measured.metrics),
                "feasible": measured.feasible,
                "valid": eligible,
                "parameters": self.values.copy(),
            }
        )

    def _observation(self, iteration, iterations):
        objective = self.current.objective
        recent = self.history[-6:]
        change1 = (objective - recent[-2]["objective"]) / max(abs(recent[-2]["objective"]), 1e-12)
        change5 = (objective - recent[0]["objective"]) / max(abs(recent[0]["objective"]), 1e-12)
        best = self.best.measurement.objective if self.best else None
        return {
            "iteration": iteration,
            "budget_used": iteration / iterations,
            "objective": objective,
            "best_objective": best,
            "relative_to_best": (objective - best) / max(abs(best), 1e-12)
            if best is not None
            else None,
            "change_1": change1,
            "change_5": change5,
            "objective_slope": (objective - recent[0]["objective"]) / (len(recent) - 1),
            "progress_proxy_slope": -abs(change5) if self.stagnation < 3 else 0,
            "stagnation": self.stagnation,
            "iterations_since_best": iteration - self.best.iteration if self.best else iteration,
            "best_is_valid": self.best is not None,
            "metrics": dict(self.current.metrics),
            "parameters": self.values.copy(),
            "settings": self.settings.copy(),
            "gates": [asdict(gate) for gate in self.gates],
        }

    def _rails(self, requested, measured):
        values = {}
        for name, spec in self.parameters.items():
            value = spec.clamp(requested[name])
            if spec.monotone == "decrease":
                value = min(value, self.values[name])
            elif spec.monotone == "increase":
                value = max(value, self.values[name])
            values[name] = value
        for gate in self.gates:
            threshold = self.settings[gate.threshold_key] if gate.threshold_key else gate.threshold
            if measured.metrics[gate.metric] > threshold:
                values[gate.parameter] = min(values[gate.parameter], gate.cap)
        return values

    async def _decide(self, iteration, iterations):
        observation = self._observation(iteration, iterations)
        try:
            decision = await self._invoke(self.control, observation)
        except (ValidationError, InvalidAction, ProviderError, OSError):
            self.fallbacks += 1
            parameters = (
                self.values.copy()
                if self.schedule is None
                else self.schedule(iteration, iterations, self.settings)
            )
            decision = Action(parameters=parameters, restart=False, note="deterministic fallback")
            self.calls[-1]["fallback"] = True
        values = self._rails(decision.parameters, self.current)
        restart = decision.restart and self.best is not None
        if restart:
            self.current = deepcopy(self.best.measurement)
            # The restored state can reactivate a gate that was clear before restart.
            values = self._rails(values, self.current)
        self.values = values
        self.calls[-1]["observation"] = observation
        self.calls[-1]["requested"] = decision.model_dump()
        self.calls[-1]["applied"] = {"parameters": values.copy(), "restart": restart}

    async def _finish(self, tail, count, iterations):
        if tail is None or count == 0:
            return None
        self.current = deepcopy(self.best.measurement if self.best else self.initial)
        self.values = dict(tail)
        for offset in range(1, count + 1):
            await self._step(iterations + offset, "tail")
        return "best" if self.best else "initial"

    async def run_meta(
        self,
        compare: Callable[[dict], Awaitable[dict]],
        *,
        rounds: int,
        settings: Mapping[str, float],
        tunables: Mapping[str, Parameter],
    ) -> dict:
        """Run caller-owned comparisons sequentially; tune before the next run.

        compare may cycle tasks/seeds and include all baselines in its summary.
        Invalid meta responses retain settings and log an error. Comparison
        failures propagate. No reflection call is spent after the final run.
        """
        current, summaries, reflections = dict(settings), [], []
        for index in range(rounds):
            summary = await compare(current.copy())
            summaries.append({"settings": current.copy(), "results": deepcopy(summary)})
            if index + 1 < rounds:
                try:
                    current = await self._invoke(self.reflect, summaries, current, tunables)
                except (ValidationError, InvalidAction, ProviderError, OSError):
                    self.calls[-1]["retained_settings"] = current.copy()
                reflections.append(deepcopy(self.calls[-1]))
        return {"settings": current, "comparisons": summaries, "reflections": reflections}

    async def _invoke(self, operation, *args):
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Capture raw model output before typed parsing or domain rejection."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise InvalidAction("numeric control does not accept tool requests")
        return response, requests
