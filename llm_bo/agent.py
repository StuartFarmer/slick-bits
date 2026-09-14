"""LLAMBO's outcome-conditioned sampling and discriminative surrogate EI."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from scipy.stats import norm
from slick import prompt
from slick.providers import Provider


class Configurations(BaseModel):
    configurations: list[dict[str, float]]


@dataclass
class Observation:
    configuration: dict[str, float]
    value: float


class LLMBO:
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[dict[str, float]], Awaitable[float]],
        bounds: dict[str, tuple[float, float]],
    ):
        self.task, self.provider, self.evaluate, self.bounds = task, provider, evaluate, bounds

    @prompt(template="warm_start.j2", output_type=Configurations)
    async def warm_start(self, count: int, *, generated: Configurations) -> list[dict[str, float]]:
        return generated.configurations

    @prompt(template="sample.j2", output_type=Configurations)
    async def sample(
        self, observations: list[dict], target: float, count: int, *, generated: Configurations
    ) -> list[dict[str, float]]:
        return generated.configurations

    @prompt(template="predict.j2")
    async def predict(
        self, observations: list[dict], configuration: dict[str, float], *, generated: str
    ) -> float:
        match = re.fullmatch(r"\s*##\s*([-+\d.eE]+)\s*##\s*", generated)
        if not match or not math.isfinite(value := float(match[1])):
            raise ValueError(f"invalid generated numerical prediction; response={generated!r}")
        return value

    def _accept(self, proposals, existing):
        accepted = []
        keys = {tuple(sorted(point.items())) for point in existing}
        for point in proposals:
            key = tuple(sorted(point.items()))
            valid = point.keys() == self.bounds.keys() and all(
                math.isfinite(value) and self.bounds[name][0] <= value <= self.bounds[name][1]
                for name, value in point.items()
            )
            if valid and key not in keys:
                accepted.append(point)
                keys.add(key)
            else:
                self.rejections.append(
                    {"configuration": point, "reason": "duplicate" if key in keys else "bounds"}
                )
        return accepted

    async def _measure(self, point):
        self.evaluations += 1
        value = float(await self.evaluate(point.copy()))
        if not math.isfinite(value):
            raise ValueError("measured objective must be finite")
        observation = Observation(point.copy(), value)
        self.observations.append(observation)
        return observation

    def _examples(self):
        examples = [
            {"configuration": item.configuration, "value": item.value} for item in self.observations
        ]
        self.rng.shuffle(examples)
        return examples

    async def _propose(self, count, alpha, lower_is_better, attempts, jitter):
        values = [item.value for item in self.observations]
        span = max(values) - min(values)
        if span == 0:
            span = 0.1 * abs(max(values))
        best = min(values) if lower_is_better else max(values)
        target = best + (-1 if lower_is_better else 1) * alpha * span
        pool = []
        for _ in range(attempts):
            self.optimizer_calls += 1
            desired = self.rng.uniform(min(best, target), max(best, target)) if jitter else target
            generated = await self.sample(self._examples(), desired, count, provider=self.provider)
            pool.extend(
                self._accept(generated, [item.configuration for item in self.observations] + pool)
            )
            if len(pool) >= count:
                break
        return pool[:count], target

    async def _select(self, candidates, predictions, lower_is_better):
        values = [item.value for item in self.observations]
        best = min(values) if lower_is_better else max(values)
        estimates = []
        for point in candidates:
            samples = []
            for _ in range(predictions):
                self.optimizer_calls += 1
                samples.append(await self.predict(self._examples(), point, provider=self.provider))
            mean, std = float(np.mean(samples)), max(float(np.std(samples)), 1e-5)
            delta = best - mean if lower_is_better else mean - best
            ei = delta * norm.cdf(delta / std) + std * norm.pdf(delta / std)
            estimates.append({"configuration": point, "mean": mean, "std": std, "ei": float(ei)})
        return max(estimates, key=lambda item: item["ei"])["configuration"], estimates

    async def run(
        self,
        initial_configurations: Sequence[dict[str, float]] = (),
        *,
        iterations: int = 10,
        warm_start_count: int = 3,
        candidates: int = 5,
        predictions: int = 10,
        sampling_attempts: int = 3,
        alpha: float = -0.2,
        lower_is_better: bool = True,
        jitter: bool = True,
        seed: int = 0,
    ) -> dict:
        self.rng = random.Random(seed)
        self.observations, self.history, self.rejections = [], [], []
        self.evaluations = self.optimizer_calls = 0
        initial = list(initial_configurations)
        if not initial:
            self.optimizer_calls += 1
            initial = self._accept(
                await self.warm_start(warm_start_count, provider=self.provider), []
            )[:warm_start_count]
        for point in initial:
            await self._measure(point)
        for _ in range(iterations if self.observations else 0):
            pool, target = await self._propose(
                candidates, alpha, lower_is_better, sampling_attempts, jitter
            )
            if not pool:
                break
            selected, estimates = await self._select(pool, predictions, lower_is_better)
            measured = await self._measure(selected)
            self.history.append({"target": target, "estimates": estimates, "selected": measured})
        best = (
            (min if lower_is_better else max)(self.observations, key=lambda item: item.value)
            if self.observations
            else None
        )
        return {
            "best": best,
            "observations": self.observations,
            "history": self.history,
            "rejections": self.rejections,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
        }
