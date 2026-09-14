"""In-context numerical distributions, inverse filtering, and discrete acquisition."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from slick import prompt
from slick.providers import Provider


def cosine(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def mmr(vectors, query, count, diversity):
    vectors = cosine(vectors)
    query = cosine(np.asarray(query)[None])[0]
    relevance = vectors @ query
    selected = [int(np.argmax(relevance))] if count else []
    while len(selected) < min(count, len(vectors)):
        redundancy = (vectors @ vectors[selected].T).max(axis=1)
        scores = diversity * relevance - (1 - diversity) * redundancy
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
    return selected


@dataclass
class Observation:
    candidate: str
    value: float


class BOLIFT:
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        embed: Callable[[Sequence[str]], Awaitable[np.ndarray]],
    ):
        self.task, self.provider, self.evaluate, self.embed = task, provider, evaluate, embed

    @prompt(template="predict.j2")
    async def predict(
        self, candidate: str, examples: list[Observation], *, generated: str
    ) -> float:
        value = float(generated.split("###")[0].strip())
        if not math.isfinite(value):
            raise ValueError(f"nonfinite generated prediction; response={generated!r}")
        return value

    @prompt(template="inverse.j2")
    async def inverse(self, target: float, examples: list[Observation], *, generated: str) -> str:
        text = generated.split("###")[0].strip()
        if not text:
            raise ValueError(f"empty generated inverse candidate; response={generated!r}")
        return text

    async def _vectors(self, texts):
        self.embedding_calls += 1
        vectors = np.asarray(await self.embed(texts))
        if not np.isfinite(vectors).all():
            raise ValueError("measured embeddings must be finite")
        return vectors

    async def _filter(self, pool, count, random_count, diversity, example_count):
        if count == 0:
            return self.rng.sample(pool, min(random_count, len(pool))) if random_count else pool
        if count + random_count >= len(pool):
            return pool
        target = max(item.value for item in self.observations) * self.rng.gauss(1.2, 0.05)
        examples = sorted(self.observations, key=lambda item: abs(item.value - target))[
            :example_count
        ]
        self.optimizer_calls += 1
        ideal = await self.inverse(target, examples, provider=self.provider)
        vectors = await self._vectors([ideal, *pool])
        indices = mmr(vectors[1:], vectors[0], count, diversity)
        selected = [pool[i] for i in indices]
        remaining = [item for item in pool if item not in selected]
        selected.extend(self.rng.sample(remaining, min(random_count, len(remaining))))
        return selected

    async def _acquire(self, pool, predictions, example_count, acquisition, exploration, diversity):
        vectors = await self._vectors([item.candidate for item in self.observations] + pool)
        observed_vectors = vectors[: len(self.observations)]
        best = max(item.value for item in self.observations)
        estimates = []
        for index, candidate in enumerate(pool):
            selection = mmr(
                observed_vectors, vectors[len(self.observations) + index], example_count, diversity
            )
            examples = [self.observations[i] for i in selection]
            samples = []
            for _ in range(predictions):
                self.optimizer_calls += 1
                samples.append(await self.predict(candidate, examples, provider=self.provider))
            samples = np.array(samples)
            if acquisition == "ei":
                value = float(np.maximum(samples - best, 0).mean())
            elif acquisition == "pi":
                value = float((samples > best).mean())
            else:
                value = float(samples.mean() + exploration * samples.std())
            estimates.append({"candidate": candidate, "samples": samples, "acquisition": value})
        return max(estimates, key=lambda item: item["acquisition"])["candidate"], estimates

    async def run(
        self,
        candidates: Sequence[str],
        initial_observations: Sequence[Observation] = (),
        *,
        iterations: int = 10,
        predictions: int = 5,
        example_count: int = 5,
        inverse_count: int = 16,
        random_count: int = 0,
        diversity: float = 0.5,
        acquisition: str = "ucb",
        exploration: float = 0.5,
        seed: int = 0,
    ) -> dict:
        self.rng = random.Random(seed)
        self.observations, self.history = list(initial_observations), []
        self.optimizer_calls = self.embedding_calls = self.evaluations = 0
        known = {item.candidate for item in self.observations}
        pool = [item for item in dict.fromkeys(candidates) if item not in known]
        for _ in range(min(iterations, len(pool))):
            if len(self.observations) < 2:
                selected, estimates = self.rng.choice(pool), []
            else:
                filtered = await self._filter(
                    pool, inverse_count, random_count, diversity, example_count
                )
                selected, estimates = await self._acquire(
                    filtered, predictions, example_count, acquisition, exploration, diversity
                )
            self.evaluations += 1
            value = float(await self.evaluate(selected))
            if not math.isfinite(value):
                raise ValueError("measured objective must be finite")
            observation = Observation(selected, value)
            self.observations.append(observation)
            pool.remove(selected)
            self.history.append({"estimates": estimates, "selected": observation})
        return {
            "best": max(self.observations, key=lambda item: item.value)
            if self.observations
            else None,
            "observations": self.observations,
            "history": self.history,
            "optimizer_calls": self.optimizer_calls,
            "embedding_calls": self.embedding_calls,
            "evaluations": self.evaluations,
        }
