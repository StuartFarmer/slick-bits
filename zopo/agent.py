"""Optimize a discrete instruction pool using ZOPO's local GP-NTK ascent."""

from collections.abc import Awaitable, Callable, Sequence

import numpy as np
from slick import prompt
from slick.providers import Provider


class ZOPO:
    """Own local GP fitting, Adam ascent, discrete projection and uncertainty exploration.

    embed returns fixed instruction vectors. features(x) supplies parameter
    Jacobians of a fixed neural network and their input derivatives, shaped
    (n, p) and (n, p, d). These are model primitives; GP regression and its
    input gradients are computed here. Scores are finite and higher is better.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        embed: Callable[[Sequence[str]], Awaitable[np.ndarray]],
        features: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.embed, self.features = embed, features

    @prompt(template="induce.j2")
    async def induce(self, examples: Sequence[str], *, generated: str) -> str:
        """Generate an instruction for the fixed embedding search pool."""
        if not generated.strip():
            raise ValueError(f"empty instruction; response={generated!r}")
        return generated.strip()

    async def _measure(self, arm, phase):
        self.evaluations += 1
        score = float(await self.evaluate(self.prompts[arm]))
        if not np.isfinite(score):
            raise ValueError("evaluation score must be finite")
        self.queried.add(arm)
        record = {"arm": arm, "prompt": self.prompts[arm], "score": score, "phase": phase}
        self.records.append(record)
        return record

    def _features(self, points):
        features, derivatives = self.features(points)
        features, derivatives = np.asarray(features), np.asarray(derivatives)
        if not np.isfinite(features).all() or not np.isfinite(derivatives).all():
            raise ValueError("neural features and input derivatives must be finite")
        return features, derivatives

    def _fit(self, center, gp_queries, noise):
        indices = np.array([r["arm"] for r in self.records])
        order = np.argsort(np.linalg.norm(self.embeddings[indices] - center, axis=1))[:gp_queries]
        training = self.embeddings[indices[order]]
        values = np.array([self.records[i]["score"] for i in order])
        self.lower, upper = training.min(axis=0), training.max(axis=0)
        constant = upper == self.lower
        self.lower = np.where(constant, 0, self.lower)
        upper = np.where(constant, 1, upper)
        self.scale = upper - self.lower
        self.train_features, _ = self._features((training - self.lower) / self.scale)
        kernel = self.train_features @ self.train_features.T + noise**2 * np.eye(len(training))
        self.factor = np.linalg.cholesky(kernel)
        self.alpha = np.linalg.solve(self.factor.T, np.linalg.solve(self.factor, values))
        self.fits += 1

    def posterior(self, point: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Return the local GP mean, variance and gradient in original embedding units."""
        features, derivatives = self._features(((point - self.lower) / self.scale)[None, :])
        cross = self.train_features @ features[0]
        mean = float(cross @ self.alpha)
        projected = np.linalg.solve(self.factor, cross)
        variance = abs(float(features[0] @ features[0] - projected @ projected))
        gradient = derivatives[0].T @ self.train_features.T @ self.alpha / self.scale
        return mean, variance, gradient

    def _nearest(self, point, count):
        available = [i for i in range(len(self.prompts)) if i not in self.queried]
        return sorted(available, key=lambda i: np.linalg.norm(self.embeddings[i] - point))[:count]

    async def _ascend(
        self,
        center,
        max_evaluations,
        gp_queries,
        noise,
        learning_rate,
        max_steps,
        neighbors,
        uncertainty_threshold,
        uncertainty_patience,
    ):
        self._fit(center, gp_queries, noise)
        point = center.copy()
        chosen = None
        steps = 0
        while self.evaluations < max_evaluations:
            _, variance, gradient = self.posterior(point)
            if steps == 0:
                if variance > uncertainty_threshold:
                    self.uncertain_rounds += 1
                elif variance < uncertainty_threshold:
                    self.uncertain_rounds = 0
            explore = not np.any(gradient) or self.uncertain_rounds >= uncertainty_patience
            if explore:
                local = self._nearest(center, min(neighbors, max_evaluations - self.evaluations))
                if not local:
                    return None
                for arm in local:
                    await self._measure(arm, "exploration")
                self.uncertain_rounds = self.stagnation = 0
                self._fit(center, gp_queries, noise)
                continue
            if steps >= max_steps or (steps > 0 and variance >= uncertainty_threshold):
                break
            self.adam_step += 1
            self.first = 0.9 * self.first + 0.1 * gradient
            self.second = 0.999 * self.second + 0.001 * gradient**2
            update = (
                learning_rate
                * (self.first / (1 - 0.9**self.adam_step))
                / (np.sqrt(self.second / (1 - 0.999**self.adam_step)) + 1e-8)
            )
            closest = self._nearest(point + update, 1)
            if not closest:
                return None
            chosen = closest[0]
            self.steps.append(
                {"from": point.copy(), "update": update.copy(), "arm": chosen, "variance": variance}
            )
            point = self.embeddings[chosen].copy()
            steps += 1
            if variance >= uncertainty_threshold:
                break
        return chosen

    async def run(
        self,
        initial_prompts: Sequence[str],
        *,
        examples: Sequence[str] = (),
        additional_candidates: int = 0,
        initial_samples: int = 5,
        max_evaluations: int = 30,
        gp_queries: int = 20,
        noise: float = 0.01,
        learning_rate: float = 0.1,
        max_steps: int = 2,
        neighbors: int = 3,
        uncertainty_threshold: float = 0.1,
        uncertainty_patience: int = 2,
        tolerance: int = 3,
        seed: int = 0,
    ) -> dict:
        """Seed a fixed pool, ascend local posteriors and query nearest unseen instructions.

        The evaluation budget includes initialization and exploration. Exhausting
        the finite pool terminates search, including zero-gradient exploration.
        """
        self.optimizer_calls = self.evaluations = self.fits = self.uncertain_rounds = 0
        self.stagnation = self.adam_step = 0
        self.records, self.steps, self.queried = [], [], set()
        candidates = list(initial_prompts)
        for _ in range(additional_candidates):
            self.optimizer_calls += 1
            candidates.append(await self.induce(examples, provider=self.provider))
        self.prompts = list(dict.fromkeys(candidates))
        self.embeddings = np.asarray(await self.embed(self.prompts))
        if not np.isfinite(self.embeddings).all():
            raise ValueError("instruction embeddings must be finite")
        self.first = np.zeros(self.embeddings.shape[1])
        self.second = np.zeros_like(self.first)
        rng = np.random.default_rng(seed)
        for arm in rng.choice(
            len(self.prompts), min(initial_samples, max_evaluations), replace=False
        ):
            await self._measure(int(arm), "initial")
        best = max(self.records, key=lambda r: r["score"])
        latest = best
        while self.evaluations < max_evaluations and len(self.queried) < len(self.prompts):
            if latest["score"] > best["score"]:
                best, self.stagnation = latest, 0
            else:
                self.stagnation += 1
            if self.stagnation > tolerance:
                latest = max(self.records, key=lambda r: r["score"])
                self.first.fill(0)
                self.second.fill(0)
                self.stagnation = self.adam_step = 0
            arm = await self._ascend(
                self.embeddings[latest["arm"]],
                max_evaluations,
                gp_queries,
                noise,
                learning_rate,
                max_steps,
                neighbors,
                uncertainty_threshold,
                uncertainty_patience,
            )
            if arm is not None and arm not in self.queried and self.evaluations < max_evaluations:
                await self._measure(arm, "ascent")
            latest = self.records[-1]
        return {
            "best": max(self.records, key=lambda r: r["score"]),
            "records": self.records,
            "steps": self.steps,
            "evaluations": self.evaluations,
            "fits": self.fits,
            "optimizer_calls": self.optimizer_calls,
        }
