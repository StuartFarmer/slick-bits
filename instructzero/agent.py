"""Optimize soft-prefix latents with InstructZero's instruction-coupled GP and EI."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import qmc
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Evaluation:
    """A scalar objective and aligned per-example performance vector."""

    score: float
    behavior: np.ndarray


def matern52(left: np.ndarray, right: np.ndarray, length: float) -> np.ndarray:
    distance = np.linalg.norm((left[:, None, :] - right[None, :, :]) / length, axis=-1)
    scaled = np.sqrt(5) * distance
    return (1 + scaled + scaled**2 / 3) * np.exp(-scaled)


class InstructZero:
    """Own projection, coupled-kernel fitting, expected improvement and observations.

    condition(provider, projected_prefix) returns a provider that truly prepends
    soft-token embeddings; it must not stringify the vector into instructions.
    evaluate returns higher finite scores and a fixed-order behavior vector.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        condition: Callable[[Provider, np.ndarray], Provider],
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.condition = condition

    @prompt(template="induce.j2")
    async def induce(self, demonstrations: Sequence[str], *, generated: str) -> str:
        """Decode a hard instruction from the externally conditioned soft prefix."""
        if not generated.strip():
            raise ValueError(f"empty induced instruction; response={generated!r}")
        return generated.strip()

    async def _observe(self, latent):
        self.optimizer_calls += 1
        instruction = await self.induce(
            self.demonstrations, provider=self.condition(self.provider, self.projection @ latent)
        )
        if instruction in self.cache:
            self.cache_hits += 1
            measured = self.cache[instruction]
        else:
            self.evaluations += 1
            measured = await self.evaluate(instruction)
            if not np.isfinite(measured.score) or not np.isfinite(measured.behavior).all():
                raise ValueError("evaluation score and behavior must be finite")
            measured = Evaluation(measured.score, np.asarray(measured.behavior).copy())
            self.cache[instruction] = measured
        self.observations.append(
            {
                "latent": latent.copy(),
                "prompt": instruction,
                "score": measured.score,
                "behavior": np.asarray(measured.behavior).copy(),
            }
        )

    def _coupling(self, log_parameters):
        latent_length, behavior_length, scale, noise = np.exp(log_parameters)
        latent_kernel = matern52(self.x, self.x, latent_length)
        inverse = np.linalg.solve(latent_kernel + 1e-4 * np.eye(len(self.x)), np.eye(len(self.x)))
        behavior_kernel = matern52(self.behavior, self.behavior, behavior_length)
        coupling = inverse @ behavior_kernel @ inverse
        covariance = scale * latent_kernel @ coupling @ latent_kernel.T
        covariance += noise * np.eye(len(self.x))
        return coupling, covariance

    def _fit(self):
        self.x = np.array([o["latent"] for o in self.observations])
        self.behavior = np.array([o["behavior"] for o in self.observations])
        values = np.array([o["score"] for o in self.observations])
        self.y = (values - values.mean()) / (values.std(ddof=1) + 1e-9)

        def negative_log_likelihood(parameters):
            _, covariance = self._coupling(parameters)
            factor = np.linalg.cholesky(covariance)
            residual = np.linalg.solve(factor, self.y)
            return np.log(np.diag(factor)).sum() + residual @ residual / 2

        fit = minimize(
            negative_log_likelihood,
            np.log([0.5, 0.5, 1.0, 0.01]),
            method="L-BFGS-B",
            bounds=[(-5, 5), (-5, 5), (-5, 5), (-12, 0)],
        )
        if not np.isfinite(fit.fun):
            raise ValueError("GP fit produced a nonfinite likelihood")
        self.parameters = fit.x
        self.coupling, covariance = self._coupling(self.parameters)
        self.factor = np.linalg.cholesky(covariance)
        self.alpha = np.linalg.solve(self.factor.T, np.linalg.solve(self.factor, self.y))
        self.fit_history.append({"objective": float(fit.fun), "converged": bool(fit.success)})

    def expected_improvement(self, candidates: np.ndarray) -> np.ndarray:
        """Compute analytic EI using the fitted instruction-coupled posterior."""
        length, _, scale, _ = np.exp(self.parameters)
        test_to_train = matern52(candidates, self.x, length)
        train_kernel = matern52(self.x, self.x, length)
        covariance = scale * test_to_train @ self.coupling @ train_kernel.T
        mean = covariance @ self.alpha
        variance = scale * np.sum((test_to_train @ self.coupling) * test_to_train, axis=1)
        projected = np.linalg.solve(self.factor, covariance.T)
        deviation = np.sqrt(np.maximum(variance - np.sum(projected**2, axis=0), 0))
        improvement = mean - self.y.max()
        z = np.divide(improvement, deviation, out=np.zeros_like(improvement), where=deviation > 0)
        ei = improvement * ndtr(z) + deviation * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
        return np.where(deviation > 0, ei, np.maximum(improvement, 0))

    def _acquire(self, batch_size, lower, upper):
        starting = np.argsort(-np.array([o["score"] for o in self.observations]))[:batch_size]
        proposals = []
        for index in starting:
            fit = minimize(
                lambda z: -float(self.expected_improvement(z[None, :])[0]),
                np.clip(self.x[index], lower, upper),
                method="L-BFGS-B",
                bounds=[(lower, upper)] * self.x.shape[1],
            )
            if not np.isfinite(fit.x).all() or not np.isfinite(fit.fun):
                raise ValueError("acquisition optimizer produced nonfinite output")
            proposals.append((fit.x, -float(fit.fun)))
        return sorted(proposals, key=lambda p: p[1], reverse=True)

    async def run(
        self,
        demonstrations: Sequence[str],
        projection: np.ndarray,
        *,
        initial_samples: int = 10,
        iterations: int = 10,
        batch_size: int = 3,
        lower: float = -1,
        upper: float = 1,
        seed: int = 0,
    ) -> dict:
        """Sample Sobol seeds, fit a coupled GP, maximize EI and decode new prefixes."""
        self.demonstrations, self.projection = demonstrations, projection
        self.observations, self.fit_history, self.cache = [], [], {}
        self.optimizer_calls = self.evaluations = self.cache_hits = 0
        sampler = qmc.Sobol(d=projection.shape[1], scramble=True, seed=seed)
        points = sampler.random_base2(int(np.ceil(np.log2(initial_samples))))[:initial_samples]
        for latent in points:
            await self._observe(latent)
        acquisitions = []
        for _ in range(iterations):
            self._fit()
            proposals = self._acquire(batch_size, lower, upper)
            acquisitions.append([value for _, value in proposals])
            for latent, _ in proposals:
                await self._observe(latent)
        return {
            "best": max(self.observations, key=lambda p: p["score"]),
            "observations": self.observations,
            "acquisitions": acquisitions,
            "fits": self.fit_history,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "cache_hits": self.cache_hits,
        }
