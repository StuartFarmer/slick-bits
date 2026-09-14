"""Positive-weight full-covariance CMA-ES, shared by the BBT family."""

import math

import numpy as np


class CMA:
    """Minimize measured fitness using rank-mu adaptation and cumulative paths.

    This implements the standard positive-weight algorithm, not pycma's active
    negative-weight covariance update, bound transforms, or automatic restarts.
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        population: int,
        rng: np.random.Generator,
        *,
        parents: int | None = None,
        covariance: np.ndarray | None = None,
    ):
        self.mean = np.array(mean, dtype=float, copy=True)
        self.sigma, self.population, self.rng = sigma, population, rng
        self.dimension = n = self.mean.size
        self.parents = mu = population // 2 if parents is None else parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1 / np.square(self.weights).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff)
        )
        self.damping = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.expected_norm = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.covariance = np.eye(n) if covariance is None else np.array(covariance, copy=True)
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.generation = 0
        self.best, self.best_loss = self.mean.copy(), math.inf

    def _eigen(self):
        values, vectors = np.linalg.eigh((self.covariance + self.covariance.T) / 2)
        # Positive roundoff protection; this does not reset learned covariance.
        values = np.maximum(values, np.finfo(float).eps * max(1, values.max()))
        return values, vectors

    def ask(self) -> np.ndarray:
        values, vectors = self._eigen()
        noise = self.rng.standard_normal((self.population, self.dimension))
        return self.mean + self.sigma * (noise * np.sqrt(values)) @ vectors.T

    def tell(self, solutions: np.ndarray, losses: np.ndarray) -> None:
        solutions, losses = np.asarray(solutions), np.asarray(losses)
        if not np.isfinite(losses).all() or not np.isfinite(solutions).all():
            raise ValueError("CMA measurements and evaluated solutions must be finite")
        order = np.argsort(losses, kind="stable")
        if losses[order[0]] < self.best_loss:
            self.best_loss = float(losses[order[0]])
            self.best = solutions[order[0]].copy()
        old_mean = self.mean.copy()
        steps = (solutions[order[: self.parents]] - old_mean) / self.sigma
        direction = self.weights @ steps
        self.mean = old_mean + self.sigma * direction
        values, vectors = self._eigen()
        whitened = vectors @ ((vectors.T @ direction) / np.sqrt(values))
        self.ps = (1 - self.cs) * self.ps + math.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * whitened
        self.generation += 1
        normalized = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
        hsig = float(normalized / self.expected_norm < 1.4 + 2 / (self.dimension + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * direction
        rank_mu = np.einsum("i,ij,ik->jk", self.weights, steps, steps)
        retention = 1 - self.c1 - self.cmu + self.c1 * (1 - hsig) * self.cc * (2 - self.cc)
        self.covariance = (
            retention * self.covariance + self.c1 * np.outer(self.pc, self.pc) + self.cmu * rank_mu
        )
        self.sigma *= math.exp(
            self.cs / self.damping * (np.linalg.norm(self.ps) / self.expected_norm - 1)
        )
