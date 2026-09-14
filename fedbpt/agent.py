"""Federated BBT with client CMA searches and variance-scaled server CMA updates."""

from collections.abc import Awaitable, Callable, Sequence

import numpy as np

from bbt.cma import CMA


class FedBPT:
    def __init__(
        self,
        task: str,
        evaluate: Callable[[str, np.ndarray, bool], Awaitable[float]],
        clients: Sequence[str],
        initial_prompt: np.ndarray,
        projection: np.ndarray,
    ):
        self.task, self.evaluate, self.clients = task, evaluate, clients
        self.initial_prompt = np.array(initial_prompt, dtype=float, copy=True)
        self.projection = projection

    async def _fitness(self, client, latent, regularize):
        prompt = self.initial_prompt + (self.projection @ latent).reshape(self.initial_prompt.shape)
        self.evaluations += 1
        original = float(await self.evaluate(client, prompt, False))
        if regularize:
            self.evaluations += 1
            perturbed = float(await self.evaluate(client, prompt, True))
            value = original / perturbed
        else:
            value = original
        if not np.isfinite(value):
            raise ValueError("measured client fitness must be finite")
        return value

    async def _client(self, client, population, local_steps, regularize, rng):
        local = CMA(
            self.server.mean, self.server.sigma, population, rng, covariance=self.server.covariance
        )
        sigmas = []
        for _ in range(local_steps):
            sigmas.append(local.sigma)
            solutions = local.ask()
            losses = [await self._fitness(client, solution, regularize) for solution in solutions]
            local.tell(solutions, np.array(losses))
        loss = await self._fitness(client, local.mean, regularize)
        return {
            "client": client,
            "mean": local.mean.copy(),
            "fitness": loss,
            "sigmas": sigmas,
            "parents": local.parents,
        }

    async def run(
        self,
        *,
        rounds: int = 20,
        clients_per_round: int | None = None,
        local_steps: int = 8,
        population: int = 20,
        sigma: float = 1,
        regularize: bool = True,
        seed: int = 0,
    ) -> dict:
        rng = np.random.default_rng(seed)
        count = len(self.clients) if clients_per_round is None else clients_per_round
        self.server = CMA(np.zeros(self.projection.shape[1]), sigma, count, rng, parents=count)
        self.evaluations, self.history = 0, []
        for _ in range(rounds):
            chosen = rng.choice(self.clients, size=count, replace=False)
            reports = [
                await self._client(client, population, local_steps, regularize, rng)
                for client in chosen
            ]
            local_sigma = self.server.sigma
            aggregate_sigma = np.sqrt(
                sum(np.square(report["sigmas"]).sum() / report["parents"] for report in reports)
                / count
            )
            self.server.sigma = float(aggregate_sigma)
            self.server.tell(
                np.array([report["mean"] for report in reports]),
                np.array([report["fitness"] for report in reports]),
            )
            ratio = self.server.sigma / aggregate_sigma
            self.server.sigma = float(ratio * local_sigma)
            self.history.append(
                {
                    "reports": reports,
                    "aggregate_sigma": aggregate_sigma,
                    "adaptation_ratio": ratio,
                    "sigma": self.server.sigma,
                    "mean": self.server.mean.copy(),
                }
            )
        prompt = self.initial_prompt + (self.projection @ self.server.mean).reshape(
            self.initial_prompt.shape
        )
        return {
            "prompt": prompt,
            "latent": self.server.mean.copy(),
            "history": self.history,
            "evaluations": self.evaluations,
        }
