"""Grow a population of source-code strategies by responding to Nash mixtures."""

import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError
from slick import prompt
from slick.providers import Provider, ProviderError

Payoffs = tuple[tuple[float, ...], ...]
Play = Callable[[str, str, int], Awaitable[float]]


class Proposal(BaseModel, extra="forbid", frozen=True):
    content: str = Field(min_length=1)


class CandidateRejected(ValueError):
    """The evaluator rejected generated code; consume this candidate's attempt."""


@dataclass(frozen=True)
class Mixture:
    population: tuple[str, ...]
    weights: tuple[float, ...]

    def sample(self, rng: random.Random) -> str:
        """Choose a source once, before a match; keep it for the whole match."""
        return rng.choices(self.population, weights=self.weights, k=1)[0]

    @property
    def source(self) -> str:
        """Portable selection code; the caller's engine loads the selected source."""
        return (
            f"STRATEGIES = {self.population!r}\n"
            f"WEIGHTS = {self.weights!r}\n\n"
            "def select_strategy(rng):\n"
            "    # Call once before the game; use this source for all its moves.\n"
            "    return rng.choices(STRATEGIES, weights=WEIGHTS, k=1)[0]\n"
        )


def fictitious_play(
    payoffs: Payoffs, *, iterations: int = 10000, seed: int = 0
) -> tuple[float, ...]:
    """Approximate a symmetric zero-sum equilibrium for a skew-symmetric matrix.

    Both players best respond to the other's empirical strategy simultaneously.
    Start with one observation per action, break ties with a seeded RNG, and
    average both players' empirical distributions into one population mixture.
    """
    rng = random.Random(seed)
    size = len(payoffs)
    row_counts, column_counts = [1] * size, [1] * size
    row_values = [sum(row) for row in payoffs]
    column_values = [sum(row[j] for row in payoffs) for j in range(size)]
    for _ in range(iterations):
        maximum, minimum = max(row_values), min(column_values)
        row = rng.choice([i for i, value in enumerate(row_values) if value == maximum])
        column = rng.choice([j for j, value in enumerate(column_values) if value == minimum])
        row_counts[row] += 1
        column_counts[column] += 1
        for i in range(size):
            row_values[i] += payoffs[i][column]
            column_values[i] += payoffs[row][i]
    total = 2 * (size + iterations)
    return tuple((row_counts[i] + column_counts[i]) / total for i in range(size))


@dataclass(frozen=True)
class Round:
    population: tuple[str, ...]
    payoffs: Payoffs
    mixture: Mixture
    selected: str


class _RecordingProvider:
    """Retain raw output before Slick parsing and generated-content validation."""

    def __init__(self, provider: Provider, record: dict):
        self.provider, self.record = provider, record

    async def acall(self, context, *, tools=None, tool_results=None):
        self.record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        self.record["response"] = response
        return response, requests


class LLMPSRO:
    """Own PSRO state; inject task, provider, and an isolated one-game evaluator.

    play(left_source, right_source, seed) returns -1, 0, or +1 for the left player.
    Each call must create fresh game/bot state. Higher scores mean better play.
    One instance supports one run at a time; run resets all in-memory records.
    """

    def __init__(self, task: str, provider: Provider, play: Play):
        self.task, self.provider, self.play = task, provider, play
        self.population: list[str] = []
        self.attempts: list[dict] = []
        self.history: list[Round] = []
        self.games_played = 0

    @prompt(template="initialize.j2", output_type=Proposal)
    async def initialize(self, *, generated: Proposal) -> str:
        """Generate a strategy without an opponent population."""
        if not generated.content.strip():
            raise CandidateRejected("generated source is blank")
        return generated.content

    @prompt(template="respond.j2", output_type=Proposal)
    async def respond(self, mixture: Mixture, *, generated: Proposal) -> str:
        """Generate a best-response candidate from the full mixed opponent source."""
        if not generated.content.strip():
            raise CandidateRejected("generated source is blank")
        return generated.content

    async def run(
        self,
        initial_population: Sequence[str] = (),
        *,
        population_size: int = 3,
        rounds: int = 5,
        candidates_per_round: int = 5,
        games_per_pair: int = 1000,
        fp_iterations: int = 10000,
        seed: int = 0,
        initial_attempts: int | None = None,
    ) -> tuple[str, ...]:
        """Initialize, measure the metagame, solve its mixture, and add one response.

        Each round uses exactly T generation attempts. If all fail, abort with
        incumbents intact. The returned population includes the final addition;
        the last recorded mixture describes the population before that addition.
        """
        self.population = list(initial_population)
        self.attempts, self.history, self.games_played = [], [], 0
        self.rng = random.Random(seed)
        if not self.population:
            budget = 3 * population_size if initial_attempts is None else initial_attempts
            await self._initialize_population(population_size, budget)
        for round_number in range(1, rounds + 1):
            population = tuple(self.population)
            payoffs = await self._build_payoffs(population, games_per_pair)
            weights = fictitious_play(
                payoffs, iterations=fp_iterations, seed=self.rng.getrandbits(64)
            )
            mixture = Mixture(population, weights)
            selected = await self._select_response(
                mixture, round_number, candidates_per_round, games_per_pair
            )
            self.history.append(Round(population, payoffs, mixture, selected))
            self.population.append(selected)
        return tuple(self.population)

    async def _initialize_population(self, size: int, budget: int) -> None:
        for _ in range(budget):
            candidate = await self._attempt(0)
            if candidate is not None:
                self.population.append(candidate["content"])
            if len(self.population) == size:
                return
        raise RuntimeError("could not initialize the population; inspect attempts")

    async def _play(self, left: str, right: str, seed: int) -> float:
        self.games_played += 1
        outcome = await self.play(left, right, seed)
        if outcome not in (-1, 0, 1):
            raise CandidateRejected("game outcome must be -1, 0, or +1")
        return float(outcome)

    async def _build_payoffs(self, population: tuple[str, ...], games: int) -> Payoffs:
        # Measure all ordered pairs, including self-play, as in Algorithm 1.
        scores = []
        for left in population:
            row = []
            for right in population:
                total = 0.0
                for _ in range(games):
                    total += await self._play(left, right, self.rng.getrandbits(64))
                row.append(total / games)
            scores.append(row)
        # Average both seat assignments and remove finite-sample asymmetry.
        return tuple(
            tuple((scores[i][j] - scores[j][i]) / 2 for j in range(len(population)))
            for i in range(len(population))
        )

    async def _select_response(
        self, mixture: Mixture, round_number: int, candidates: int, games: int
    ) -> str:
        # Share opponent samples and game seeds across candidates for fair comparison.
        schedule = [
            (mixture.sample(self.rng), self.rng.getrandbits(64), i % 2 == 0) for i in range(games)
        ]
        best = None
        for _ in range(candidates):
            candidate = await self._attempt(round_number, mixture, schedule)
            if candidate is not None and (best is None or candidate["win_rate"] > best["win_rate"]):
                best = candidate
        if best is None:
            raise RuntimeError("no valid response candidate; inspect attempts")
        return best["content"]

    async def _attempt(
        self,
        round_number: int,
        mixture: Mixture | None = None,
        schedule: Sequence[tuple[str, int, bool]] = (),
    ) -> dict | None:
        record = {"round": round_number, "id": len(self.attempts) + 1}
        self.attempts.append(record)
        provider = _RecordingProvider(self.provider, record)
        try:
            if mixture is None:
                record["content"] = await self.initialize(provider=provider)
            else:
                record["content"] = await self.respond(mixture, provider=provider)
                wins, total = 0, 0.0
                for opponent, seed, first in schedule:
                    left, right = (record["content"], opponent)
                    if not first:
                        left, right = right, left
                    outcome = await self._play(left, right, seed)
                    score = outcome if first else -outcome
                    wins += score > 0
                    total += score
                record["win_rate"] = wins / len(schedule)
                record["mean_score"] = total / len(schedule)
            return record
        except (ValidationError, CandidateRejected, ProviderError, TimeoutError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            return None
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
