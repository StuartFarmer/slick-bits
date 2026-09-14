"""Co-evolve environments and numerical agents with PINSKY's DE and transfer loop."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, StringConstraints, ValidationError
from slick import Provider, prompt

Operation = Literal["remove", "add", "move"]
Mutate = Callable[[str, Operation, np.random.Generator], Awaitable[str]]
Solve = Callable[[str], Awaitable[bool]]


class EnvironmentProposal(BaseModel, extra="forbid"):
    environment: Annotated[str, StringConstraints(min_length=1)]


class CandidateRejected(ValueError):
    """A generated environment cannot be evaluated in the caller's domain."""


@dataclass(frozen=True)
class Config:
    iterations: int = 5000
    mutation_timer: int = 25
    transfer_timer: int = 10
    max_children: int = 8
    max_environments: int = 30
    mutation_rate: float = 0.8
    continuation_rate: float = 0.5
    operation_probabilities: tuple[float, float, float] = (0.25, 0.5, 0.25)
    population_size: int = 50
    de_evaluations: int = 1500
    scaling_factor: float = 0.6
    crossover_rate: float = 0.4
    lower_bound: float = -5.0
    upper_bound: float = 5.0


@dataclass(frozen=True)
class Evaluation:
    score: float
    solved: bool


@dataclass
class Pair:
    id: int
    parent_id: int | None
    born: int
    environment: str
    parameters: np.ndarray
    evaluation: Evaluation | None = None


@dataclass
class Attempt:
    id: int
    iteration: int
    parent_id: int
    environment: str | None = None
    operations: list[str] = field(default_factory=list)
    random_solved: bool | None = None
    strong_solved: bool | None = None
    accepted: bool = False
    error: str | None = None


@dataclass(frozen=True)
class Transfer:
    iteration: int
    source_id: int
    target_id: int
    previous_score: float
    score: float


@dataclass
class Result:
    active: list[Pair] = field(default_factory=list)
    retired: list[Pair] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    transfers: list[Transfer] = field(default_factory=list)
    generations: list[tuple[str, str]] = field(default_factory=list)
    evaluation_calls: int = 0
    solver_calls: int = 0


class _RecordingProvider:
    def __init__(self, provider, records):
        self.provider, self.records = provider, records

    async def acall(self, context, *, tools=None, tool_results=None):
        response, calls = await self.provider.acall(context, tools=tools, tool_results=tool_results)
        self.records.append((context, response))
        if calls:
            raise CandidateRejected("Environment mutation requires text, not tool calls")
        return response, calls


class PINSKY:
    """Own one run; callbacks own domain execution, isolation, and transport retries.

    Environments are serialized text, agents are flat real parameter vectors, and
    measured scores are maximized. Success is a separate observed boolean.
    Supply mutate for native domain edits, or a provider for Slick text edits.
    """

    def __init__(
        self,
        task: str,
        provider: Provider | None,
        evaluate: Callable[[str, np.ndarray], Awaitable[Evaluation]],
        *,
        random_solve: Solve,
        strong_solve: Solve,
        mutate: Mutate | None = None,
        seed: int = 0,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.random_solve, self.strong_solve = random_solve, strong_solve
        self.mutate = mutate
        self.rng = np.random.default_rng(seed)
        self.result = Result()

    @prompt(template="remove.j2", output_type=EnvironmentProposal)
    async def remove(self, environment: str, *, generated: EnvironmentProposal) -> str:
        """Remove one mutable element from an environment."""
        return self.check_environment(generated.environment)

    @prompt(template="add.j2", output_type=EnvironmentProposal)
    async def add(self, environment: str, *, generated: EnvironmentProposal) -> str:
        """Add one mutable element to an environment."""
        return self.check_environment(generated.environment)

    @prompt(template="move.j2", output_type=EnvironmentProposal)
    async def move(self, environment: str, *, generated: EnvironmentProposal) -> str:
        """Reposition one mutable element in an environment."""
        return self.check_environment(generated.environment)

    @staticmethod
    def check_environment(environment: str) -> str:
        if not environment.strip():
            raise CandidateRejected("Generated environment is blank")
        return environment  # Preserve whitespace in serialized artifacts.

    async def edit(self, environment: str, operation: Operation) -> str:
        if self.mutate is not None:
            return self.check_environment(await self.mutate(environment, operation, self.rng))
        provider = _RecordingProvider(self.provider, self.result.generations)
        try:
            return await getattr(self, operation)(environment, provider=provider)
        except ValidationError as exc:
            raise CandidateRejected(f"Invalid environment response: {exc}") from exc

    async def assess(self, environment: str, parameters: np.ndarray) -> Evaluation:
        self.result.evaluation_calls += 1
        evaluation = await self.evaluate(environment, parameters.copy())
        if not np.isfinite(evaluation.score):
            raise ValueError("Measured fitness must be finite")
        return evaluation

    async def reproduce(self, iteration: int, config: Config) -> None:
        parents = tuple(self.result.active)
        for _ in range(config.max_children):
            parent = parents[self.rng.integers(len(parents))]
            attempt = Attempt(len(self.result.attempts) + 1, iteration, parent.id)
            self.result.attempts.append(attempt)
            attempt.environment = parent.environment
            try:
                if self.rng.random() < config.mutation_rate:
                    while True:
                        operation = str(
                            self.rng.choice(
                                ["remove", "add", "move"], p=config.operation_probabilities
                            )
                        )
                        attempt.operations.append(operation)
                        attempt.environment = await self.edit(attempt.environment, operation)
                        if self.rng.random() >= config.continuation_rate:
                            break
                self.result.solver_calls += 1
                attempt.strong_solved = await self.strong_solve(attempt.environment)
                self.result.solver_calls += 1
                attempt.random_solved = await self.random_solve(attempt.environment)
            except CandidateRejected as exc:
                attempt.error = str(exc)
                continue
            if attempt.random_solved or not attempt.strong_solved:
                continue
            attempt.accepted = True
            self.result.active.append(
                Pair(
                    attempt.id, parent.id, iteration, attempt.environment, parent.parameters.copy()
                )
            )
        excess = max(0, len(self.result.active) - config.max_environments)
        self.result.retired.extend(self.result.active[:excess])
        del self.result.active[:excess]

    async def optimize(self, pair: Pair, config: Config) -> None:
        # Upstream restarts DE each phase, keeping only the paired incumbent.
        population = self.rng.uniform(-1, 1, (config.population_size, pair.parameters.size))
        population[0] = pair.parameters
        population = np.clip(population, config.lower_bound, config.upper_bound)
        scores = np.array(
            [(await self.assess(pair.environment, parameters)).score for parameters in population],
            dtype=float,
        )
        for start in range(0, config.de_evaluations, config.population_size):
            count = min(config.population_size, config.de_evaluations - start)
            trials, trial_scores = [], []
            for target in range(count):
                donors = [index for index in range(config.population_size) if index != target]
                a, b, c = self.rng.choice(donors, size=3, replace=False)
                donor = np.clip(
                    population[a] + config.scaling_factor * (population[b] - population[c]),
                    config.lower_bound,
                    config.upper_bound,
                )
                forced = self.rng.integers(pair.parameters.size)
                mask = self.rng.random(pair.parameters.size) < config.crossover_rate
                mask[forced] = True
                trial = np.where(mask, donor, population[target])
                trials.append(trial)
                trial_scores.append((await self.assess(pair.environment, trial)).score)
            # Deferred replacement preserves DE/rand/1/bin's generation snapshot.
            for target, (trial, score) in enumerate(zip(trials, trial_scores)):
                if score >= scores[target]:
                    population[target], scores[target] = trial, score
        pair.parameters = population[np.argmax(scores)].copy()

    async def optimize_population(self, config: Config) -> None:
        # ponytail: serial evaluations; add isolated workers if rollout throughput matters.
        for pair in self.result.active:
            await self.optimize(pair, config)
        for pair in self.result.active:
            pair.evaluation = await self.assess(pair.environment, pair.parameters)

    async def transfer(self, iteration: int) -> None:
        pairs = self.result.active
        agents = [pair.parameters.copy() for pair in pairs]
        measurements = [
            [await self.assess(target.environment, agent) for agent in agents] for target in pairs
        ]
        for target_index, target in enumerate(pairs):
            row = measurements[target_index]
            winner = target_index
            for source_index, evaluation in enumerate(row):
                if evaluation.score > row[winner].score:
                    winner = source_index
            if winner != target_index:
                target.parameters = agents[winner].copy()
                self.result.transfers.append(
                    Transfer(
                        iteration,
                        pairs[winner].id,
                        target.id,
                        row[target_index].score,
                        row[winner].score,
                    )
                )
            target.evaluation = row[winner]

    async def run(
        self, environment: str, parameters: np.ndarray, *, config: Config = Config()
    ) -> Result:
        initial = Pair(0, None, 0, environment, np.array(parameters, dtype=float, copy=True))
        initial.evaluation = await self.assess(environment, initial.parameters)
        self.result.active.append(initial)
        for iteration in range(config.iterations):
            if iteration % config.mutation_timer == 0:
                await self.reproduce(iteration, config)
            await self.optimize_population(config)
            if (iteration + 1) % config.transfer_timer == 0:
                await self.transfer(iteration)
        return self.result
