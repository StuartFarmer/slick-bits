"""Evolve instructions using major-error feedback, crossover, and guided mutation."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class ErrorCase:
    """Caller-reported failure; expected includes any reference explanation."""

    input: str
    expected: str
    predicted: str
    justification: str = ""


@dataclass(frozen=True)
class Evaluation:
    """Training fitness (finite, nonnegative, higher is better) and all failures."""

    score: float
    errors: tuple[ErrorCase, ...] = ()


@dataclass(frozen=True)
class Candidate:
    prompt: str
    score: float
    examples: tuple[ErrorCase, ...]
    feedback: str


class GreenTEA:
    """Own sequential evolution; evaluation, embeddings, and providers are injected.

    Evaluations and feedback are cached by instruction for one run, so use a
    fixed training set and repeatable scoring. Errors propagate without retries;
    counters and raw generated responses remain available after failure.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        topics: Callable[[Sequence[str], random.Random], Sequence[int]],
        *,
        analyzer_provider: Provider | None = None,
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.topics = topics
        self.analyzer_provider = provider if analyzer_provider is None else analyzer_provider
        self.responses = []

    def _parse(self, operation: str, raw: str, *tags: str) -> list[str]:
        self.responses.append({"operation": operation, "raw": raw})
        values = []
        for tag in tags:
            opening, closing = f"<{tag}>", f"</{tag}>"
            match = re.search(re.escape(opening) + r"(.*?)" + re.escape(closing), raw, re.DOTALL)
            if (
                raw.count(opening) != 1
                or raw.count(closing) != 1
                or match is None
                or not match.group(1).strip()
            ):
                raise ValueError(f"expected one nonempty {opening}...{closing}; response={raw!r}")
            values.append(match.group(1).strip())
        return values

    @prompt(template="analyze.j2")
    async def analyze(
        self, instruction: str, examples: Sequence[ErrorCase], *, generated: str
    ) -> str:
        """Check both feedback sections while retaining the paper's tagged prose."""
        self._parse("analyze", generated, "erroranalysis", "suggestion")
        return generated

    @prompt(template="crossover.j2")
    async def crossover(self, parent1: str, parent2: str, *, generated: str) -> str:
        """Combine traits from two parents into one intermediate instruction."""
        return self._parse("crossover", generated, "ChildPrompt")[0]

    @prompt(template="mutate.j2")
    async def mutate(
        self,
        child: str,
        parent1: str,
        parent2: str,
        examples1: Sequence[ErrorCase],
        examples2: Sequence[ErrorCase],
        feedback1: str,
        feedback2: str,
        *,
        generated: str,
    ) -> str:
        """Apply both parents' feedback to the intermediate instruction."""
        return self._parse("mutate", generated, "OptimizedPrompt")[0]

    async def _assess(self, instruction: str) -> Candidate:
        if instruction in self.cache:
            self.cache_hits += 1
            return self.cache[instruction]
        self.evaluations += 1
        evaluation = await self.evaluate(instruction)
        score = float(evaluation.score)
        if not math.isfinite(score) or score < 0:
            raise ValueError("fitness must be finite and nonnegative (higher is better)")
        examples, feedback = (), ""
        if evaluation.errors:
            indices = self.topics([case.expected for case in evaluation.errors], self.rng)
            examples = tuple(evaluation.errors[i] for i in indices)
            if examples:
                self.optimizer_calls += 1
                feedback = await self.analyze(
                    instruction, examples, provider=self.analyzer_provider
                )
        candidate = Candidate(instruction, score, examples, feedback)
        self.cache[instruction] = candidate
        return candidate

    async def _offspring(self) -> list[Candidate]:
        # Scale roulette weights to avoid overflow without changing probabilities.
        maximum = max(p.score for p in self.population)
        weights = [p.score / maximum for p in self.population] if maximum else None
        offspring = []
        for _ in range(self.size):
            first, second = self.rng.choices(self.population, weights=weights, k=2)
            self.optimizer_calls += 1
            child = await self.crossover(first.prompt, second.prompt, provider=self.provider)
            self.optimizer_calls += 1
            instruction = await self.mutate(
                child,
                first.prompt,
                second.prompt,
                first.examples,
                second.examples,
                first.feedback,
                second.feedback,
                provider=self.provider,
            )
            offspring.append(await self._assess(instruction))
        return offspring

    def _select(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        # Preserve unique incumbents and stable ties; upstream uses an unordered set.
        unique = {p.prompt: p for p in candidates}
        return sorted(unique.values(), key=lambda p: p.score, reverse=True)[: self.size]

    def _snapshot(self, iteration: int) -> dict:
        return {
            "iteration": iteration,
            "best_score": self.population[0].score,
            "mean_score": sum(p.score / len(self.population) for p in self.population),
        }

    async def run(
        self, initial_prompts: Sequence[str], *, iterations: int = 20, seed: int = 0
    ) -> dict:
        """Evaluate seeds, then generate K children and keep the best K each round.

        Supply a nonempty sequence of distinct seed instructions (K is its size).
        Exactly `iterations` generations are evaluated, including the last one.
        Each call resets search state; run one search at a time per instance.
        """
        self.size = len(initial_prompts)
        self.rng = random.Random(seed)
        self.cache, self.responses = {}, []
        self.evaluations = self.optimizer_calls = self.cache_hits = 0
        self.population = self._select([await self._assess(p) for p in initial_prompts])
        history = [self._snapshot(0)]
        for iteration in range(1, iterations + 1):
            offspring = await self._offspring()
            self.population = self._select(self.population + offspring)
            history.append(self._snapshot(iteration))
        return {
            "best": self.population[0],
            "population": self.population.copy(),
            "history": history,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "cache_hits": self.cache_hits,
            "responses": self.responses.copy(),
        }
