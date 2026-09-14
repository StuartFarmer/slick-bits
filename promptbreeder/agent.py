"""Co-evolve task prompts, mutation prompts, and successful contexts by tournament."""

import json
import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from slick import prompt
from slick.providers import Provider

# Seed data reused from Dylan Banarse's MIT-licensed minimal implementation.
_SEEDS = json.loads((Path(__file__).parent / "prompts/seeds.json").read_text())
MUTATIONS = tuple(_SEEDS["mutations"])
STYLES = tuple(_SEEDS["styles"])


@dataclass(frozen=True)
class Unit:
    prompts: tuple[str, ...]
    mutation: str
    context: tuple[str, ...] = ()
    lineage: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class Evaluation:
    score: float
    correct_workings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Individual:
    unit: Unit
    evaluation: Evaluation


class PromptBreeder:
    """Own nine operators, prompt crossover, and binary tournament replacement.

    Evaluate the entire prompt sequence and context. Scores are finite/nonnegative.
    Similarity is caller-supplied embedding cosine similarity for EDA filtering.
    Errors propagate without retries. Evaluator owns training-batch sampling.
    Lineages store full prompt sequences. Correct workings must contain enough
    input/output context to act as self-contained demonstrations.
    """

    operators = (
        "zero",
        "first",
        "distribution",
        "ranked",
        "lineage",
        "hyper_zero",
        "hyper_first",
        "lamarckian",
        "context",
    )

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Unit], Awaitable[Evaluation]],
        similarity: Callable[[str, str], Awaitable[float]],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.similarity = similarity
        self.rng = random.Random(0)
        self.evaluations = 0
        self.history: list[Individual] = []
        self.population: list[Individual] = []
        self.best: Individual | None = None
        self.raw_responses: list[str] = []
        self.fallbacks: list[tuple[str, str]] = []
        self.operator_history: list[str] = []
        self.pairs: list[tuple[int, int]] = []

    def _text(self, generated: str) -> str:
        # Retain raw continuations even when postprocessing rejects them.
        self.raw_responses.append(generated)
        text = generated.strip()
        if not text:
            raise ValueError("generated prompt must not be blank")
        return text

    @prompt(template="initialize.j2")
    async def initialize(self, mutation: str, style: str, *, generated: str) -> str:
        return self._text(generated)

    @prompt(template="zero.j2")
    async def zero(self, *, generated: str) -> str:
        text = self._text(generated)
        # A hint may span lines; stop at the next numbered/bulleted entry.
        text = re.split(r"\n\s*(?:\(?\d+[.)]|[-*])\s+", text, maxsplit=1)[0]
        text = re.sub(r"^(?:\(?\d+[.)]|[-*])\s*", "", text).strip()
        if not text:
            raise ValueError("generated first hint must not be blank")
        return text

    @prompt(template="first.j2")
    async def first(self, instruction: str, mutation: str, *, generated: str) -> str:
        return self._text(generated)

    @prompt(template="distribution.j2")
    async def distribution(self, parents: Sequence[str], *, generated: str) -> str:
        return self._text(generated)

    @prompt(template="ranked.j2")
    async def ranked(self, parents: Sequence[str], mutation: str, *, generated: str) -> str:
        return self._text(generated)

    @prompt(template="lineage.j2")
    async def lineage(self, ancestors: Sequence[str], *, generated: str) -> str:
        return self._text(generated)

    @prompt(template="hyper_zero.j2")
    async def hyper_zero(self, style: str, *, generated: str) -> str:
        return self._text(generated)

    @prompt(template="hyper_first.j2")
    async def hyper_first(self, mutation: str, *, generated: str) -> str:
        return self._text(generated)

    @prompt(template="lamarckian.j2")
    async def lamarckian(self, workings: Sequence[str], *, generated: str) -> str:
        return self._text(generated)

    @prompt(template="start.j2")
    async def start(
        self, instruction: str, question: str, context: Sequence[str], *, generated: str
    ) -> str:
        return generated

    @prompt(template="continue.j2")
    async def continue_solution(self, transcript: str, instruction: str, *, generated: str) -> str:
        return generated

    async def predict(self, unit: Unit, question: str, *, provider: Provider) -> tuple[str, ...]:
        """Execute the prompt sequence; return all outputs, with the answer last.

        The caller supplies the inference provider, independently of mutation
        generation. No labels or mutation prompt enter the inference context.
        """
        first = await self.start(unit.prompts[0], question, unit.context, provider=provider)
        transcript = await PromptBreeder.start.render(self, unit.prompts[0], question, unit.context)
        outputs = [first]
        transcript += "\n" + first
        for instruction in unit.prompts[1:]:
            output = await self.continue_solution(transcript, instruction, provider=provider)
            transcript += "\n" + instruction + "\n" + output
            outputs.append(output)
        return tuple(outputs)

    async def _assess(self, unit: Unit) -> Individual:
        # Counts evaluator invocations, including failed/invalid measurements.
        self.evaluations += 1
        result = await self.evaluate(unit)
        if not math.isfinite(result.score) or result.score < 0:
            raise ValueError("fitness must be finite and nonnegative")
        individual = Individual(unit, result)
        self.history.append(individual)
        if self.best is None or result.score > self.best.evaluation.score:
            self.best = individual
        return individual

    async def _initialize(self, mutations, styles, size, prompt_count):
        population = []
        for _ in range(size):
            mutation = self.rng.choice(mutations)
            prompts = tuple(
                [
                    await self.initialize(mutation, self.rng.choice(styles), provider=self.provider)
                    for _ in range(prompt_count)
                ]
            )
            population.append(await self._assess(Unit(prompts, mutation)))
        return population

    async def _diverse(self, population, slot, ranked):
        # Filter fitter prompts first, then present retained prompts worst-to-best.
        ordered = sorted(population, key=lambda p: p.evaluation.score, reverse=True)
        if not ranked:
            self.rng.shuffle(ordered)
        parents = []
        for individual in ordered:
            text = individual.unit.prompts[slot]
            distinct = True
            for existing in parents:
                similarity = await self.similarity(text, existing)
                if not math.isfinite(similarity):
                    raise ValueError("measured similarity must be finite")
                if similarity > 0.95:
                    distinct = False
                    break
            if distinct:
                parents.append(text)
        return parents[::-1] if ranked else parents

    def _context(self, unit: Unit, workings: Sequence[str], maximum: int) -> Unit:
        if not maximum:
            return replace(unit, context=())
        context = list(unit.context[:maximum])
        novel = list(dict.fromkeys(w for w in workings if w not in context))
        if len(context) < maximum:
            context.extend(self.rng.sample(novel, min(maximum - len(context), len(novel))))
        elif novel:
            # The paper replaces ONE example after a batch, not one per success.
            context[self.rng.randrange(maximum)] = self.rng.choice(novel)
        return replace(unit, context=tuple(context))

    def _resample_context(self, unit: Unit, workings: Sequence[str], maximum: int) -> Unit:
        if not maximum or self.rng.random() >= 0.1:
            return unit
        pool = list(dict.fromkeys((*unit.context, *workings)))
        self.rng.shuffle(pool)
        # Section 3.2.5 leaves resampling underspecified: Bernoulli(1/K)
        # per available verified working, capped at K; an empty context is valid.
        context = tuple(w for w in pool if self.rng.random() < 1 / maximum)[:maximum]
        return replace(unit, context=context)

    async def _mutate(self, parent, population, styles, maximum, operator):
        unit = self._context(parent.unit, parent.evaluation.correct_workings, maximum)
        unit = self._resample_context(unit, parent.evaluation.correct_workings, maximum)
        slot = self.rng.randrange(len(unit.prompts))
        instruction, mutation = unit.prompts[slot], unit.mutation
        workings = parent.evaluation.correct_workings or unit.context
        if operator == "lamarckian" and not workings:
            fallback = self.rng.choice(("zero", "first"))
            self.fallbacks.append((operator, fallback))
            operator = fallback
        if operator == "zero":
            instruction = await self.zero(provider=self.provider)
        elif operator == "first":
            instruction = await self.first(instruction, mutation, provider=self.provider)
        elif operator in ("distribution", "ranked"):
            parents = await self._diverse(population, slot, operator == "ranked")
            if operator == "ranked":
                instruction = await self.ranked(parents, mutation, provider=self.provider)
            else:
                instruction = await self.distribution(parents, provider=self.provider)
        elif operator == "lineage":
            ancestors = tuple(prompts[slot] for prompts in unit.lineage) or (instruction,)
            instruction = await self.lineage(ancestors, provider=self.provider)
        elif operator in ("hyper_zero", "hyper_first"):
            if operator == "hyper_zero":
                mutation = await self.hyper_zero(self.rng.choice(styles), provider=self.provider)
            else:
                mutation = await self.hyper_first(mutation, provider=self.provider)
            instruction = await self.first(instruction, mutation, provider=self.provider)
        elif operator == "lamarckian":
            instruction = await self.lamarckian(workings, provider=self.provider)
        elif operator == "context":
            context = list(unit.context)
            self.rng.shuffle(context)
            unit = replace(unit, context=tuple(context))
        else:
            raise ValueError(f"unknown mutation operator: {operator}")
        prompts = list(unit.prompts)
        prompts[slot] = instruction
        if self.rng.random() < 0.1:
            donors = [p for p in population if p is not parent]
            largest = max(p.evaluation.score for p in donors)
            weights = [p.evaluation.score / largest for p in donors] if largest else None
            donor = self.rng.choices(donors, weights=weights, k=1)[0]
            prompts[slot] = self.rng.choice(donor.unit.prompts)
        return replace(unit, prompts=tuple(prompts), mutation=mutation)

    def _record_elite(self, population: list[Individual]) -> None:
        index = max(range(len(population)), key=lambda i: population[i].evaluation.score)
        elite = population[index]
        lineage = elite.unit.lineage
        if not lineage or lineage[-1] != elite.unit.prompts:
            unit = replace(elite.unit, lineage=lineage + (elite.unit.prompts,))
            population[index] = replace(elite, unit=unit)

    async def _tournament(self, population, pair, styles, maximum, operators):
        winner, loser = sorted(pair, key=lambda i: population[i].evaluation.score, reverse=True)
        operator = self.rng.choice(operators)
        self.operator_history.append(operator)
        self.pairs.append(pair)
        child = await self._mutate(population[winner], population, styles, maximum, operator)
        # Keep the winner and overwrite the loser, even when the child is worse.
        population[loser] = await self._assess(child)
        self._record_elite(population)

    async def _generation(self, styles, maximum, operators, remaining):
        order = list(range(len(self.population)))
        self.rng.shuffle(order)
        pairs = list(zip(order[::2], order[1::2]))
        for pair in pairs[:remaining]:
            await self._tournament(self.population, pair, styles, maximum, operators)

    async def run(
        self,
        mutations: Sequence[str] = MUTATIONS,
        styles: Sequence[str] = STYLES,
        *,
        population_size: int = 50,
        prompt_count: int = 2,
        generations: int = 20,
        tournaments: int | None = None,
        context_size: int = 0,
        operators: Sequence[str] = operators,
        seed: int = 0,
    ) -> dict:
        """Run disjoint binary tournaments; maximize nonnegative measured fitness.

        Use a fresh instance per run. Explicit tournaments replaces the generation
        budget; odd populations receive one random bye per generation. Evaluation
        invokes the caller once per initial unit and once per offspring.
        """
        self.rng = random.Random(seed)
        self.population = await self._initialize(mutations, styles, population_size, prompt_count)
        self._record_elite(self.population)
        budget = generations * (population_size // 2) if tournaments is None else tournaments
        while len(self.operator_history) < budget:
            await self._generation(
                styles, context_size, operators, budget - len(self.operator_history)
            )
        return {
            "best": self.best,
            "population": self.population,
            "evaluations": self.evaluations,
            "operators": self.operator_history,
            "pairs": self.pairs,
            "history": self.history,
            "fallbacks": self.fallbacks,
        }
