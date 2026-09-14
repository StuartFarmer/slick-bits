"""Reflective Evolution (ReEvo), with Slick calls and caller-owned evaluation."""

import ast
import math
import random
import re
from dataclasses import asdict, dataclass, field
from itertools import combinations

from .prompts import generate, reflect_long, reflect_pair


@dataclass(frozen=True)
class Task:
    description: str
    function_description: str
    signature: str
    black_box: bool = False
    initial_reflection: str = ""

    def __post_init__(self):
        tree = ast.parse(self.signature + "\n    pass")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("signature must declare one synchronous Python function")


@dataclass(frozen=True)
class Config:
    population_size: int = 10
    initial_size: int = 30
    max_evaluations: int = 100
    crossover_rate: float = 1.0
    mutation_rate: float = 0.5
    short_reflection: bool = True
    long_reflection: bool = True
    maximize: bool = False
    seed: int = 0

    def __post_init__(self):
        for name in ("population_size", "initial_size", "max_evaluations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("crossover_rate", "mutation_rate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if not (
            int(self.population_size * self.crossover_rate)
            or int(self.population_size * self.mutation_rate)
        ):
            raise ValueError("rates must produce at least one offspring per generation")


@dataclass
class Individual:
    id: int
    code: str
    stage: str
    generation: int
    parents: list[int] = field(default_factory=list)
    objective: float | None = None
    error: str = ""
    response: str = ""


@dataclass
class Result:
    individuals: list[Individual] = field(default_factory=list)
    reflections: list[dict] = field(default_factory=list)
    best: Individual | None = None
    best_history: list[float | None] = field(default_factory=list)
    stop_reason: str = ""

    def to_dict(self):
        return asdict(self)


def extract_code(response, task):
    """Check syntax and calling interface without executing generated code.

    This is validation, not a security sandbox. The evaluator owns execution.
    """
    match = re.fullmatch(r"\s*```(?:python)?\s*\n(.*?)\n?```\s*", response, re.S)
    source = (match[1] if match else response).strip() + "\n"
    tree = ast.parse(source)
    compile(tree, "<heuristic>", "exec")
    expected = ast.parse(task.signature + "\n    pass").body[0]
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == expected.name]
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError(f"define one undecorated function named {expected.name}")
    # Type hints are optional; argument names/kinds/defaults are the task contract.
    for fn in (expected, functions[0]):
        for node in ast.walk(fn.args):
            if isinstance(node, ast.arg):
                node.annotation = None
    if ast.dump(functions[0].args) != ast.dump(expected.args):
        raise ValueError("generated function does not match the required signature")
    return source


class ReEvo:
    """One search run. evaluate(code) must asynchronously return a finite objective.

    Generator, initializer and reflector may use different Slick providers.
    Configure provider temperature at construction; the initializer can use 1.3
    while generator/reflector use 1.0, as in Appendix C. Provider failures abort
    the run; candidate evaluation failures are recorded and consume one shot.
    """

    def __init__(
        self,
        task,
        evaluate,
        generator,
        *,
        reflector=None,
        initializer=None,
        config=None,
    ):
        self.task = task
        self.evaluate = evaluate
        self.generator = generator
        self.reflector = reflector if reflector is not None else generator
        self.initializer = initializer if initializer is not None else generator
        self.config = config or Config()
        self.result = Result()
        self.population = []
        self.long_term = task.initial_reflection
        self.rng = random.Random(self.config.seed)
        self.calls = []
        self.started = False

    def quality(self, individual):
        if individual.objective is None:
            return math.inf
        return -individual.objective if self.config.maximize else individual.objective

    @property
    def remaining(self):
        return self.config.max_evaluations - len(self.result.individuals)

    async def ask(self, function, *args, provider, **kwargs):
        record = {
            "operator": function.__name__,
            "prompt": await function.render(*args, **kwargs),
        }
        self.calls.append(record)
        try:
            response = await function(*args, provider=provider, **kwargs)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        record["response"] = response
        return response

    async def score(self, response, stage, generation, parents=()):
        if not self.remaining:
            raise RuntimeError("evaluation budget exhausted")
        individual = Individual(
            len(self.result.individuals),
            "",
            stage,
            generation,
            list(parents),
            response=response,
        )
        self.result.individuals.append(individual)
        try:
            individual.code = extract_code(response, self.task)
            value = float(await self.evaluate(individual.code))
            if not math.isfinite(value):
                raise ValueError("objective must be finite")
            individual.objective = value
        except Exception as exc:
            individual.error = f"{type(exc).__name__}: {exc}"
        if individual.objective is not None and (
            self.result.best is None or self.quality(individual) < self.quality(self.result.best)
        ):
            self.result.best = individual
        self.result.best_history.append(
            self.result.best.objective if self.result.best is not None else None
        )
        return individual

    async def run(self, *, seed_code=""):
        if self.started:
            raise RuntimeError("create a new ReEvo instance for each run")
        self.started = True
        if seed_code:
            seed = await self.score(seed_code, "seed", 0)
            if seed.error:
                self.result.stop_reason = "invalid_seed"
                return self.result
        for _ in range(min(self.config.initial_size, self.remaining)):
            response = await self.ask(
                generate,
                self.task,
                "initial",
                seed_code=seed_code,
                provider=self.initializer,
            )
            self.population.append(await self.score(response, "initial", 0))

        generation = 0
        while self.remaining:
            generation += 1
            valid = [x for x in self.population if x.objective is not None]
            elite = self.result.best
            if elite is not None and elite not in valid:
                valid.append(elite)
            if not valid:
                self.result.stop_reason = "no_valid_individuals"
                break
            cross_count = min(
                int(self.config.population_size * self.config.crossover_rate),
                self.remaining,
            )
            # ponytail: quadratic pair list is tiny at population 10/30; sample
            # by fitness groups if populations grow to thousands.
            pairs = [(a, b) for a, b in combinations(valid, 2) if a.objective != b.objective]
            if cross_count and not pairs:
                self.result.stop_reason = "no_distinct_parents"
                break
            selected = [
                sorted(self.rng.choice(pairs), key=self.quality, reverse=True)
                for _ in range(cross_count)
            ]
            insights = []
            for worse, better in selected:
                insight = ""
                if self.config.short_reflection:
                    insight = await self.ask(
                        reflect_pair, self.task, worse, better, provider=self.reflector
                    )
                insights.append(insight)
            offspring = []
            for (worse, better), insight in zip(selected, insights):
                response = await self.ask(
                    generate,
                    self.task,
                    "crossover",
                    worse=worse,
                    better=better,
                    reflection=insight,
                    provider=self.generator,
                )
                offspring.append(
                    await self.score(response, "crossover", generation, (worse.id, better.id))
                )
            if self.config.long_reflection and any(insights):
                self.long_term = await self.ask(
                    reflect_long,
                    self.task,
                    self.long_term,
                    insights,
                    provider=self.reflector,
                )
                # Bound carried memory even when the reflector ignores the limit.
                self.long_term = " ".join(self.long_term.split()[:49])
            self.result.reflections.append(
                {
                    "generation": generation,
                    "pairs": [[a.id, b.id] for a, b in selected],
                    "short_term": insights,
                    "long_term": self.long_term,
                }
            )
            elite = self.result.best  # Includes the just-evaluated crossover offspring.
            for _ in range(
                min(
                    int(self.config.population_size * self.config.mutation_rate),
                    self.remaining,
                )
            ):
                response = await self.ask(
                    generate,
                    self.task,
                    "mutation",
                    better=elite,
                    reflection=self.long_term,
                    provider=self.generator,
                )
                offspring.append(await self.score(response, "mutation", generation, (elite.id,)))
            self.population = offspring
        if not self.result.stop_reason:
            self.result.stop_reason = "budget"
        return self.result
