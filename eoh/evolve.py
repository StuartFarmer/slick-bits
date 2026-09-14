"""Evolve heuristic thoughts and code with Slick and isolated fitness evaluation."""

import argparse
import asyncio
import inspect
import json
import math
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from slick import prompt, prompts
from slick.providers import CodexCLI, OpenAIAPI, ProviderError

import problems
from worker import evaluate_candidate

OPERATORS = ("E1", "E2", "M1", "M2", "M3")
INSTRUCTIONS = {
    "INIT": "Design a new heuristic without parent heuristics.",
    "E1": "Design a heuristic as different as possible from all parent heuristics.",
    "E2": "Identify the common idea in the parents; design a different heuristic based on "
    "that shared idea by introducing new components.",
    "M1": "Modify the parent's logic to improve its performance.",
    "M2": "Keep the parent's algorithmic structure; change its numerical parameters "
    "and parameter settings to improve performance.",
    "M3": "Identify the parent's essential and redundant components. Simplify it by "
    "removing redundant components while preserving useful behavior.",
}


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thought: str = Field(min_length=1)
    code: str = Field(min_length=1)

    @field_validator("thought", "code")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class Heuristic(Proposal):
    id: int
    fitness: float = Field(allow_inf_nan=False)


@prompt(template="propose.j2", output_type=Proposal)
async def propose(task, instruction, parents, thoughts=True, *, generated: Proposal) -> Proposal:
    """Propose a thought and heuristic; evaluation and acceptance belong to the caller."""
    return generated


def select_parents(population, count, rng):
    """Weighted sampling without replacement; rank 1 is the highest fitness."""
    ranked = sorted(population, key=lambda h: h.fitness, reverse=True)
    weights = [1 / (rank + len(ranked)) for rank in range(1, len(ranked) + 1)]
    selected = []
    for _ in range(count):
        index = rng.choices(range(len(ranked)), weights=weights, k=1)[0]
        selected.append(ranked.pop(index))
        weights.pop(index)
    return selected


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


async def evolve(
    provider,
    problem,
    evaluate,
    *,
    population_size=20,
    generations=20,
    parents=5,
    seed=0,
    operators=OPERATORS,
    thoughts=True,
    init_attempts=None,
    output=None,
):
    if (
        type(population_size) is not int
        or population_size < 1
        or type(generations) is not int
        or generations < 0
        or type(parents) is not int
        or not 1 <= parents <= population_size
    ):
        raise ValueError("require population >= parents >= 1 and generations >= 0")
    if not operators or len(set(operators)) != len(operators) or set(operators) - set(OPERATORS):
        raise ValueError("operators must be a nonempty unique subset of E1,E2,M1,M2,M3")
    if problem not in problems.TASKS:
        raise ValueError(f"unknown problem: {problem}")
    init_attempts = 3 * population_size if init_attempts is None else init_attempts
    if type(init_attempts) is not int or init_attempts < population_size:
        raise ValueError("init_attempts must be an integer >= population size")
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "attempts.jsonl").touch(exist_ok=False)
    rng = random.Random(seed)
    attempts = 0
    history = []

    async def attempt(operator, generation, selected):
        nonlocal attempts
        attempts += 1
        record = {
            "id": attempts,
            "generation": generation,
            "operator": operator,
            "parents": [h.id for h in selected],
        }
        record["prompt"] = await propose.render(
            problems.TASKS[problem], INSTRUCTIONS[operator], selected, thoughts
        )
        candidate = None
        try:
            proposal = await propose(
                problems.TASKS[problem],
                INSTRUCTIONS[operator],
                selected,
                thoughts,
                provider=provider,
            )
            record.update(proposal.model_dump())
            result = await evaluate(proposal.code)
            if "error" not in result and not math.isfinite(result["fitness"]):
                result = {"error": "fitness must be finite"}
            record["evaluation"] = result
            if "error" not in result:
                candidate = Heuristic(
                    **proposal.model_dump(), id=attempts, fitness=result["fitness"]
                )
        except (ValidationError, ProviderError, ValueError, TimeoutError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        if output:
            with (output / "attempts.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        return candidate

    def save(population, generation):
        row = {
            "generation": generation,
            "attempts": attempts,
            "best": population[0].fitness,
            "mean": sum(h.fitness for h in population) / len(population),
            "population": [h.model_dump() for h in population],
        }
        history.append(row)
        if output:
            write_json(output / "history.json", history)
            write_json(output / "population.json", row["population"])
            (output / "best.py").write_text(population[0].code + "\n")
            (output / "best.txt").write_text(population[0].thought + "\n")
            print(
                f"Generation {generation}: best={row['best']:.6g}, attempts={attempts}",
                flush=True,
            )

    population = []
    while len(population) < population_size and attempts < init_attempts:
        candidate = await attempt("INIT", 0, [])
        if candidate is not None:
            population.append(candidate)
    if len(population) != population_size:
        raise RuntimeError(
            f"initial population incomplete: {len(population)}/{population_size} "
            f"valid after {attempts} attempts; inspect attempt errors"
        )
    population.sort(key=lambda h: h.fitness, reverse=True)
    save(population, 0)
    for generation in range(1, generations + 1):
        offspring = []
        for operator in operators:
            for _ in range(population_size):
                selected = select_parents(
                    population, parents if operator.startswith("E") else 1, rng
                )
                candidate = await attempt(operator, generation, selected)
                if candidate is not None:
                    offspring.append(candidate)
        population = sorted(population + offspring, key=lambda h: h.fitness, reverse=True)[
            :population_size
        ]
        save(population, generation)
    return population


def source_for(function, problem):
    return "import numpy as np\n\n" + inspect.getsource(function).replace(
        f"def {function.__name__}(", f"def {problems.INTERFACES[problem][0]}(", 1
    )


BASELINES = {
    "binpacking": [
        problems.first_fit,
        problems.best_fit,
        problems.funsearch,
        problems.paper_binpacking,
    ],
    "tsp": [problems.tsp_penalty],
    "flowshop": [problems.flow_perturb],
}


class DemoProvider:
    """Canned heuristic variations, never model discovery."""

    def __init__(self, problem):
        self.problem = problem
        self.calls = 0

    async def acall(self, context):
        functions = BASELINES[self.problem]
        function = functions[self.calls % len(functions)]
        self.calls += 1
        return Proposal(
            thought=f"Offline demonstration: {function.__name__}.",
            code=source_for(function, self.problem),
        ).model_dump_json(), []


async def run(args):
    if args.data:
        dataset = problems.validate_dataset(json.loads(args.data.read_text()))
        if dataset["problem"] != args.problem:
            raise ValueError("dataset problem differs from --problem")
    else:
        dataset = problems.generate_dataset(
            args.problem,
            size=args.size,
            count=args.instances,
            capacity=args.capacity,
            machines=args.machines,
            seed=args.seed,
        )
    if (
        args.preset == "paper"
        and args.problem == "tsp"
        and any("optimum" not in instance for instance in dataset["instances"])
    ):
        raise ValueError("paper TSP fitness requires --data with an optimum for every instance")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "dataset.json", dataset)
    write_json(
        args.output / "config.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )

    async def evaluate(code):
        return await evaluate_candidate(
            code,
            dataset,
            timeout=args.eval_timeout,
            iterations=args.iterations,
            seconds=args.seconds,
            seed=args.seed,
        )

    if args.candidate:
        source = args.candidate.read_text()
        result = await evaluate(source)
        (args.output / "candidate.py").write_text(source)
        write_json(args.output / "evaluation.json", result)
        if "error" in result:
            raise ValueError(result["error"])
        print(json.dumps(result, indent=2))
        return
    baseline_results = {}
    for function in BASELINES[args.problem]:
        baseline_results[function.__name__] = await evaluate(source_for(function, args.problem))
    write_json(args.output / "baselines.json", baseline_results)
    if args.provider == "demo":
        provider = DemoProvider(args.problem)
    elif args.provider == "codex":
        provider = CodexCLI(model=args.model, timeout=args.timeout)
    else:
        provider = OpenAIAPI(model=args.model, timeout=args.timeout, max_output_tokens=8192)
    await evolve(
        provider,
        args.problem,
        evaluate,
        population_size=args.population,
        generations=args.generations,
        parents=args.parents,
        seed=args.seed,
        operators=args.operators,
        thoughts=not args.code_only,
        init_attempts=args.init_attempts,
        output=args.output,
    )


def main(argv=None):
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=tuple(problems.TASKS), default="binpacking")
    parser.add_argument("--preset", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--provider", choices=("demo", "codex", "openai"), default="demo")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, default=Path("runs/eoh"))
    parser.add_argument("--data", type=Path, help="JSON instance set")
    parser.add_argument(
        "--candidate", type=Path, help="evaluate existing trusted code; no evolution"
    )
    for flag in (
        "population",
        "generations",
        "parents",
        "size",
        "instances",
        "iterations",
    ):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument(
        "--machines", type=int, help="fixed machine count; 0 samples 2–20 per instance"
    )
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--seconds", type=float, help="wall seconds per GLS instance")
    parser.add_argument("--eval-timeout", type=float, help="hard wall timeout per whole candidate")
    parser.add_argument("--timeout", type=float, default=120, help="provider timeout in seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-attempts", type=int)
    parser.add_argument("--operators", nargs="+", choices=OPERATORS, default=list(OPERATORS))
    parser.add_argument("--code-only", action="store_true", help="omit thoughts from evolution")
    args = parser.parse_args(argv)
    paper = args.preset == "paper"
    defaults = {
        "population": (20 if args.problem == "binpacking" else 10) if paper else 2,
        "generations": 20 if paper else 1,
        "parents": 5 if paper else 2,
        "size": {"binpacking": 5000, "tsp": 100, "flowshop": 50}[args.problem]
        if paper
        else (100 if args.problem == "binpacking" else 8),
        "instances": (5 if args.problem == "binpacking" else 64) if paper else 2,
        "machines": 0 if paper else 3,
        "iterations": 1000 if paper else 5,
        "seconds": 60 if paper else 1,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.eval_timeout is None:
        count = args.instances
        if args.data:
            try:
                count = len(json.loads(args.data.read_text())["instances"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                parser.error(str(exc))
        args.eval_timeout = max(30, count * args.seconds + 10)
    if (
        any(
            getattr(args, name) < 1
            for name in (
                "population",
                "parents",
                "size",
                "instances",
                "capacity",
                "iterations",
            )
        )
        or args.machines < 0
        or args.generations < 0
        or args.parents > args.population
        or any(
            not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0
            for name in ("timeout", "eval_timeout", "seconds")
        )
    ):
        parser.error("positive sizes/budgets, generations >= 0, and parents <= population required")
    if args.provider == "openai" and not args.model and not args.candidate:
        parser.error("--model required with --provider openai")
    if args.provider == "demo" and args.model:
        parser.error("--model requires a real provider")
    try:
        asyncio.run(run(args))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
