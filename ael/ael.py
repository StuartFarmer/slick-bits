"""Algorithm Evolution using Large Language Model (AEL), via Slick."""

import argparse
import ast
import asyncio
import inspect
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from slick import prompt, prompts
from slick.providers import OpenRouterAPI, Provider

from tsp import (
    construct_tour,
    distances,
    exact_length,
    greedy,
    paper_heuristic,
    tour_length,
)


@dataclass(frozen=True)
class Config:
    population_size: int = 10
    generations: int = 10
    parents: int = 2
    offspring: int = 1
    crossover: float = 1.0
    mutation: float = 0.2
    seed: int = 0

    def __post_init__(self):
        for name in ("population_size", "parents", "offspring"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.generations) is not int or self.generations < 0:
            raise ValueError("generations must be a nonnegative integer")
        if self.parents > self.population_size:
            raise ValueError("parents cannot exceed population size")
        if not 0 <= self.crossover <= 1 or not 0 <= self.mutation <= 1:
            raise ValueError("crossover and mutation must be probabilities in [0, 1]")


@dataclass(frozen=True)
class Individual:
    description: str
    code: str
    id: str = ""
    fitness: float | None = None


@prompt(template="propose.j2")
async def propose(operation: str, parents: list[Individual], *, generated: str) -> str:
    """Generate a tagged description and code; the caller parses and evaluates them."""
    return generated


def parse_response(response):
    description = re.search(r"<start>(.*?)<end>", response, re.S)
    code = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.S)
    if not description or not description[1].strip() or not code:
        raise ValueError("expected <start>description<end> and a fenced Python function")
    source = code[1].strip() + "\n"
    tree = ast.parse(source)
    compile(tree, "<candidate>", "exec")
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_next_node"
    ]
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError("define exactly one undecorated select_next_node function")
    args = functions[0].args
    positional = args.posonlyargs + args.args
    if [a.arg for a in positional[:4]] != [
        "current_node",
        "destination_node",
        "unvisited_nodes",
        "distance_matrix",
    ]:
        raise ValueError("select_next_node must use the four prescribed inputs")
    if len(positional) - len(args.defaults) > 4 or any(d is None for d in args.kw_defaults):
        raise ValueError("additional parameters must have defaults")
    return Individual(description[1].strip(), source)


class DemoProvider(Provider):
    """Canned algorithms only: tests the Slick path, not LLM discovery."""

    def __init__(self):
        self.calls = 0

    async def acall(self, context, *, tools=None, tool_results=None):
        choices = [greedy, paper_heuristic]
        selector = choices[self.calls % len(choices)]
        self.calls += 1
        source = "import numpy as np\n" + inspect.getsource(greedy)
        source += "\n" + inspect.getsource(paper_heuristic)
        source += f"\ndef select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):\n    return {selector.__name__}(current_node, destination_node, unvisited_nodes, distance_matrix)\n"
        return (
            f"<start>Offline fixture: {selector.__name__}.<end>\n```python\n{source}```",
            [],
        )


class Evaluator:
    def __init__(self, backend="docker", timeout=30, image="slick-ael:local", seed=0):
        if backend not in ("docker", "trusted"):
            raise ValueError("backend must be docker or trusted")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("evaluation timeout must be positive and finite")
        self.backend, self.timeout, self.image, self.seed = (
            backend,
            timeout,
            image,
            seed,
        )

    def __call__(self, code, instances):
        request = json.dumps(
            {
                "code": code,
                "seed": self.seed,
                "coordinates": [i["coordinates"] for i in instances],
            },
            allow_nan=False,
        )
        worker = Path(__file__).with_name("tsp.py").resolve()
        name = "ael-" + uuid.uuid4().hex
        if self.backend == "docker":
            command = [
                "docker",
                "run",
                "--rm",
                "-i",
                "--name",
                name,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--memory",
                "512m",
                "--cpus",
                "1",
                "--user",
                "65534:65534",
                "--env",
                "OPENBLAS_NUM_THREADS=1",
                "--env",
                "OMP_NUM_THREADS=1",
                "--mount",
                f"type=bind,source={worker},target=/worker.py,readonly",
                self.image,
                "python",
                "-I",
                "-B",
                "/worker.py",
            ]
        else:
            command = [sys.executable, "-I", "-B", str(worker)]
        started = time.monotonic()
        # File-backed output avoids unbounded RAM use if a candidate prints repeatedly.
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryFile() as stdout,
            tempfile.TemporaryFile() as stderr,
        ):
            environment = {
                "PATH": os.environ.get("PATH", os.defpath),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
            if self.backend == "docker":
                # Docker needs its normal socket/context settings, but they are not
                # passed into the container (only the two explicit --env flags are).
                environment = os.environ.copy()
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                cwd=tmp,
                env=environment,
                start_new_session=True,
            )
            try:
                process.communicate(request.encode(), timeout=self.timeout)
            except BaseException:
                timed_out = sys.exc_info()[0] is subprocess.TimeoutExpired
                try:
                    if self.backend == "docker":
                        subprocess.run(
                            ["docker", "rm", "-f", name],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=10,
                            check=True,
                        )
                        process.wait(timeout=5)
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                finally:
                    process.kill()
                    process.wait()
                if timed_out:
                    raise TimeoutError(f"candidate exceeded {self.timeout}s") from None
                raise
            stderr.seek(0)
            if process.returncode:
                detail = stderr.read(4000).decode(errors="replace")
                if self.backend == "docker" and process.returncode in (125, 126, 127):
                    raise RuntimeError(f"Docker evaluation unavailable: {detail}")
                raise ValueError(f"candidate exited {process.returncode}: {detail}")
            stdout.seek(0)
            try:
                result = json.loads(stdout.read(8_000_000))
                tours = result["tours"]
                if len(tours) != len(instances):
                    raise ValueError("candidate returned the wrong number of tours")
                lengths = [
                    tour_length(tour, distances(i["coordinates"]))
                    for tour, i in zip(tours, instances)
                ]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid worker response") from exc
        fitness = float(
            np.mean(
                [
                    100 * (length / i["reference_length"] - 1)
                    for length, i in zip(lengths, instances)
                ]
            )
        )
        if not math.isfinite(fitness):
            raise ValueError("candidate fitness must be finite")
        return {
            "fitness": fitness,
            "lengths": lengths,
            "tours": tours,
            "seconds": time.monotonic() - started,
        }


def validate_dataset(data):
    if not isinstance(data.get("reference"), str) or not data["reference"].strip():
        raise ValueError("dataset must identify its reference solver or heuristic")
    if not isinstance(data.get("instances"), list) or not data["instances"]:
        raise ValueError("dataset must contain instances")
    for instance in data["instances"]:
        distances(instance["coordinates"])
        length = instance["reference_length"]
        if (
            isinstance(length, bool)
            or not isinstance(length, (int, float))
            or not math.isfinite(length)
            or length <= 0
        ):
            raise ValueError("reference lengths must be positive and finite")
    return data


def make_dataset(sizes, count, seed, reference="exact", solver_timeout=60):
    if not sizes or any(type(n) is not int or n < 2 for n in sizes) or count < 1:
        raise ValueError("sizes must be >= 2 and instance count must be positive")
    if reference not in ("exact", "greedy"):
        raise ValueError("reference must be exact or greedy")
    instances = []
    rng = np.random.default_rng(seed)
    for size in sizes:
        for _ in range(count):
            coordinates = rng.random((size, 2)).tolist()
            matrix = distances(coordinates)
            length = (
                exact_length(matrix, solver_timeout)
                if reference == "exact"
                else tour_length(construct_tour(greedy, matrix), matrix)
            )
            instances.append({"coordinates": coordinates, "reference_length": length})
    return validate_dataset(
        {
            "reference": "HiGHS-optimal" if reference == "exact" else "greedy (not optimal)",
            "seed": seed,
            "instances": instances,
        }
    )


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


async def evolve(provider, evaluator, data, config, output):
    validate_dataset(data)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "dataset.json", data)
    write_json(
        output / "config.json",
        {
            **asdict(config),
            "provider": type(provider).__name__,
            "model": getattr(provider, "model", None),
            "evaluation": vars(evaluator),
        },
    )
    rng, counter = random.Random(config.seed), 0
    instances = data["instances"]
    greedy_code = inspect.getsource(greedy).replace("def greedy(", "def select_next_node(")
    # Fail before any paid calls if the evaluation environment cannot run Python.
    greedy_metrics = evaluator(greedy_code, instances)
    write_json(output / "greedy.evaluation.json", greedy_metrics)

    async def create(operation, parents, generation):
        nonlocal counter
        counter += 1
        identifier = f"candidate-{counter:04d}"
        event = {
            "id": identifier,
            "operation": operation,
            "generation": generation,
            "parents": [p.id for p in parents],
        }
        context = await propose.render(operation, parents)
        (output / f"{identifier}.prompt.txt").write_text(context)
        try:
            response = await propose(operation, parents, provider=provider)
            (output / f"{identifier}.response.txt").write_text(response)
            candidate = parse_response(response)
            candidate = Individual(candidate.description, candidate.code, identifier)
            (output / f"{identifier}.py").write_text(candidate.code)
        except (ValueError, SyntaxError) as exc:
            event["error"] = str(exc)
            candidate = None
        except Exception as exc:
            event["error"] = str(exc)
            raise
        finally:
            with (output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
        return candidate

    def evaluate(candidate):
        if candidate is None:
            return None
        try:
            metrics = evaluator(candidate.code, instances)
        except (ValueError, TimeoutError) as exc:
            write_json(output / f"{candidate.id}.evaluation.json", {"error": str(exc)})
            return None
        write_json(output / f"{candidate.id}.evaluation.json", metrics)
        return Individual(candidate.description, candidate.code, candidate.id, metrics["fitness"])

    population = []
    # Bounded replacements for malformed/failed initialization; provider outages abort.
    for _ in range(3 * config.population_size):
        candidate = evaluate(await create("initialization", [], 0))
        if candidate is not None:
            population.append(candidate)
        if len(population) == config.population_size:
            break
    if len(population) != config.population_size:
        raise RuntimeError("could not initialize a full valid population; inspect saved candidates")
    population.sort(key=lambda a: a.fitness)
    initial, history = population.copy(), []

    def record(generation):
        history.append(
            {
                "generation": generation,
                "best": population[0].fitness,
                "mean": float(np.mean([a.fitness for a in population])),
                "population": [a.id for a in population],
                "fitness": [a.fitness for a in population],
            }
        )
        write_json(output / "history.json", history)
        (output / "best.py").write_text(population[0].code)
        write_json(output / "population.json", [asdict(a) for a in population])
        print(
            f"generation {generation}: best {population[0].fitness:.3f}% ({data['reference']})",
            flush=True,
        )

    record(0)
    for generation in range(1, config.generations + 1):
        children = []
        for _ in range(config.population_size):
            parents = rng.sample(population, config.parents)
            crossover = rng.random() < config.crossover
            for _ in range(config.offspring):
                candidate = (
                    await create("crossover", parents, generation)
                    if crossover
                    else rng.choice(parents)
                )
                if candidate is None:
                    continue
                mutated = rng.random() < config.mutation
                if mutated:
                    candidate = await create("mutation", [candidate], generation)
                if crossover or mutated:
                    candidate = evaluate(candidate)
                if candidate is not None:
                    children.append(candidate)
        # Select only after all N iterations: every parent comes from the old generation.
        population = sorted(population + children, key=lambda a: a.fitness)[
            : config.population_size
        ]
        record(generation)
    result = {
        "provider": type(provider).__name__,
        "model": getattr(provider, "model", None),
        "best": asdict(population[0]),
        "population": [asdict(a) for a in population],
        "initial": [asdict(a) for a in initial],
        "history": history,
        "reference": data["reference"],
        "greedy_gap": greedy_metrics["fitness"],
        "direct_llm_mean_gap": float(np.mean([a.fitness for a in initial])),
        "direct_llm_best_gap": initial[0].fitness,
    }
    write_json(output / "result.json", result)
    return result


def benchmark(result_path, dataset, evaluator, output):
    data = validate_dataset(json.loads(dataset.read_text()))
    result = json.loads(result_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    algorithms = {
        "greedy": inspect.getsource(greedy).replace("def greedy(", "def select_next_node("),
        "paper-figure-6": "import numpy as np\n"
        + inspect.getsource(greedy)
        + "\n"
        + inspect.getsource(paper_heuristic).replace(
            "def paper_heuristic(", "def select_next_node("
        ),
        "ael": result["best"]["code"],
    }
    algorithms.update({f"initial-{i}": a["code"] for i, a in enumerate(result["initial"])})
    write_json(output / "dataset.json", data)
    rows = []
    for label, code in algorithms.items():
        (output / f"{label}.py").write_text(code)
        for size in sorted({len(i["coordinates"]) for i in data["instances"]}):
            instances = [i for i in data["instances"] if len(i["coordinates"]) == size]
            row = {
                "algorithm": label,
                "size": size,
                "reference": data["reference"],
                "evolution_provider": result.get("provider"),
                "model": result.get("model"),
            }
            try:
                row.update(evaluator(code, instances))
                row["mean_length"] = float(np.mean(row["lengths"]))
            except (ValueError, TimeoutError) as exc:
                row["error"] = str(exc)
            rows.append(row)
            write_json(output / "results.json", rows)
    return rows


def main(argv=None):
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dataset = sub.add_parser("dataset", help="generate fixed instances and reference lengths")
    dataset.add_argument("--sizes", nargs="+", type=int, default=[50])
    dataset.add_argument("--instances", type=int, default=64)
    dataset.add_argument("--reference", choices=("exact", "greedy"), default="exact")
    dataset.add_argument("--solver-timeout", type=float, default=60)
    dataset.add_argument("--seed", type=int, default=0)
    dataset.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("evolve", help="evolve algorithms using OpenRouter")
    run.add_argument("--model", required=True, help="OpenRouter model ID")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--population-size", type=int, default=10)
    run.add_argument("--generations", type=int, default=10)
    run.add_argument("--parents", type=int, default=2)
    run.add_argument("--offspring", type=int, default=1)
    run.add_argument("--crossover", type=float, default=1)
    run.add_argument("--mutation", type=float, default=0.2)
    run.add_argument("--llm-timeout", type=float, default=120)
    run.add_argument("--seed", type=int, default=0)
    compare = sub.add_parser("benchmark", help="evaluate best, initial population, and baselines")
    compare.add_argument("--result", type=Path, required=True)
    compare.add_argument("--dataset", type=Path, required=True)
    demo = sub.add_parser("demo", help="offline canned-provider smoke run")
    for p in (run, compare, demo):
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--evaluation-timeout", type=float, default=30)
        p.add_argument("--image", default="slick-ael:local")
    for p in (run, compare):
        p.add_argument(
            "--trusted-code",
            action="store_true",
            help="execute as ordinary local Python; only inside a separate trusted environment",
        )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    if args.command == "dataset":
        data = make_dataset(
            args.sizes, args.instances, args.seed, args.reference, args.solver_timeout
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, data)
        print(args.output)
        return
    evaluator = Evaluator(
        "trusted" if args.command == "demo" or args.trusted_code else "docker",
        args.evaluation_timeout,
        args.image,
        getattr(args, "seed", 0),
    )
    if args.command == "benchmark":
        benchmark(args.result, args.dataset, evaluator, args.output)
        print(args.output / "results.json")
        return
    if args.command == "demo":
        config = Config(population_size=3, generations=2)
        data = make_dataset([8], 4, 0)
        provider = DemoProvider()
    else:
        config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
        data = validate_dataset(json.loads(args.dataset.read_text()))
        provider = OpenRouterAPI(model=args.model, timeout=args.llm_timeout, max_output_tokens=4096)
    asyncio.run(evolve(provider, evaluator, data, config, args.output))
    print(args.output / "best.py")


if __name__ == "__main__":
    main()
