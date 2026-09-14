"""QUBE heuristic evolution using Slick: python -m qube.run --help."""

import argparse
import ast
import asyncio
import json
import math
import os
import signal
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
from slick import prompt, prompts
from slick.providers import LiteLLMAPI, Provider, ProviderError

from qube.problems import DEMO_CODES, SEEDS, read_or_library, validate_data, weibull
from qube.search import Database, Program, Recent


@prompt(template="generate.j2")
async def generate(task: str, parent1: str, parent2: str, *, generated: str) -> str:
    """Return candidate source for caller-owned validation and isolated evaluation."""
    return generated


class DemoProvider(Provider):
    def __init__(self, task):
        self.task, self.calls, self.context = task, 0, ""

    async def acall(self, context, *, tools=None, tool_results=None):
        self.context = context
        codes = DEMO_CODES[self.task]
        code = codes[self.calls % len(codes)]
        self.calls += 1
        return code, []


def clean_code(source, task):
    if len(source) > 32_000:
        raise ValueError("candidate exceeds 32,000 characters")
    source = source.strip()
    if source.startswith("```") and source.endswith("```"):
        source = "\n".join(source.splitlines()[1:-1])
    tree = ast.parse(source)
    name = "update_dist" if task == "tsp" else "priority"
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError(f"define exactly one undecorated {name} function")
    if any(
        not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)) for node in tree.body
    ):
        raise ValueError("only functions and imports may appear at module level")
    compile(tree, "<candidate>", "exec")
    # Interface/syntax validation is not a security sandbox. Docker is the boundary.
    return source + "\n"


async def bounded_output(process, request):
    async def read(stream):
        result = bytearray()
        while chunk := await stream.read(65536):
            result.extend(chunk)
            if len(result) > 1_048_576:
                raise ValueError("evaluator output exceeds 1 MiB")
        return bytes(result)

    async def send():
        try:
            process.stdin.write(request)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    tasks = [
        asyncio.create_task(read(process.stdout)),
        asyncio.create_task(read(process.stderr)),
        asyncio.create_task(send()),
    ]
    try:
        stdout, stderr, _ = await asyncio.gather(*tasks)
        await process.wait()
        return stdout, stderr
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def evaluate_code(code, args, instances):
    request = json.dumps(
        {
            "code": code,
            "task": args.task,
            "instances": instances,
            "iterations": args.iterations,
        },
        allow_nan=False,
    ).encode()
    container = None
    if args.provider == "demo":
        if code not in DEMO_CODES[args.task]:
            raise ValueError("host evaluation is restricted to fixed demo programs")
        command = [sys.executable, "-m", "qube.worker"]
    else:
        container = "qube-" + uuid.uuid4().hex
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            container,
            "--log-driver",
            "none",
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
            "--memory-swap",
            "512m",
            "--cpus",
            "1",
            "--user",
            "65534:65534",
            "--ulimit",
            "nofile=64:64",
            "--ulimit",
            "fsize=1048576:1048576",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            args.image,
        ]
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        start_new_session=True,
    )
    try:
        raw, errors = await asyncio.wait_for(bounded_output(process, request), args.eval_timeout)
        if process.returncode:
            raise ValueError(
                f"evaluator exited {process.returncode}: {errors[:2000].decode(errors='replace')}"
            )
        result = json.loads(raw)
        signature = tuple(float(x) for x in result["signature"])
        if len(signature) != len(instances):
            raise ValueError("wrong number of per-instance results")
        return Program(code, signature, float(result["score"]))
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Drain only residual pipe buffers after killing the local writers.
        await process.communicate()
        if container:
            # Killing the Docker CLI alone leaves its container running.
            cleanup = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                container,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(cleanup.wait(), 10)
            except TimeoutError:
                cleanup.kill()
                await cleanup.wait()


def load_instances(args):
    if args.data:
        instances = (
            read_or_library(args.data)
            if args.task == "binpack" and args.data.suffix != ".json"
            else json.loads(args.data.read_text())
        )
    elif args.task == "binpack":
        instances = weibull(args.items, args.instances, args.seed)
    elif args.task == "capset":
        instances = [args.dimension]
    else:
        rng = np.random.default_rng(args.seed)
        instances = [
            {"cities": rng.random((args.cities, 2)).tolist()} for _ in range(args.instances)
        ]
    validate_data(args.task, instances)
    return instances


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


async def evolve(args):
    validate_args(args)
    instances = load_instances(args)
    args.output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of an existing run.
    history_path = args.output / "history.jsonl"
    with history_path.open("x") as history:
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        save_json(args.output / "config.json", config)
        save_json(args.output / "instances.json", instances)
        provider = (
            DemoProvider(args.task)
            if args.provider == "demo"
            else LiteLLMAPI(
                args.model,
                api_base=args.api_base,
                timeout=args.llm_timeout,
                options={"temperature": 1.0, "top_p": 0.95, "max_tokens": 4096},
            )
        )
        initial = await evaluate_code(SEEDS[args.task], args, instances)
        db = Database(
            initial,
            islands=args.islands,
            k=args.k,
            reset_interval=args.reset_interval,
            temperature=args.program_temperature,
            seed=args.seed,
        )
        recent = Recent()
        evaluator_slots = asyncio.Semaphore(args.evaluators)
        resets = []
        (args.output / "best.py").write_text("import numpy as np\n\n" + initial.code)
        save_json(args.output / "best.json", asdict(initial))

        async def sample(index, parents, clusters):
            code, program, error = None, None, None
            try:
                code = await asyncio.wait_for(
                    generate(args.task, parents[0].code, parents[1].code, provider=provider),
                    args.llm_timeout,
                )
                code = clean_code(code, args.task)
                async with evaluator_slots:
                    program = await evaluate_code(code, args, instances)
            except (
                ProviderError,
                TimeoutError,
                ValueError,
                SyntaxError,
                TypeError,
                KeyError,
                OSError,
                RuntimeError,
            ) as exc:
                error = f"{type(exc).__name__}: {exc}"[:2000]
            previous_best = db.best
            db.record(index, clusters, program)
            recent.add(program, parents)
            history.write(
                json.dumps(
                    {
                        "sample": db.generated,
                        "island": index,
                        "code": code,
                        "parents": [p.code for p in parents],
                        "score": program.score if program else None,
                        "signature": program.signature if program else None,
                        "error": error,
                        "best_score": db.best.score,
                        **recent.metrics(),
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            history.flush()
            if db.best is not previous_best:
                (args.output / "best.py").write_text("import numpy as np\n\n" + db.best.code)
                save_json(args.output / "best.json", asdict(db.best))

        while db.generated < args.samples:
            boundary = (
                min(
                    args.samples,
                    (db.generated // args.reset_interval + 1) * args.reset_interval,
                )
                if args.reset_interval
                else args.samples
            )
            remaining = boundary - db.generated

            async def sampler():
                nonlocal remaining
                while remaining:
                    count = min(args.samples_per_prompt, remaining)
                    remaining -= count
                    index, parents, clusters = db.select(count)
                    # Independent Slick calls share the same two-parent prompt.
                    # TaskGroup cancels siblings cleanly if the run is interrupted.
                    async with asyncio.TaskGroup() as group:
                        for _ in range(count):
                            group.create_task(sample(index, parents, clusters))

            async with asyncio.TaskGroup() as group:
                for _ in range(args.samplers):
                    group.create_task(sampler())
            # Drain in-flight offspring before replacing island state.
            if args.reset_interval and db.generated % args.reset_interval == 0:
                resets.extend({"sample": db.generated, **event} for event in db.reset())
                save_json(args.output / "resets.json", resets)

        metric = (
            "cap_set_size"
            if args.task == "capset"
            else "negative_mean_tour_length"
            if args.task == "tsp" and "optimum" not in instances[0]
            else "negative_excess_ratio"
        )
        summary = {
            "task": args.task,
            "provider": args.provider,
            "model": args.model,
            "generated": db.generated,
            "accepted": db.accepted,
            "rejected": db.generated - db.accepted,
            "resets": len(resets),
            "initial_score": initial.score,
            "best_score": db.best.score,
            "score_metric": metric,
            **recent.metrics(),
        }
        save_json(args.output / "summary.json", summary)
        return summary


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=tuple(SEEDS), default="binpack")
    p.add_argument("--provider", choices=("demo", "litellm"), default="demo")
    p.add_argument("--model", help="LiteLLM model ID, e.g. openai/OpenCoder-8B-Instruct")
    p.add_argument("--api-base", help="optional OpenAI-compatible inference endpoint")
    p.add_argument("--image", default="qube-evaluator:local")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--samples-per-prompt", type=int)
    p.add_argument("--samplers", type=int, default=2)
    p.add_argument("--evaluators", type=int, default=2)
    p.add_argument("--islands", type=int)
    p.add_argument("--k", type=float)
    p.add_argument("--reset-interval", type=int)
    p.add_argument("--program-temperature", type=float, default=1.0)
    p.add_argument("--eval-timeout", type=float)
    p.add_argument("--llm-timeout", type=float, default=120)
    p.add_argument("--data", type=Path, help="JSON instances; also OR-Library text for binpack")
    p.add_argument("--items", type=int, default=1000)
    p.add_argument("--instances", type=int, default=5)
    p.add_argument("--dimension", type=int, default=8)
    p.add_argument("--cities", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("qube/runs/demo"))
    return p


def validate_args(args):
    is_or = args.task == "binpack" and args.data is not None
    defaults = {
        "islands": 1 if args.task == "tsp" else 10,
        "k": {"binpack": 0.0008 if is_or else 0.0001, "capset": 32, "tsp": 1e-5}[args.task],
        "samples_per_prompt": 1 if args.task == "tsp" else 4,
        "reset_interval": {"binpack": 32768, "capset": 262144, "tsp": 0}[args.task],
        "eval_timeout": {"binpack": 30 if is_or else 60, "capset": 90, "tsp": 90}[args.task],
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in (
        "samples",
        "samples_per_prompt",
        "samplers",
        "evaluators",
        "islands",
        "items",
        "instances",
        "iterations",
        "program_temperature",
        "eval_timeout",
        "llm_timeout",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 1 <= args.dimension <= 10 or args.cities < 3:
        raise ValueError("dimension must be 1–10; cities must be at least 3")
    if not math.isfinite(args.k) or args.k < 0 or args.reset_interval < 0:
        raise ValueError("k and reset interval must be nonnegative and finite")
    if args.provider == "litellm" and not args.model:
        raise ValueError("--model is required with --provider litellm")
    if args.provider == "demo" and (args.model or args.api_base):
        raise ValueError("--model and --api-base require --provider litellm")


def main():
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    p = make_parser()
    args = p.parse_args()
    try:
        summary = asyncio.run(evolve(args))
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
