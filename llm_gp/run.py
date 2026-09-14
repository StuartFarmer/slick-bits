"""Run the paper variants: python -m llm_gp.run --variant all --provider demo."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import random
import shutil
import statistics
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import slick
from slick import prompts
from slick.providers import CodexCLI

from llm_gp.core import VARIANTS, Config, evolve, split_data
from llm_gp.operators import DemoProvider, MeasuredOpenAI, Operators


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_json(handle, value):
    handle.write(json.dumps(value, allow_nan=False) + "\n")
    handle.flush()


def usage_summary(calls, input_rate=None, output_rate=None, offline=False):
    known = [call["usage"] for call in calls if call["usage"] is not None]
    complete = len(known) == len(calls)
    inputs = sum(u["input_tokens"] for u in known)
    outputs = sum(u["output_tokens"] for u in known)
    cost = None
    if not calls or offline:
        cost = 0.0
    elif complete and input_rate is not None and output_rate is not None:
        cost = (inputs * input_rate + outputs * output_rate) / 1_000_000
    return {
        "calls": len(calls),
        "errors": dict(Counter(c["operation"] for c in calls if c["error"])),
        "fallbacks": sum(c["fallback"] for c in calls),
        "response_seconds": sum(c["seconds"] for c in calls),
        "input_tokens": inputs if complete else None,
        "output_tokens": outputs if complete else None,
        "usage_known_calls": len(known),
        "estimated_cost_usd": cost,
    }


async def experiment(args):
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source / path.name)
    shutil.copytree(Path(__file__).resolve().parent / "prompts", source / "prompts")
    slick_root = Path(slick.__file__).parent
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {**vars(args), "output": str(args.output)},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "slick_path": str(slick_root),
        "slick_version": importlib.metadata.version("slick-ai"),
        "slick_source_sha256": {
            str(p.relative_to(slick_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in slick_root.rglob("*.py")
        },
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature if args.provider == "openai" else None,
        "offline_canned": args.provider == "demo",
    }
    write_json(args.output / "metadata.json", metadata)
    completed = []
    for run in range(args.runs):
        data = split_data(args.seed + run)
        write_json(args.output / f"data-{run:03}.json", data)
        for variant in variants:
            config = Config(
                variant=variant,
                population_size=args.population_size,
                generations=args.generations,
                crossover_rate=args.crossover_rate,
                mutation_rate=args.mutation_rate,
                seed=args.seed + run,
                train_size=args.train_size,
            )
            directory = args.output / f"{variant}-{run:03}"
            directory.mkdir()
            write_json(directory / "config.json", asdict(config))
            if args.provider == "openai":
                backend = MeasuredOpenAI(
                    args.model,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    max_output_tokens=args.max_output_tokens,
                )
            elif args.provider == "codex":
                backend = CodexCLI(model=args.model, timeout=args.timeout)
            else:
                backend = DemoProvider()
            with (
                (directory / "calls.jsonl").open("w", encoding="utf-8") as calls,
                (directory / "generations.jsonl").open("w", encoding="utf-8") as generations,
            ):
                ops = Operators(
                    backend,
                    random.Random(config.seed),
                    n_shots=args.n_shots,
                    retries=args.retries,
                    timeout=args.timeout,
                    seconds=args.seconds,
                    max_calls=args.max_calls,
                    on_call=lambda c: append_json(calls, c),
                )
                result = await evolve(
                    config,
                    data,
                    ops,
                    on_generation=lambda g: append_json(generations, g),
                )
            result["usage"] = usage_summary(
                ops.calls,
                args.input_rate,
                args.output_rate,
                offline=args.provider == "demo",
            )
            result.update(variant=variant, seed=config.seed)
            write_json(directory / "result.json", result)
            completed.append(result)
            best = result["best"]
            print(
                f"{variant} seed={config.seed}: FE={result['evaluations']}, "
                f"best={best}, calls={len(ops.calls)}, fallbacks={result['usage']['fallbacks']}, "
                f"stop={result['stop_reason']}",
                flush=True,
            )

    summary = {}
    for variant in variants:
        rows = [r for r in completed if r["variant"] == variant]
        times = [r["seconds"] for r in rows]
        errors = [r["best_holdout_mse"] for r in rows if r["best_holdout_mse"] is not None]
        costs = [r["usage"]["estimated_cost_usd"] for r in rows]
        summary[variant] = {
            "runs": len(rows),
            "mean_seconds": statistics.mean(times),
            "stdev_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
            "mean_holdout_mse": statistics.mean(errors) if errors else None,
            "valid_holdout_runs": len(errors),
            "solved_holdout": sum(error <= 1e-12 for error in errors),
            "evaluations": sum(r["evaluations"] for r in rows),
            "calls": sum(r["usage"]["calls"] for r in rows),
            "total_estimated_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
        }
    write_json(args.output / "summary.json", summary)
    print(f"Saved {args.output}")


def main(argv=None):
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="llm-gp-mu-xo")
    parser.add_argument("--provider", choices=("demo", "openai", "codex"), default="demo")
    parser.add_argument("--model", help="explicit live model ID; demo ignores this")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scale", action="store_true", help="default to population 20, generations 60"
    )
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument(
        "--train-size",
        type=int,
        help="use this many training examples for every variant",
    )
    parser.add_argument("--n-shots", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="for API models without temperature",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-calls", type=int, default=10000, help="per run, including retries")
    parser.add_argument("--seconds", type=float, default=60000, help="LLM wall time limit per run")
    parser.add_argument("--timeout", type=float, default=60, help="per call seconds")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--input-rate", type=float, help="USD per million input tokens")
    parser.add_argument("--output-rate", type=float, help="USD per million output tokens")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("llm_gp/runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
    )
    args = parser.parse_args(argv)
    if args.population_size is None:
        args.population_size = 20 if args.scale else 10
    if args.generations is None:
        args.generations = 60 if args.scale else 30
    if args.omit_temperature:
        args.temperature = None
    try:
        Config(
            population_size=args.population_size,
            generations=args.generations,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            train_size=args.train_size,
        )
        Operators(
            DemoProvider(),
            random.Random(0),
            n_shots=args.n_shots,
            retries=args.retries,
            max_calls=args.max_calls,
            seconds=args.seconds,
            timeout=args.timeout,
        )
        if args.runs < 1 or args.max_output_tokens < 1:
            raise ValueError("runs and max-output-tokens must be positive")
        if args.train_size is not None and args.train_size > 67:
            raise ValueError("train-size must be <= 67")
        if args.temperature is not None and (
            not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2
        ):
            raise ValueError("temperature must be in [0, 2]")
        if (args.input_rate is None) != (args.output_rate is None):
            raise ValueError("supply both input-rate and output-rate")
        if any(
            r is not None and (not math.isfinite(r) or r < 0)
            for r in (args.input_rate, args.output_rate)
        ):
            raise ValueError("token rates must be finite and nonnegative")
        if args.provider != "demo" and not args.model:
            raise ValueError("live providers require --model")
        if args.provider == "openai" and importlib.util.find_spec("openai") is None:
            raise ValueError("install the optional OpenAI SDK: pip install 'openai>=2,<3'")
        if args.output.exists():
            raise ValueError("output directory already exists; choose a new path")
    except ValueError as exc:
        parser.error(str(exc))
    asyncio.run(experiment(args))


if __name__ == "__main__":
    main()
