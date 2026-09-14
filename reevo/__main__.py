"""Run ReEvo on TSP; the default demo uses canned code through Slick."""

import argparse
import asyncio
import json
import math
from dataclasses import asdict
from pathlib import Path

from slick import prompts
from slick.providers import LiteLLMAPI, Provider

from .core import Config, ReEvo
from .tsp import TSPEvaluator, make_instances, task_and_seed


class DemoProvider(Provider):
    """Canned heuristics and hints: exercises the pipeline, not LLM discovery."""

    def __init__(self, black_box=False):
        self.black_box = black_box
        self.count = 0

    async def acall(self, context, *, tools=None, tool_results=None):
        if "[Improved code]" not in context:
            return (
                "Favor lower attributes; compare nonlinear transformations while retaining diverse choices.",
                [],
            )
        powers = (0.0, 1.0, 2.0, 0.5, 3.0, 1.5, 4.0)
        power = powers[self.count % len(powers)]
        self.count += 1
        argument = "edge_attr" if self.black_box else "distance_matrix"
        expression = "edge_attr[:, 0]" if self.black_box else "distance_matrix"
        return (
            f"import numpy as np\ndef heuristics({argument}: np.ndarray) -> np.ndarray:\n"
            f"    return 1 / np.maximum({expression}, 1e-9) ** {power}\n"
        ), []


async def run(args):
    task, seed = task_and_seed(args.black_box)
    config = Config(
        population_size=args.population_size,
        initial_size=args.initial_size,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        crossover_rate=0 if args.no_crossover else 1,
        mutation_rate=0 if args.no_mutation else 0.5,
        short_reflection=not args.no_short_reflection,
        long_reflection=not args.no_long_reflection,
    )
    if args.provider == "demo":
        generator = reflector = initializer = DemoProvider(args.black_box)
    else:
        generator = LiteLLMAPI(
            args.model,
            timeout=args.llm_timeout,
            options={"temperature": args.temperature},
        )
        initializer = LiteLLMAPI(
            args.model,
            timeout=args.llm_timeout,
            options={"temperature": args.temperature + 0.3},
        )
        reflector = LiteLLMAPI(
            args.reflector_model or args.model,
            timeout=args.llm_timeout,
            options={"temperature": args.temperature},
        )
    evaluator = TSPEvaluator(
        make_instances(args.instances, args.nodes, seed=args.seed),
        black_box=args.black_box,
        ants=args.ants,
        iterations=args.aco_iterations,
        seed=args.seed,
        timeout=args.eval_timeout,
    )
    search = ReEvo(
        task,
        evaluator,
        generator,
        reflector=reflector,
        initializer=initializer,
        config=config,
    )
    # Exclusive creation avoids silently overwriting a previous expensive run.
    args.output.mkdir(parents=True, exist_ok=False)
    state = {
        "provider": args.provider,
        "model": args.model,
        "reflector_model": args.reflector_model or args.model,
        "config": asdict(config),
        "task": asdict(task),
        "settings": {key: value for key, value in vars(args).items() if key != "output"},
    }
    try:
        result = await search.run(seed_code=seed)
        if result.best is not None:
            # Never expose held-out scores to selection or reflection.
            held_out = TSPEvaluator(
                make_instances(args.test_instances, args.nodes, seed=args.seed + 1),
                black_box=args.black_box,
                ants=args.ants,
                iterations=args.aco_iterations,
                seed=args.seed + 1,
                timeout=args.eval_timeout,
            )
            state["test_objective"] = await held_out(result.best.code)
            baseline = (
                "import numpy as np\n"
                + task.signature
                + (
                    "\n    return 1 / np.maximum(edge_attr[:, 0], 1e-9)\n"
                    if args.black_box
                    else "\n    return 1 / np.maximum(distance_matrix, 1e-9)\n"
                )
            )
            state["test_baseline_objective"] = await held_out(baseline)
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state.update(search.result.to_dict(), calls=search.calls)
        for individual in search.result.individuals:
            if individual.code:
                (args.output / f"candidate-{individual.id:03d}.py").write_text(individual.code)
        if search.result.best is not None:
            (args.output / "best.py").write_text(search.result.best.code)
        temporary = args.output / "result.tmp"
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "result.json")
    best = result.best.objective if result.best is not None else None
    print(
        f"{args.provider}: {len(result.individuals)} evaluations; stop={result.stop_reason}; "
        f"validation={best}; held-out={state.get('test_objective')}"
    )
    print(f"Saved {args.output / 'result.json'}")
    return 0 if result.best is not None else 1


def main(argv=None):
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("demo", "litellm"), default="demo")
    parser.add_argument("--model", help="model ID; required for LiteLLM")
    parser.add_argument("--reflector-model")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--llm-timeout", type=float, default=120)
    parser.add_argument("--eval-timeout", type=float, default=30)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--initial-size", type=int, default=30)
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--test-instances", type=int, default=8)
    parser.add_argument("--ants", type=int, default=8)
    parser.add_argument("--aco-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--black-box", action="store_true")
    for component in ("short-reflection", "long-reflection", "crossover", "mutation"):
        parser.add_argument(f"--no-{component}", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/reevo"))
    args = parser.parse_args(argv)
    if args.provider == "litellm" and not args.model:
        parser.error("--model is required for LiteLLM (use a provider-prefixed model ID)")
    if args.provider == "demo" and (args.model or args.reflector_model):
        parser.error("model options require a real provider")
    if args.test_instances < 1:
        parser.error("--test-instances must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    if not math.isfinite(args.llm_timeout) or args.llm_timeout <= 0:
        parser.error("--llm-timeout must be finite and positive")
    try:
        return asyncio.run(run(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
