"""Optimize discrete prompts with EvoPROMPT and Slick; defaults to an offline demo."""

import argparse
import asyncio
import hashlib
import importlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

from slick import prompt, prompts
from slick.providers import OpenAIAPI, OpenRouterAPI, Provider

from evoprompt import optimize


@prompt(template="answer.j2")
async def answer(instruction: str, text: str, demonstrations: list[dict], *, generated: str) -> str:
    """Generate a task answer from the instruction and fixed demonstrations."""
    return generated


def exact_match(examples, predictions):
    """Case-insensitive, whitespace-normalized match against any reference."""

    def normalize(text):
        return " ".join(text.casefold().split())

    correct = 0
    for example, prediction in zip(examples, predictions):
        targets = example["target"]
        if isinstance(targets, str):
            targets = [targets]
        correct += normalize(prediction) in {normalize(t) for t in targets}
    return correct / len(examples)


def validate_data(data):
    if not isinstance(data, dict):
        raise ValueError("dataset must be a JSON object")
    prompts = data.get("prompts")
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(p, str) or not p.strip() for p in prompts)
    ):
        raise ValueError("prompts must be a nonempty list of nonempty strings")
    if not isinstance(data.get("dev"), list) or not data["dev"]:
        raise ValueError("dev must be a nonempty list")
    splits = {}
    for split in ("demonstrations", "dev", "test"):
        rows = data.get(split, [])
        if not isinstance(rows, list) or (split == "test" and split in data and not rows):
            raise ValueError(f"{split} must be a list (test must be nonempty if supplied)")
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("input"), str)
                or not row["input"].strip()
            ):
                raise ValueError(f"{split} rows need a nonempty string input")
            target = row.get("target")
            targets = [target] if isinstance(target, str) else target
            if (
                not isinstance(targets, list)
                or not targets
                or any(not isinstance(t, str) or not t.strip() for t in targets)
            ):
                raise ValueError(f"{split} targets must be nonempty strings or lists of strings")
            if split == "demonstrations" and not isinstance(target, str):
                raise ValueError("demonstration targets must be single strings")
        splits[split] = {" ".join(row["input"].casefold().split()) for row in rows}
    for left, right in (
        ("demonstrations", "dev"),
        ("demonstrations", "test"),
        ("dev", "test"),
    ):
        if splits[left] & splits[right]:
            raise ValueError(f"input overlap between {left} and {right}; use disjoint splits")


async def evaluate_dataset(instruction, examples, demonstrations, provider, metric):
    predictions = [
        await answer(instruction, row["input"], demonstrations, provider=provider)
        for row in examples
    ]
    score = float(metric(examples, predictions))
    if not math.isfinite(score) or score < 0:
        raise ValueError("metric must return a finite, nonnegative score (higher is better)")
    return {"score": score, "predictions": predictions}


class DemoProvider(Provider):
    """Canned prompt variations and lexical sentiment answers; not a real LLM."""

    def __init__(self, role):
        self.role = role
        self.calls = 0

    async def acall(self, context, *, tools=None, tool_results=None):
        self.calls += 1
        if self.role == "optimizer":
            return (
                "Offline canned response, not model-generated evolution.\n"
                "<prompt>Classify the sentiment as positive or negative. "
                "Return only the label. "
                f"Consider the overall sentiment (variation {self.calls}).</prompt>",
                [],
            )
        text = context.rsplit("Input: ", 1)[-1].split("\nOutput:", 1)[0].casefold()
        label = "negative" if any(word in text for word in ("bad", "awful", "hate")) else "positive"
        return label, []


def make_provider(name, model, timeout, role):
    if name == "demo":
        if model:
            raise ValueError("demo does not accept a model")
        return DemoProvider(role)
    if name not in ("openai", "openrouter"):
        raise ValueError("only text-only API providers are supported by the runner")
    if not model:
        raise ValueError(f"an explicit {role} model is required for {name}")
    provider_type = OpenAIAPI if name == "openai" else OpenRouterAPI
    return provider_type(model=model, timeout=timeout, max_output_tokens=2048)


async def run(args):
    data = json.loads(args.data.read_text())
    validate_data(data)
    if args.population_size < len(data["prompts"]):
        raise ValueError("population-size cannot be smaller than the initial prompts list")
    metric = exact_match
    if args.metric != "exact_match":
        module, separator, name = args.metric.partition(":")
        if not separator or not name:
            raise ValueError("metric must be exact_match or module:function")
        metric = getattr(importlib.import_module(module), name)
        if not callable(metric):
            raise ValueError("metric must be callable")
    target_name = args.target_provider or args.provider
    target_model = args.target_model
    if target_model is None and target_name == args.provider:
        target_model = args.model
    # A demo scorer must never select prompts for a real optimization run.
    if (args.provider == "demo") != (target_name == "demo"):
        raise ValueError("use demo for both providers, or real providers for both")
    if (
        args.provider == "demo"
        and args.data.resolve() != Path(__file__).with_name("demo.json").resolve()
    ):
        raise ValueError("demo supports only demo.json; select real providers for your dataset")
    optimizer = make_provider(args.provider, args.model, args.timeout, "optimizer")
    target = make_provider(target_name, target_model, args.timeout, "target")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "target_provider": target_name,
        "target_model": target_model,
        "python": platform.python_version(),
        "slick": version("slick-ai"),
        "demo": args.provider == "demo",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("evoprompt.py", "run.py")
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    demonstrations = data.get("demonstrations", [])
    with (args.output / "events.jsonl").open("w") as log:

        def record(event):
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()
            if event["event"] == "population":
                print(
                    f"Generation {event['iteration']}: best={event['best_score']:.4f} "
                    f"mean={event['mean_score']:.4f}",
                    flush=True,
                )

        async def evaluate(instruction):
            result = await evaluate_dataset(
                instruction, data["dev"], demonstrations, target, metric
            )
            record({"event": "dev", "prompt": instruction, **result})
            return result["score"]

        try:
            result = await optimize(
                data["prompts"],
                evaluate,
                optimizer,
                algorithm=args.algorithm,
                population_size=args.population_size,
                iterations=args.iterations,
                seed=args.seed,
                cache=not args.no_cache,
                on_event=record,
            )
            # Save the selected prompt even if the later held-out evaluation fails.
            (args.output / "best_prompt.txt").write_text(result["best"]["prompt"] + "\n")
            result["demo"] = args.provider == "demo"
            if "test" in data:
                result["test"] = await evaluate_dataset(
                    result["best"]["prompt"],
                    data["test"],
                    demonstrations,
                    target,
                    metric,
                )
                record(
                    {
                        "event": "test",
                        "prompt": result["best"]["prompt"],
                        **result["test"],
                    }
                )
            (args.output / "result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
        except Exception as exc:
            record({"event": "error", "type": type(exc).__name__, "message": str(exc)})
            raise
    print(f"Saved {args.output / 'result.json'}" + (" (OFFLINE DEMO)" if result["demo"] else ""))


def main(argv=None):
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).with_name("demo.json"))
    p.add_argument("--algorithm", choices=("ga", "de"), default="de")
    p.add_argument("--population-size", type=int, default=10)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--provider", choices=("demo", "openai", "openrouter"), default="demo")
    p.add_argument("--model")
    p.add_argument("--target-provider", choices=("demo", "openai", "openrouter"))
    p.add_argument("--target-model", help="defaults to --model when providers match")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--metric", default="exact_match", help="exact_match or trusted module:function")
    p.add_argument("--no-cache", action="store_true", help="reevaluate repeated prompt strings")
    p.add_argument("--output", type=Path, default=Path("runs/evoprompt"))
    args = p.parse_args(argv)
    if args.population_size < (2 if args.algorithm == "ga" else 3) or args.iterations < 0:
        p.error("require population-size >= 2 for GA / >= 3 for DE and iterations >= 0")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error("timeout must be finite and positive")
    try:
        asyncio.run(run(args))
    except (ValueError, OSError, ImportError, AttributeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
