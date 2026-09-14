"""Run APEX with Slick on JSON/JSONL examples, or a credential-free canned demo."""

import argparse
import asyncio
import hashlib
import json
import random
import re
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import numpy as np
from slick import prompt, prompts
from slick.providers import CodexCLI, LiteLLMAPI, Provider

from apex import Config, Document, optimize


@prompt(template="predict.j2")
async def predict(prefix: str, question: str, suffix: str, *, generated: str) -> str:
    """Generate the target answer for the assembled task text."""
    return generated


def answer(text, mode):
    text = text.strip()
    if mode == "number":
        # GSM8K gold uses ####; model answers use the final numeric token.
        tail = text.rsplit("####", 1)[-1].replace(",", "")
        numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", tail)
        return str(Decimal(numbers[-1]).normalize()) if numbers else None
    text = re.split(r"(?:the )?(?:final )?answer\s*(?:is|:)\s*", text, flags=re.IGNORECASE)[
        -1
    ].strip()
    if mode == "choice":
        choices = re.findall(r"\(([A-Za-z])\)", text)
        if choices:
            return choices[-1].upper()
        return text.strip(". ").upper() if re.fullmatch(r"[A-Za-z][.]?", text) else None
    if mode == "dyck":
        return re.sub(r"\s+", "", text) if re.fullmatch(r"[\s\[\](){}<>]+", text) else None
    return text.rstrip(".").strip().casefold()


def load_examples(path):
    raw = path.read_text(encoding="utf-8")
    data = (
        [json.loads(line) for line in raw.splitlines() if line.strip()]
        if path.suffix == ".jsonl"
        else json.loads(raw)
    )
    if isinstance(data, dict):
        data = data.get("examples")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a nonempty list or BBH {{examples: [...]}}")
    result = []
    for row in data:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: each example must be an object")
        question = row.get("input", row.get("question"))
        target = row.get("target", row.get("answer"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}: each example needs nonempty input/question text")
        if isinstance(target, bool) or not isinstance(target, (str, int, float)):
            raise ValueError(f"{path}: each example needs a string or numeric target/answer")
        if not str(target).strip():
            raise ValueError(f"{path}: empty target")
        result.append(dict(input=question, target=str(target)))
    return result


def check_split(train, test, mode):
    if not train or not test:
        raise ValueError("training and test sets must both be nonempty")
    keys = [
        set(" ".join(row["input"].split()).casefold() for row in split) for split in (train, test)
    ]
    if keys[0] & keys[1]:
        raise ValueError("training and test inputs overlap")
    for row in train + test:
        if answer(row["target"], mode) in (None, ""):
            raise ValueError(f"target cannot be parsed in {mode} mode: {row['target']!r}")


async def evaluate(text, examples, provider, separator, suffix, mode):
    correct, outputs = 0, []
    for row in examples:
        output = await predict(text + separator, row["input"], suffix, provider=provider)
        predicted, expected = answer(output, mode), answer(row["target"], mode)
        matched = predicted is not None and predicted == expected
        correct += matched
        outputs.append(
            dict(
                input=row["input"],
                target=row["target"],
                output=output,
                predicted=predicted,
                correct=matched,
            )
        )
    return dict(score=correct / len(examples), outputs=outputs)


class DemoProvider(Provider):
    """Synthetic behavior, deliberately sensitive to a known rephrasing."""

    async def acall(self, context, *, tools=None, tool_results=None):
        if "<rephrased>" in context:
            return "Compute carefully.", []
        left, right = map(int, re.findall(r"What is (\d+) \+ (\d+)\?", context)[-1])
        return str(left + right if "Compute carefully." in context else 0), []


def demo_encode(texts):
    """Canned features, not a substitute for semantic embeddings in experiments."""
    return np.array([[1.0, len(text) / 100.0] for text in texts])


def make_encoder(model):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model)
    return lambda texts: encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)


async def run(args):
    config = Config(
        iterations=args.iterations,
        beam_size=args.beam_size,
        alpha=args.alpha,
        regularization=args.regularization,
        random_probability=args.random_probability,
        history_limit=args.history_limit,
        history_distance=args.history_distance,
        guided_mutation=not args.no_guided_mutation,
        seed=args.seed,
    )
    if args.provider == "demo":
        document = Document.split("Add the integers.")
        rows = [dict(input=f"What is {i} + {i + 1}?", target=str(2 * i + 1)) for i in range(1, 7)]
        train, test = rows[:4], rows[4:]
        target = mutator = DemoProvider()
        encode = demo_encode
    else:
        if not args.prompt:
            raise ValueError("--prompt is required for real providers")
        source = args.prompt.read_bytes().decode("utf-8")
        document = (
            Document.from_json(json.loads(source))
            if args.prompt.suffix == ".json"
            else Document.split(source)
        )
        if args.data:
            rows = load_examples(args.data)
            random.Random(args.seed).shuffle(rows)
            train, test = rows[: len(rows) // 2], rows[len(rows) // 2 :]
        elif args.train and args.test:
            train, test = load_examples(args.train), load_examples(args.test)
        else:
            raise ValueError("provide --data for a 50/50 split, or both --train and --test")
        if args.provider == "litellm":
            if not args.model:
                raise ValueError("--model is required for litellm")
            target = LiteLLMAPI(
                args.model,
                timeout=args.timeout,
                options={"temperature": 0, "max_tokens": args.max_tokens},
            )
            mutator = LiteLLMAPI(
                args.mutator_model or args.model,
                timeout=args.timeout,
                options={"temperature": 0.5, "max_tokens": args.max_tokens},
            )
        else:
            target = CodexCLI(model=args.model, timeout=args.timeout)
            mutator = CodexCLI(model=args.mutator_model or args.model, timeout=args.timeout)
        encode = None  # load weights only after validating data and reserving output
    check_split(train, test, args.answer_mode)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "initial.txt").write_text(document.text, encoding="utf-8")
    (args.output / "fragments.json").write_text(
        json.dumps(
            [
                dict(text=text, mutable=i in document.mutable)
                for i, text in enumerate(document.parts)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata = dict(
        config=asdict(config),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        demo=args.provider == "demo",
        train=train,
        test=test,
        slick_path=__import__("slick").__file__,
        prompt_sha256=hashlib.sha256(document.text.encode()).hexdigest(),
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if encode is None:
        encode = make_encoder(args.encoder)

    with (args.output / "evaluations.jsonl").open("w", encoding="utf-8") as evaluations:

        async def training_score(text):
            result = await evaluate(
                text, train, target, args.separator, args.suffix, args.answer_mode
            )
            evaluations.write(json.dumps(dict(prompt=text, **result)) + "\n")
            evaluations.flush()
            return result["score"]

        with (args.output / "history.jsonl").open("w", encoding="utf-8") as trajectory:

            def record(entry):
                trajectory.write(json.dumps(entry, allow_nan=False) + "\n")
                trajectory.flush()
                print(
                    f"iteration={entry['iteration']} evaluations={entry['evaluations']} "
                    f"best={entry.get('best_score', entry['score']):.3f}",
                    flush=True,
                )

            result = await optimize(document, training_score, mutator, encode, config, record)

    # Persist the selected prompt before test calls; test failure cannot lose search results.
    (args.output / "best.txt").write_text(result["best"]["prompt"], encoding="utf-8")
    (args.output / "search.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    original_test = await evaluate(
        document.text, test, target, args.separator, args.suffix, args.answer_mode
    )
    best_test = (
        original_test
        if result["best"]["prompt"] == document.text
        else await evaluate(
            result["best"]["prompt"], test, target, args.separator, args.suffix, args.answer_mode
        )
    )
    result["test"] = dict(original=original_test, best=best_test)
    result["calls"] = dict(
        mutation=config.iterations,
        train=result["evaluations"] * len(train),
        test=len(test) * (1 if best_test is original_test else 2),
    )
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    label = "CANNED DEMO" if args.provider == "demo" else "APEX"
    print(
        f"{label}: train {result['initial_score']:.3f} -> {result['best']['score']:.3f}; "
        f"test {original_test['score']:.3f} -> {best_test['score']:.3f}\n{args.output / 'best.txt'}"
    )
    return result


def main(argv=None):
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=("demo", "litellm", "codex"), default="demo")
    p.add_argument("--model")
    p.add_argument("--mutator-model")
    p.add_argument("--encoder", default="sentence-transformers/sentence-t5-base")
    p.add_argument("--prompt", type=Path, help="plain text or explicit fragment JSON")
    p.add_argument("--data", type=Path, help="dataset to split equally using --seed")
    p.add_argument("--train", type=Path)
    p.add_argument("--test", type=Path)
    p.add_argument("--answer-mode", choices=("text", "choice", "number", "dyck"), default="text")
    p.add_argument("--separator", default="\n\nQ: ")
    p.add_argument("--suffix", default="\nA: Let's think step by step.")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--beam-size", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--regularization", type=float, default=1.0)
    p.add_argument("--random-probability", type=float, default=0.5)
    p.add_argument("--history-limit", type=int, default=4)
    p.add_argument("--history-distance", type=float, default=0.5)
    p.add_argument("--no-guided-mutation", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--output", type=Path, default=Path("runs/apex-demo"))
    args = p.parse_args(argv)
    if args.data and (args.train or args.test):
        p.error("--data cannot be combined with --train/--test")
    if args.timeout <= 0 or args.max_tokens <= 0:
        p.error("--timeout and --max-tokens must be positive")
    if args.provider == "demo" and any(
        (args.prompt, args.data, args.train, args.test, args.model, args.mutator_model)
    ):
        p.error("custom data/prompts/models require --provider litellm or codex")
    try:
        asyncio.run(run(args))
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
