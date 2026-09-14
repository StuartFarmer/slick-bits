"""Offline behavioral checks: python test_evoprompt.py (no credentials needed)."""

import asyncio
import contextlib
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from slick import prompts
from slick.providers import Provider

from evoprompt import de_offspring, extract_prompt, ga_offspring, optimize, variation
from run import answer, evaluate_dataset, exact_match, main, make_provider, validate_data


class ScriptedProvider(Provider):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.contexts = []

    async def acall(self, context, *, tools=None, tool_results=None):
        self.contexts.append(context)
        return next(self.responses), []


async def check():
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    assert "Prompt 2: second" in await ga_offspring.render("first", "second")
    assert "Basic Prompt: target" in await de_offspring.render("first", "second", "best", "target")
    assert "Instruction: initial" in await variation.render("initial")
    for demonstrations in ([], [{"input": "example", "target": "label"}]):
        rendered = await answer.render("Classify.", "current", demonstrations)
        assert rendered.endswith("Input: current\nOutput:")
        assert ("Input: example\nOutput: label" in rendered) == bool(demonstrations)

    scores = {"a": 1, "b": 2, "c": 3, "weak": 0, "strong": 5, "middle": 2.5}
    evaluated = []

    async def evaluate(instruction):
        evaluated.append(instruction)
        return scores[instruction]

    provider = ScriptedProvider(
        [f"Crossover and mutation. <prompt>{p}</prompt>" for p in ("weak", "strong", "middle")]
    )
    events = []
    ga = await optimize(
        ["a", "b", "c"],
        evaluate,
        provider,
        algorithm="ga",
        iterations=1,
        seed=7,
        on_event=events.append,
    )
    assert ga["population"] == [
        {"prompt": "strong", "score": 5},
        {"prompt": "c", "score": 3},
        {"prompt": "middle", "score": 2.5},
    ], "GA must retain the global top N from parents and offspring"
    assert ga["best"] == ga["population"][0]
    assert ga["initial_best"] == {"prompt": "c", "score": 3}
    assert len(evaluated) == 6 and ga["optimizer_calls"] == 3
    assert all("Cross over" in c and "Mutate" in c for c in provider.contexts)
    evolution = [e for e in events if e["event"] == "evolution"]
    assert all(set(e["parents"]) <= {"a", "b", "c"} for e in evolution)

    provider = ScriptedProvider([f"<prompt>{p}</prompt>" for p in ("strong", "weak", "middle")])
    events = []
    de = await optimize(
        ["a", "b", "c"],
        evaluate,
        provider,
        algorithm="de",
        iterations=1,
        seed=7,
        on_event=events.append,
    )
    assert de["population"] == [
        {"prompt": "strong", "score": 5},
        {"prompt": "b", "score": 2},
        {"prompt": "c", "score": 3},
    ], "DE must compare each trial only to its own target"
    for i, event in enumerate(e for e in events if e["event"] == "evolution"):
        donor1, donor2, best, target = event["parents"]
        assert target == "abc"[i] and best == "c", "DE uses a generation snapshot"
        assert len({donor1, donor2, target}) == 3
    assert all("different parts" in c and "Prompt 3: c" in c for c in provider.contexts)

    # Only the positive-fitness individual may reproduce; ties retain incumbents.
    async def sparse_score(p):
        return int(p == "c")

    events = []
    sparse = await optimize(
        ["a", "b", "c"],
        sparse_score,
        ScriptedProvider(["<prompt>c</prompt>"] * 6),
        algorithm="ga",
        iterations=2,
        on_event=events.append,
    )
    assert all(e["parents"] == ["c", "c"] for e in events if e["event"] == "evolution")
    assert sparse["evaluations"] == 3 and sparse["cache_hits"] == 6
    assert len(sparse["population"]) == 3
    assert [h["best_score"] for h in sparse["history"]] == [1, 1, 1]

    # Uniform fallback for zero fitness, local seeded RNG, and optional caching.
    async def zero(p):
        return 0

    outputs = []
    for _ in range(2):
        provider = ScriptedProvider(["<prompt>a</prompt>"] * 6)
        result = await optimize(
            ["a", "b", "c"],
            zero,
            provider,
            algorithm="ga",
            iterations=2,
            seed=9,
            cache=False,
        )
        assert result["evaluations"] == 9 and result["cache_hits"] == 0
        outputs.append(provider.contexts)
    assert outputs[0] == outputs[1]

    # Initial manual prompts can be filled with LLM-generated variations.
    provider = ScriptedProvider(["<prompt>b</prompt>", "<prompt>c</prompt>"])
    result = await optimize(["a"], evaluate, provider, population_size=3, iterations=0)
    assert result["best"]["prompt"] == "c" and result["optimizer_calls"] == 2
    assert result["evaluations"] == 3 and len(result["history"]) == 1

    for bad in (
        "unmarked",
        "<prompt> </prompt>",
        "<prompt>x</prompt><prompt>y</prompt>",
        "<prompt><prompt>x</prompt>",
    ):
        try:
            extract_prompt(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted malformed offspring: {bad!r}")
    assert extract_prompt("Explanation\n<prompt> hello\nworld </prompt>") == "hello\nworld"

    for options in (
        {"algorithm": "other"},
        {"iterations": -1},
        {"iterations": 1.5},
        {"population_size": 2},
        {"population_size": True},
    ):
        try:
            await optimize(["a", "b", "c"], evaluate, ScriptedProvider([]), **options)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid options: {options}")
    for value in (-1, math.nan, math.inf):

        async def invalid_score(p):
            return value

        try:
            await optimize(["a", "b", "c"], invalid_score, ScriptedProvider([]), iterations=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid fitness: {value}")
    print(
        "EvoPROMPT checks passed: GA/DE selection, snapshots, initialization, caching, validation."
    )

    data = {
        "prompts": ["Choose a label.", "Categorize the input.", "Classify the text."],
        "demonstrations": [{"input": "demo input", "target": "negative"}],
        "dev": [{"input": "dev input", "target": ["positive", "yes"]}],
        "test": [{"input": "HELDOUT SECRET", "target": "positive"}],
    }
    validate_data(data)
    provider = ScriptedProvider([" YES "])
    result = await evaluate_dataset(
        "Classify.", data["dev"], data["demonstrations"], provider, exact_match
    )
    assert result == {"score": 1.0, "predictions": [" YES "]}
    assert "demo input" in provider.contexts[0]
    assert "HELDOUT SECRET" not in provider.contexts[0]
    assert "yes" not in provider.contexts[0], "dev answers must not reach task inference"
    assert exact_match([{"target": "positive"}], ["not positive"]) == 0
    custom = await evaluate_dataset(
        "Summarize.",
        data["dev"],
        [],
        ScriptedProvider(["a summary"]),
        lambda rows, pred: 42.0,
    )
    assert custom["score"] == 42.0
    for bad in (
        {**data, "dev": []},
        {**data, "test": data["dev"]},
        {**data, "demonstrations": data["test"]},
        {**data, "dev": [{"input": "x", "target": []}]},
    ):
        try:
            validate_data(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dataset or split leakage accepted")


def check_cli():
    try:
        make_provider("codex", None, 30, "target")
    except ValueError:
        pass
    else:
        raise AssertionError("tool-enabled provider could read held-out dataset files")
    provider = make_provider("openai", "test-model", 30, "target")
    payload = provider._encode("only the task input", {}, [])
    assert payload["input"] == "only the task input" and "tools" not in payload
    with TemporaryDirectory() as directory:
        for algorithm in ("ga", "de"):
            output = Path(directory) / algorithm
            args = [
                "--algorithm",
                algorithm,
                "--iterations",
                "1",
                "--population-size",
                "4",
                "--output",
                str(output),
            ]
            main(args)
            result = json.loads((output / "result.json").read_text())
            assert len(result["population"]) == 4 and result["optimizer_calls"] == 6
            assert result["test"]["score"] == 1
            events = [
                json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
            ]
            assert events[-1]["event"] == "test"
            before = (output / "result.json").read_bytes()
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    main(args)
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError("existing run was overwritten")
            assert (output / "result.json").read_bytes() == before
    print(
        "Runner checks passed: Slick inference, metrics, split isolation, both CLIs, saved artifacts."
    )


if __name__ == "__main__":
    asyncio.run(check())
    check_cli()
