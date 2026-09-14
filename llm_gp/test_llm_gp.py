"""Offline integration checks: python -m llm_gp.test_llm_gp."""

import asyncio
import json
import math
import os
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from slick import prompts
from slick.providers import Provider, ProviderError

from llm_gp.core import (
    Config,
    evaluate,
    evolve,
    parse_expression,
    split_data,
    tree_depth,
)
from llm_gp.operators import (
    DemoProvider,
    MeasuredOpenAI,
    Operators,
    choice_prompt,
    crossover_prompt,
    initialize_prompt,
    mutation_prompt,
)
from llm_gp.run import usage_summary


class ScriptedProvider(Provider):
    def __init__(self, responses):
        self.responses = iter(responses)

    async def acall(self, context, *, tools=None, tool_results=None):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response, []


async def check():
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    original_directory = Path.cwd()
    with TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            assert "# Output Format" in await initialize_prompt.render()
            assert "Rephrase" in await mutation_prompt.render("x0", [])
            assert "Recombine" in await crossover_prompt.render(["x0", "x1"], [])
            for operation in ("selection", "replacement", "best"):
                rendered = await choice_prompt.render(operation, [], 1, [])
                assert (
                    "Repeated IDs are allowed."
                    if operation == "selection"
                    else "Do not repeat IDs."
                ) in rendered
        finally:
            os.chdir(original_directory)
    data = [(2.0, 3.0, 13.0), (-2.0, 0.0, 4.0)]
    assert evaluate("x0*x0 + x1*x1", data) == 0
    assert evaluate("0", data) == 92.5
    for code in [
        "__import__('os').getcwd()",
        "x0**2",
        "x0/1",
        "True",
        "2",
        "x2",
        "[x0]",
        "x0.real",
        "1 if x0 else 0",
        "1" * 5000,
    ]:
        assert math.isinf(evaluate(code, data)), code
    split = split_data(42)
    assert [len(split[k]) for k in ("train", "test", "holdout")] == [67, 29, 25]
    assert len({tuple(row) for rows in split.values() for row in rows}) == 121
    assert split == split_data(42)

    for variant in ("gp", "random", "llm-random", "llm-gp-mu-xo", "llm-gp"):
        config = Config(
            variant=variant,
            population_size=5,
            generations=4,
            crossover_rate=1,
            mutation_rate=1,
        )
        ops = Operators(DemoProvider(), random.Random(7))
        result = await evolve(config, split, ops)
        assert result["evaluations"] == 20, variant
        assert len(result["history"]) == 4
        assert all(len(g["population"]) == 5 for g in result["history"])
        for generation in result["history"]:
            for individual in generation["population"]:
                tree = parse_expression(individual["expression"])
                if variant == "gp":
                    assert tree_depth(tree) <= config.max_depth
        assert result["best"]["fitness"] == evaluate(
            result["best"]["expression"], result["training_data"]
        )
        if variant in ("gp", "llm-gp-mu-xo"):
            scores = [g["best_fitness"] for g in result["history"]]
            assert scores == sorted(scores, reverse=True), scores
        if variant.startswith("llm"):
            assert ops.calls and all(c["prompt"] and c["response"] for c in ops.calls)
            assert not any(c["error"] for c in ops.calls)
        else:
            assert not ops.calls

    ops = Operators(
        ScriptedProvider(
            [
                '{"expression":"__import__("os")"}',
                '{"new_expression":"x0**2"}',
                '{"expressions":["x0"]}',
                '{"individuals":[999]}',
                '{"individuals":[0,0]}',
            ]
        ),
        random.Random(1),
    )
    assert await ops.initialize() == "0"
    assert await ops.mutate("x1", ["x0"]) == "x1"
    assert await ops.crossover(["x0", "x1"], []) == ["x0", "x1"]
    population = [
        {"expression": "x0", "fitness": 3},
        {"expression": "x1", "fitness": 1},
    ]
    assert len(await ops.choose("selection", population, 2)) == 2
    replacement = await ops.choose("replacement", population, 2)
    assert len({id(p) for p in replacement}) == 2
    assert all(c["error"] for c in ops.calls)

    ops = Operators(
        ScriptedProvider(
            [
                ProviderError("transient"),
                '{"expression":"x1"}',
            ]
        ),
        random.Random(1),
        retry_delay=0,
    )
    assert await ops.initialize() == "x1"
    assert len(ops.calls) == 2 and ops.calls[0]["error"]

    # A valid model choice may discard the optimum: preserve measured best
    # separately and independently evaluate the model's worse final designation.
    ops = Operators(
        ScriptedProvider(
            [
                '{"expression":"x0"}',
                '{"expression":"x0*x0 + x1*x1"}',
                '{"individuals":[0,0]}',
                '{"individuals":[0,2]}',
                '{"individuals":[0]}',
            ]
        ),
        random.Random(0),
    )
    result = await evolve(
        Config(
            variant="llm-gp",
            population_size=2,
            generations=2,
            crossover_rate=0,
            mutation_rate=0,
        ),
        split,
        ops,
    )
    assert result["evaluations"] == 4 and result["best"]["fitness"] == 0
    assert result["designated_best"]["expression"] == "x0"
    assert result["designated_holdout_mse"] > 0 and result["best_holdout_mse"] == 0

    for response in (
        '{"individuals":[true]}',
        '{"individuals":["0"]}',
        '{"individuals":[-1]}',
        "not json",
    ):
        ops = Operators(ScriptedProvider([response]), random.Random(0))
        assert (await ops.choose("best", population, 1))[0]["expression"] == "x1"
        assert ops.calls[0]["fallback"]

    ops = Operators(
        ScriptedProvider([ProviderError("failure")] * 3),
        random.Random(0),
        retry_delay=0,
    )
    assert await ops.initialize() == "0"
    assert len(ops.calls) == 3 and ops.calls[-1]["fallback"]

    class SlowProvider(Provider):
        async def acall(self, context, **kwargs):
            await asyncio.sleep(1)
            return '{"expression":"1"}', []

    ops = Operators(SlowProvider(), random.Random(0), timeout=0.001, retries=0)
    assert await ops.initialize() == "0" and ops.calls[0]["fallback"]

    # Exercise the real Slick API encode/decode and measured usage boundary;
    # only the external SDK transport is replaced.
    from unittest.mock import AsyncMock, patch

    from slick.providers import OpenAIAPI

    backend = MeasuredOpenAI("test-model", temperature=0.8)
    response = SimpleNamespace(
        status="completed",
        model="resolved-test-model",
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 100, "output_tokens": 20}),
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text='{"expression":"x0"}')],
            )
        ],
    )
    with patch.object(OpenAIAPI, "_asend", AsyncMock(return_value=response)) as send:
        ops = Operators(backend, random.Random(0))
        assert await ops.initialize() == "x0"
        assert send.call_args.args[0]["temperature"] == 0.8
    assert ops.calls[0]["resolved_model"] == "resolved-test-model"
    assert usage_summary(ops.calls, 1, 2)["estimated_cost_usd"] == 0.00014
    assert usage_summary(ops.calls)["estimated_cost_usd"] is None
    backend.temperature = None
    assert "temperature" not in backend._encode("hello", {}, [])

    ops = Operators(DemoProvider(), random.Random(1), max_calls=2)
    result = await evolve(Config(population_size=5, generations=2), split, ops)
    assert result["stop_reason"] == "budget" and len(ops.calls) == 2
    assert result["evaluations"] == 2
    json.dumps(result, allow_nan=False)
    print("LLM_GP checks passed (all five variants, evaluation, fallbacks, budgets).")


if __name__ == "__main__":
    asyncio.run(check())
