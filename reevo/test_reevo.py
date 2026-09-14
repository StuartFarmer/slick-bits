"""Offline integration checks: python -m reevo.test_reevo."""

import asyncio
import json
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from slick import prompts
from slick.providers import Provider, ProviderError

from reevo.core import Config, Individual, ReEvo, Task
from reevo.prompts import generate, reflect_long, reflect_pair


class ScriptedProvider(Provider):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    async def acall(self, context, *, tools=None, tool_results=None):
        self.prompts.append(context)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response, []


def code(value):
    return f"def heuristic(x):\n    return {value}\n"


async def check():
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    original_directory = Path.cwd()
    with TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            for black_box in (False, True):
                task = Task("Minimize", "Return a number.", "def heuristic(x):", black_box)
                worse = Individual(0, code(2), "initial", 0, objective=2)
                better = Individual(1, code(1), "initial", 0, objective=1)
                for stage in ("initial", "crossover", "mutation"):
                    rendered = await generate.render(task, stage, worse=worse, better=better)
                    assert "[Improved code]" in rendered and task.signature in rendered
                pair = await reflect_pair.render(task, worse, better)
                assert ("50 words" if black_box else "20 words") in pair
                assert "carry hint" in await reflect_long.render(task, "carry hint", ["new hint"])
        finally:
            os.chdir(original_directory)
    task = Task("Minimize the output", "Return a number.", "def heuristic(x):")

    async def evaluate(source):
        # Only our fixed, numeric return statements are accepted in this test.
        return float(source.strip().rsplit("return ", 1)[1])

    generator = ScriptedProvider([code(8), code(6), code(4), code(3), code(5)])
    reflector = ScriptedProvider(
        [
            "prefer smaller constants",
            "reduce the constant",
            "retain reductions across generations",
        ]
    )
    search = ReEvo(
        task,
        evaluate,
        generator,
        reflector=reflector,
        config=Config(initial_size=2, population_size=2, max_evaluations=6),
    )
    result = await search.run(seed_code=code(10))
    assert len(result.individuals) == 6
    assert [x.stage for x in result.individuals] == [
        "seed",
        "initial",
        "initial",
        "crossover",
        "crossover",
        "mutation",
    ]
    assert result.best.objective == 3  # A worse mutation must not replace the elite.
    assert result.stop_reason == "budget"
    assert "prefer smaller constants" in generator.prompts[2]
    assert "retain reductions" in generator.prompts[4]
    assert code(3).strip() in generator.prompts[4]  # Mutate the post-crossover elite.
    assert len(result.reflections) == 1
    assert result.best_history == [10, 8, 6, 4, 3, 3]

    # Invalid candidates consume budget, and partial initialization is bounded.
    for reply in ["not python", code("nan"), "def other(x):\n    return 1"]:
        result = await ReEvo(
            task,
            evaluate,
            ScriptedProvider([reply]),
            config=Config(initial_size=30, max_evaluations=1),
        ).run()
        assert len(result.individuals) == 1 and result.best is None
        assert result.individuals[0].error

    result = await ReEvo(
        task,
        evaluate,
        ScriptedProvider([code(1), code(1)]),
        config=Config(initial_size=2, max_evaluations=10),
    ).run()
    assert result.stop_reason == "no_distinct_parents" and len(result.individuals) == 2

    result = await ReEvo(
        task,
        evaluate,
        ScriptedProvider([code(1), code(7)]),
        config=Config(initial_size=2, max_evaluations=2, maximize=True),
    ).run()
    assert result.best.objective == 7

    # Prior summaries carry into subsequent generations, with bounded memory.
    generator = ScriptedProvider([code(x) for x in (8, 6, 4, 3, 5, 2, 1, 9)])
    reflector = ScriptedProvider(
        ["smaller", "reduce", "remember reductions", "refine", "improve", "word " * 80]
    )
    result = await ReEvo(
        task,
        evaluate,
        generator,
        reflector=reflector,
        config=Config(initial_size=2, population_size=2, max_evaluations=8),
    ).run()
    assert result.best.objective == 1
    assert "remember reductions" in reflector.prompts[-1]
    assert len(result.reflections[-1]["long_term"].split()) < 50
    for generation in result.reflections:
        for worse, better in generation["pairs"]:
            assert result.individuals[worse].objective > result.individuals[better].objective

    for short, long in ((False, False), (True, False), (False, True)):
        reflector = ScriptedProvider(["hint", "hint"] if short else [])
        result = await ReEvo(
            task,
            evaluate,
            ScriptedProvider([code(x) for x in (5, 4, 3, 2)]),
            reflector=reflector,
            config=Config(
                initial_size=2,
                population_size=2,
                max_evaluations=4,
                short_reflection=short,
                long_reflection=long,
                mutation_rate=0,
            ),
        ).run()
        assert result.best.objective == 2 and len(result.individuals) == 4
        assert result.reflections[0]["long_term"] == ""

    # Disabled crossover becomes repeated elite mutation, with no invented pairs.
    result = await ReEvo(
        task,
        evaluate,
        ScriptedProvider([code(5), code(4), code(3)]),
        config=Config(initial_size=1, population_size=2, max_evaluations=3, crossover_rate=0),
    ).run()
    assert result.best.objective == 3 and len(result.individuals) == 3

    for config in [
        dict(max_evaluations=0),
        dict(mutation_rate=math.nan),
        dict(crossover_rate=0, mutation_rate=0),
    ]:
        try:
            Config(**config)
        except ValueError:
            pass
        else:
            raise AssertionError(config)

    try:
        await ReEvo(
            task,
            evaluate,
            ScriptedProvider([ProviderError("offline")]),
            config=Config(max_evaluations=1),
        ).run()
    except ProviderError:
        pass
    else:
        raise AssertionError("provider failures must propagate")

    # A raw/fenced response retains helpers and normal Python syntax.
    source = "import math\n\ndef helper(x):\n    return x\n\n" + code(2)
    result = await ReEvo(
        task,
        evaluate,
        ScriptedProvider([f"```python\n{source}```"]),
        config=Config(max_evaluations=1),
    ).run()
    assert result.best.objective == 2 and "def helper" in result.best.code
    json.dumps(result.to_dict(), allow_nan=False)
    print("ReEvo core checks passed.")

    from reevo.tsp import TSPEvaluator, aco, make_instances, task_and_seed

    square = np.array(
        [
            [0.0, 1.0, 2**0.5, 1.0],
            [1.0, 0.0, 1.0, 2**0.5],
            [2**0.5, 1.0, 0.0, 1.0],
            [1.0, 2**0.5, 1.0, 0.0],
        ]
    )
    eta = np.where(square == 1, 1.0, 0.0)
    assert aco(square, eta, ants=4, iterations=2, seed=0) == 4
    assert np.isfinite(aco(square, np.zeros((4, 4)), ants=4, iterations=2, seed=0))
    assert np.array_equal(make_instances(2, 5, seed=0), make_instances(2, 5, seed=0))
    assert not np.array_equal(make_instances(2, 5, seed=0), make_instances(2, 5, seed=1))
    instances = np.array([square])
    for black_box in (False, True):
        tsp_task, seed = task_and_seed(black_box)
        evaluator = TSPEvaluator(instances, black_box=black_box, ants=4, iterations=2)
        value = await evaluator(seed)
        assert 4 <= value <= 4 + 2 * 2**0.5
        assert value == await evaluator(seed)
        if black_box:
            text = tsp_task.description + tsp_task.function_description + tsp_task.signature + seed
            assert not any(word in text.lower() for word in ("distance", "tsp", "salesman"))
        signature = tsp_task.signature
        for body in [
            "return np.ones((2, 2))",
            "raise ValueError('broken')",
            "return np.full_like("
            + ("edge_attr[:, 0]" if black_box else "distance_matrix")
            + ", np.nan)",
        ]:
            try:
                await evaluator("import numpy as np\n" + signature + "\n    " + body)
            except ValueError:
                pass
            else:
                raise AssertionError(body)
    fast = TSPEvaluator(instances, timeout=1, ants=4, iterations=2)
    try:
        await fast("def heuristics(distance_matrix):\n    while True: pass")
    except TimeoutError:
        pass
    else:
        raise AssertionError("nonterminating candidates must time out")

    with TemporaryDirectory() as directory:
        output = Path(directory) / "demo"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "reevo",
            "--output",
            str(output),
            "--max-evaluations",
            "8",
            "--initial-size",
            "3",
            "--population-size",
            "2",
            "--nodes",
            "8",
            "--instances",
            "2",
            "--test-instances",
            "2",
            "--ants",
            "4",
            "--aco-iterations",
            "3",
            "--black-box",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        assert proc.returncode == 0, (stdout, stderr)
        saved = json.loads((output / "result.json").read_text())
        assert len(saved["individuals"]) <= 8
        assert saved["test_objective"] > 0
        assert (output / "best.py").read_text() == saved["best"]["code"]
        assert saved["calls"] and saved["provider"] == "demo"
    print("TSP checks passed (ACO, validation, timeout, black-box prompts, saved CLI run).")


if __name__ == "__main__":
    asyncio.run(check())
