"""Run with: .venv/bin/python test_ael.py [--docker] (no model credentials)."""

import asyncio
import itertools
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import openai
from slick import prompts
from slick.providers import OpenRouterAPI

from ael import (
    Config,
    DemoProvider,
    Evaluator,
    evolve,
    make_dataset,
    parse_response,
    propose,
)
from tsp import construct_tour, distances, exact_length, greedy, paper_heuristic


def main():
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    matrix = distances(square)
    assert construct_tour(greedy, matrix) == [0, 1, 2, 3, 0]
    assert construct_tour(paper_heuristic, distances([[0, 0], [1, 0]])) == [0, 1, 0]
    # Removing subtour elimination would choose two disconnected triangles.
    points = [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]]
    d = distances(points)
    brute = min(
        sum(d[a, b] for a, b in zip((0, *p), (*p, 0))) for p in itertools.permutations(range(1, 6))
    )
    assert math.isclose(exact_length(d, 10), brute)
    assert exact_length(matrix, 10) == 4
    try:
        construct_tour(lambda *args: 0, matrix)
    except ValueError:
        pass
    else:
        raise AssertionError("revisited nodes must be rejected")

    good = "<start>Nearest node.<end>\n```python\ndef select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):\n    return min(unvisited_nodes, key=lambda n: distance_matrix[current_node][n])\n```"
    proposal = parse_response(good)

    # Exercise Slick + the actual SDK, replacing only HTTP transport.
    async def check_openrouter():
        def respond(request):
            assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
            body = json.loads(request.content)
            assert body["model"] == "test/model"
            assert "select_next_node" in body["messages"][0]["content"]
            return httpx.Response(
                200,
                json={
                    "id": "offline",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test/model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": good},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        original_client = openai.AsyncOpenAI

        def client(**kwargs):
            return original_client(
                **kwargs,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            )

        with patch.object(openai, "AsyncOpenAI", client):
            result = await propose(
                "initialization",
                [],
                provider=OpenRouterAPI("test/model", api_key="offline-test"),
            )
        assert parse_response(result).code == proposal.code

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        template = (prompts.TEMPLATE_ROOT / "propose.j2").read_text()
        (root / "propose.j2").write_text(template + "\nLOADED_FROM_TEMPLATE")
        with patch.object(prompts, "TEMPLATE_ROOT", root):
            assert "LOADED_FROM_TEMPLATE" in asyncio.run(propose.render("initialization", []))
    asyncio.run(check_openrouter())
    evaluator = Evaluator(backend="trusted", timeout=5)
    instances = [{"coordinates": square, "reference_length": 4.0}]
    score = evaluator(proposal.code, instances)
    assert score["fitness"] == 0 and score["lengths"] == [4]
    score = evaluator(proposal.code, instances + [{"coordinates": square, "reference_length": 2}])
    assert score["fitness"] == 50, "fitness is the mean of instance gaps, not a ratio of means"
    bad = proposal.code.replace(
        "return min(unvisited_nodes, key=lambda n: distance_matrix[current_node][n])",
        "return current_node",
    )
    try:
        evaluator(bad, instances)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid tours must not earn a fitness")
    loop = proposal.code[: proposal.code.index("    return")] + "    while True: pass\n"
    try:
        Evaluator(backend="trusted", timeout=0.3)(loop, instances)
    except TimeoutError:
        pass
    else:
        raise AssertionError("candidate deadline must be enforced")
    if "--docker" in sys.argv:
        isolated = Evaluator(timeout=10)
        # These checks execute in the container, with no host credential reads.
        boundary = """import os, socket
assert os.environ.get("AEL_PRIVATE_TEST") is None
try:
    open("/ael-write-probe", "w")
except OSError:
    pass
else:
    raise AssertionError("container filesystem is writable")
try:
    socket.create_connection(("1.1.1.1", 53), timeout=0.2)
except OSError:
    pass
else:
    raise AssertionError("container has external network access")
"""
        with patch.dict(os.environ, {"AEL_PRIVATE_TEST": "must-stay-on-host"}):
            assert isolated(boundary + proposal.code, instances)["fitness"] == 0
        try:
            Evaluator(timeout=2)(loop, instances)
        except TimeoutError:
            pass
        else:
            raise AssertionError("Docker evaluation deadline must be enforced")
        print("Docker boundary and timeout checks passed.")
    for response in (
        "nonsense",
        good.replace("select_next_node", "wrong_name"),
        good.replace("distance_matrix):", "distance_matrix, required):"),
    ):
        try:
            parse_response(response)
        except (ValueError, SyntaxError):
            pass
        else:
            raise AssertionError("malformed candidates must be rejected")

    data = make_dataset([6], 2, 42, "exact", 10)
    assert data == make_dataset([6], 2, 42, "exact", 10)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "evolution"
        cfg = Config(
            population_size=3,
            generations=2,
            parents=2,
            offspring=2,
            crossover=1,
            mutation=1,
            seed=8,
        )
        result = asyncio.run(evolve(DemoProvider(), evaluator, data, cfg, output))
        assert len(result["population"]) == 3
        assert len(result["initial"]) == 3
        assert len(result["history"]) == 3
        best = [g["best"] for g in result["history"]]
        assert best == sorted(best, reverse=True), "elitism must retain the best"
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        assert sum(e["operation"] == "crossover" for e in events) == 12
        assert sum(e["operation"] == "mutation" for e in events) == 12
        # All parent IDs within a generation must come from its starting population.
        for generation in (1, 2):
            parents = set(result["history"][generation - 1]["population"])
            assert all(
                set(e["parents"]) <= parents
                for e in events
                if e["generation"] == generation and e["operation"] == "crossover"
            )
        assert (output / "best.py").is_file()
        assert json.loads((output / "result.json").read_text())["best"]["fitness"] == best[-1]

        no_variation = Config(
            population_size=2,
            generations=1,
            parents=2,
            offspring=2,
            crossover=0,
            mutation=0,
        )
        result = asyncio.run(
            evolve(DemoProvider(), evaluator, data, no_variation, Path(tmp) / "clones")
        )
        assert result["history"][0]["best"] == result["history"][1]["best"]
        assert len((Path(tmp) / "clones/events.jsonl").read_text().splitlines()) == 2

        class ScriptedProvider(DemoProvider):
            def __init__(self, replies):
                self.replies = iter(replies)

            async def acall(self, context, **kwargs):
                return next(self.replies), []

        far = good.replace("return min(", "return max(")
        improved = asyncio.run(
            evolve(
                ScriptedProvider(["invalid", far, far, good, "invalid"]),
                evaluator,
                {"reference": "exact square", "instances": instances},
                Config(population_size=2, generations=1, mutation=0),
                Path(tmp) / "improvement",
            )
        )
        assert improved["history"][0]["best"] > 20
        assert improved["history"][1]["best"] == 0
        assert len(improved["population"]) == 2, "invalid offspring must not shrink the population"

    print(
        "AEL checks passed: exact TSP, valid tours, deadlines, Slick prompts, evolution, artifacts."
    )


if __name__ == "__main__":
    main()
