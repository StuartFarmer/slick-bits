"""Offline correctness checks: .venv/bin/python test_optimizer.py."""

import asyncio
import itertools
import math
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
from slick import prompts

from cmsa import (
    VARIANTS,
    Config,
    adapt,
    cmsa,
    generate_solution,
    graph_neighbors,
    independent,
    read_graph,
    selection_probabilities,
    solve_reduced,
)
from experiment import make_graph, mean_ranks
from improve import DemoProvider, baseline_source, dialogue, propose, validate_candidate


def check():
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    p = selection_probabilities([0, 1], [set(), set()], [-1, 8], False)
    assert math.isclose(sum(p), 1) and p[0] > p[1] > 0
    q = selection_probabilities([0, 1], [set(), set()], [-1, 8], True)
    assert math.isclose(sum(q), 1) and abs(q[0] - q[1]) < abs(p[0] - p[1])
    assert selection_probabilities([0], [set()], [-1], True) == [1.0]
    ages = [-1, 0, 1, 2]
    adapt(ages, {1}, 3)
    assert ages == [-1, 0, 2, -1]

    graphs = [
        nx.empty_graph(0),
        nx.empty_graph(6),
        nx.complete_graph(6),
        nx.path_graph(7),
        nx.cycle_graph(7),
        nx.star_graph(6),
    ]
    graphs += [nx.gnp_random_graph(8, 0.4, seed=i) for i in range(5)]
    for graph in graphs:
        neighbors = graph_neighbors(graph)
        optimum = max(
            len(s)
            for k in range(len(graph) + 1)
            for s in map(set, itertools.combinations(range(len(graph)), k))
            if independent(neighbors, s)
        )
        solution, optimal = solve_reduced(neighbors, set(range(len(graph))), 5)
        assert optimal and len(solution) == optimum and independent(neighbors, solution)
        pool = set(range(0, len(graph), 2))
        reduced, optimal = solve_reduced(neighbors, pool, 5)
        expected = max(
            len(s)
            for k in range(len(pool) + 1)
            for s in map(set, itertools.combinations(pool, k))
            if independent(neighbors, s)
        )
        assert optimal and reduced <= pool and len(reduced) == expected
        for variant in VARIANTS:
            for rate in (0, 1):
                age = [-1] * len(graph)
                sol = generate_solution(
                    neighbors,
                    age,
                    random.Random(7),
                    variant=variant,
                    determinism_rate=rate,
                )
                assert independent(neighbors, sol)
                assert all(v in sol or neighbors[v] & sol for v in range(len(graph)))
                assert all(age[v] == (0 if v in sol else -1) for v in range(len(graph)))
            result = cmsa(neighbors, Config(variant=variant, iterations=2, time_limit=5), seed=7)
            assert independent(neighbors, set(result["vertices"]))
            assert result["score"] <= optimum
            scores = [entry["score"] for entry in result["history"]]
            assert scores == sorted(scores)
        for variant in ("v1", "v2"):
            for seed in range(10):
                a, b = [-1] * len(graph), [-1] * len(graph)
                x = generate_solution(neighbors, a, random.Random(seed), variant=variant)
                y = generate_solution(neighbors, b, random.Random(seed), variant=variant + "-perf")
                assert x == y and a == b

    neighbors = graph_neighbors(nx.path_graph(4))
    # The archive uses degree, even when the youngest vertex has higher degree.
    assert generate_solution(
        neighbors, [20, -1, -1, 20], random.Random(0), variant="v1", determinism_rate=1
    ) == {0, 3}
    timeout = SimpleNamespace(status=1, x=None, message="time limit")
    with patch("cmsa.milp", return_value=timeout):
        assert solve_reduced(neighbors, {0, 1, 2, 3}, 1) == (None, False)
        result = cmsa(neighbors, Config(iterations=1), seed=1)
        assert result["score"] > 0 and independent(neighbors, set(result["vertices"]))
    incumbent = SimpleNamespace(status=1, x=[1.0, 0.0, 0.0, 1.0], message="time limit")
    with patch("cmsa.milp", return_value=incumbent):
        assert solve_reduced(neighbors, {0, 1, 2, 3}, 1) == ({0, 3}, False)
    assert generate_solution(neighbors, [-1] * 4, random.Random(0), deadline=0) == set()
    try:
        cmsa(neighbors, Config(iterations=1), constructor=lambda *a, **kw: {0, 1})
    except ValueError:
        pass
    else:
        raise AssertionError("infeasible replacement constructor accepted")
    for bad in (
        Config(age_max=0),
        Config(determinism_rate=float("nan")),
        Config(candidate_list_size=0),
        Config(time_limit=-1),
    ):
        try:
            cmsa(neighbors, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid configuration accepted")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "graph.txt"
        path.write_text("4\n0 1\n1 2\n")
        assert read_graph(path) == [{1}, {0, 2}, {1}, set()]
        path.write_text("c example\np edge 4 2\ne 1 2\ne 2 3\n")
        assert read_graph(path) == [{1}, {0, 2}, {1}, set()]
        for invalid in ("3\n0 3\n", "3\n1 1\n", "p edge 3 2\ne 1 2\n"):
            path.write_text(invalid)
            try:
                read_graph(path)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid graph accepted")
    print("CMSA checks passed: brute-force optima, five variants, ages, timeouts, graph input.")

    for family in ("er", "ba", "ws"):
        g = make_graph(family, 20, 0.2, 4)
        assert g == make_graph(family, 20, 0.2, 4)
        validate_graph_result = cmsa(g, Config(iterations=1, time_limit=1))
        assert independent(g, set(validate_graph_result["vertices"]))
    assert mean_ranks([[3, 3, 1], [1, 2, 3]]) == [2.25, 1.75, 2.0]
    source = baseline_source()
    assert "selection_probabilities" not in source and "entropy" not in source
    with TemporaryDirectory() as directory:
        root = Path(directory)
        template = (prompts.TEMPLATE_ROOT / "propose.j2").read_text()
        (root / "propose.j2").write_text(template + "\nLOADED_FROM_TEMPLATE")
        with patch.object(prompts, "TEMPLATE_ROOT", root):
            assert "LOADED_FROM_TEMPLATE" in asyncio.run(
                propose.render(source, "heuristic", "", [])
            )
    provider = DemoProvider()
    proposal = asyncio.run(propose(source, "heuristic", "", [], provider=provider))
    assert "def cmsa(" in provider.context and source in provider.context
    validate_candidate(proposal.code)
    assert proposal.code and proposal.rationale
    asyncio.run(
        propose(
            source,
            "performance",
            "Keep the random draws unchanged.",
            [{"proposal": proposal.model_dump()}],
            provider=provider,
        )
    )
    assert "Keep the random draws unchanged." in provider.context
    assert proposal.code in provider.context
    with TemporaryDirectory() as directory:
        args = SimpleNamespace(
            output=Path(directory),
            source=None,
            provider="demo",
            model=None,
            timeout=30,
            mode="heuristic",
            feedback="Improve this.",
        )
        asyncio.run(dialogue(args))
        args.feedback = "Preserve feasibility."
        asyncio.run(dialogue(args))
        assert (Path(directory) / "candidate-002.py").exists()
        assert "Preserve feasibility." in (Path(directory) / "prompt-002.txt").read_text()
    for code in (
        "def nope(): pass",
        "def generate_solution(:",
        'print("side effect")\ndef generate_solution(*a, **k): return set()',
    ):
        try:
            validate_candidate(code)
        except (ValueError, SyntaxError):
            pass
        else:
            raise AssertionError("invalid candidate accepted")
    print(
        "Slick checks passed: complete baseline context, typed proposal, feedback and syntax checks."
    )


if __name__ == "__main__":
    check()
