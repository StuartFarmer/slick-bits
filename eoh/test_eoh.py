"""Offline checks: python eoh/test_eoh.py (requires Slick and NumPy)."""

import asyncio
import itertools
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from slick import prompts

import evolve
import problems
from worker import evaluate_candidate


def setUpModule():
    previous_root = prompts.TEMPLATE_ROOT
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    unittest.addModuleCleanup(setattr, prompts, "TEMPLATE_ROOT", previous_root)


class EoHChecks(unittest.TestCase):
    def test_guidance_escapes_actual_local_optima(self):
        xy = np.random.default_rng(14).random((10, 2))
        distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)

        def identity(matrix, tour, used):
            return matrix

        _, unguided = problems.solve_tsp(distances, identity, iterations=20, seconds=5)
        route, guided = problems.solve_tsp(
            distances, problems.tsp_penalty, iterations=20, seconds=5
        )
        self.assertAlmostEqual(unguided, 2.548035844842225)
        self.assertLess(guided, unguided - 0.02)
        self.assertEqual(sorted(route), list(range(10)))

        times = np.random.default_rng(8).integers(1, 20, (8, 4)).astype(float)

        def no_perturbation(sequence, matrix, m, n):
            return matrix, np.arange(n)

        _, unguided = problems.solve_flowshop(times, no_perturbation, iterations=20, seconds=5)
        np.random.seed(0)
        sequence, guided = problems.solve_flowshop(
            times, problems.flow_perturb, iterations=20, seconds=5
        )
        self.assertEqual(unguided, 110)
        self.assertEqual(guided, 109)
        self.assertEqual(sorted(sequence), list(range(8)))

    def test_local_search_improves_and_respects_selected_jobs(self):
        times = np.array([[9.0, 6.0, 7.0], [9.0, 6.0, 7.0], [8.0, 3.0, 1.0], [3.0, 3.0, 8.0]])
        sequence = [0, 1, 2, 3]
        restricted, restricted_cost = problems.local_search(
            times,
            sequence,
            "flowshop",
            time.perf_counter() + 2,
            jobs=[0],
            one_move=True,
        )
        improved, improved_cost = problems.local_search(
            times, sequence, "flowshop", time.perf_counter() + 2, one_move=True
        )
        self.assertEqual(restricted, sequence)
        self.assertEqual(restricted_cost, 40)
        self.assertEqual(improved_cost, 35)
        self.assertEqual(sorted(improved), sequence)

    def test_evolution_budget_elitism_and_slick_prompts(self):
        class Provider:
            def __init__(self):
                self.prompts = []

            async def acall(self, context):
                self.prompts.append(context)
                return json.dumps(
                    {
                        "thought": "Prefer tight bins.",
                        "code": "def score(item, bins):\n    return item - bins",
                    }
                ), []

        async def run():
            provider = Provider()

            async def evaluate(code):
                return {"fitness": float(len(provider.prompts)), "values": [1]}

            with tempfile.TemporaryDirectory() as directory:
                result = await evolve.evolve(
                    provider,
                    "binpacking",
                    evaluate,
                    population_size=2,
                    generations=2,
                    parents=2,
                    seed=7,
                    output=Path(directory),
                )
                self.assertEqual(len(provider.prompts), 22)
                self.assertEqual([h.fitness for h in result], [22, 21])
                records = [
                    json.loads(line)
                    for line in (Path(directory) / "attempts.jsonl").read_text().splitlines()
                ]
                for generation in (1, 2):
                    for operator in evolve.OPERATORS:
                        rows = [
                            r
                            for r in records
                            if r["generation"] == generation and r["operator"] == operator
                        ]
                        self.assertEqual(len(rows), 2)
                        for row in rows:
                            self.assertEqual(len(row["parents"]), 2 if operator[0] == "E" else 1)
                            self.assertTrue(
                                all(p < (3 if generation == 1 else 13) for p in row["parents"])
                            )
                self.assertIn("Prefer tight bins.", provider.prompts[2])
                self.assertIn("return item - bins", provider.prompts[2])
                self.assertTrue((Path(directory) / "best.py").exists())

        asyncio.run(run())

    def test_packing_and_lower_bound(self):
        items = [6, 4, 6, 4, 10]
        self.assertEqual(problems.pack(items, 10, problems.best_fit), 3)
        self.assertEqual(problems.bin_lower_bound([6, 6, 6], 10), 3)
        self.assertEqual(problems.bin_lower_bound(items, 10), 3)

        def invalid(item, bins):
            return np.full(len(bins), np.nan)

        with self.assertRaises(ValueError):
            problems.pack(items, 10, invalid)

        # Mutating the supplied capacities must not corrupt packing state.
        def mutating(item, bins):
            bins[:] = 0
            return bins

        self.assertEqual(problems.pack(items, 10, mutating), 3)

    def test_local_search_feasibility_and_original_objective(self):
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)

        def update(matrix, tour, used):
            matrix[:] = 0
            return matrix

        route, value = problems.solve_tsp(distances, update, iterations=2, seconds=2)
        self.assertEqual(sorted(route), list(range(4)))
        self.assertAlmostEqual(value, 4)
        self.assertAlmostEqual(problems.tour_length(distances, route), value)
        times = np.array([[2.0, 1.0], [1.0, 3.0], [3.0, 2.0]])
        optimum = min(problems.makespan(times, p) for p in itertools.permutations(range(3)))

        def flow_update(sequence, matrix, m, n):
            return matrix * 0, np.arange(n)

        sequence, value = problems.solve_flowshop(times, flow_update, iterations=2, seconds=2)
        self.assertEqual(sorted(sequence), list(range(3)))
        self.assertEqual(value, optimum)
        self.assertEqual(value, problems.makespan(times, sequence))

    def test_worker_rejects_bad_output_and_stops_infinite_code(self):
        dataset = {
            "problem": "binpacking",
            "instances": [{"items": [6, 4, 6, 4], "capacity": 10}],
        }

        async def run():
            result = await evaluate_candidate(
                "def score(item, bins):\n    return item-bins", dataset, timeout=5
            )
            self.assertEqual(result["fitness"], 1)
            for code in (
                "def score(item, bins):\n    return [float('nan')]*len(bins)",
                "def wrong(item, bins):\n    return bins",
                "def score(item, bins):\n    while True: pass",
            ):
                result = await evaluate_candidate(code, dataset, timeout=0.4)
                self.assertIn("error", result)
                self.assertNotIn("fitness", result)

        asyncio.run(run())

    def test_invalid_initialization_is_bounded(self):
        class BadProvider:
            calls = 0

            async def acall(self, context):
                self.calls += 1
                return "not JSON", []

        async def run():
            provider = BadProvider()

            async def evaluate(code):
                self.fail("Malformed proposals must not be evaluated")

            with self.assertRaisesRegex(RuntimeError, "initial"):
                await evolve.evolve(
                    provider,
                    "binpacking",
                    evaluate,
                    population_size=2,
                    parents=2,
                    init_attempts=3,
                )
            self.assertEqual(provider.calls, 3)

        asyncio.run(run())

    def test_invalid_offspring_keep_incumbents_and_consume_budget(self):
        async def run():
            calls = 0

            async def evaluate(code):
                nonlocal calls
                calls += 1
                return {"fitness": 0.75} if calls <= 2 else {"error": "invalid scores"}

            provider = evolve.DemoProvider("binpacking")
            result = await evolve.evolve(
                provider,
                "binpacking",
                evaluate,
                population_size=2,
                parents=2,
                generations=1,
            )
            self.assertEqual(provider.calls, 12)
            self.assertEqual([h.id for h in result], [1, 2])

        asyncio.run(run())

    def test_nonfinite_fitness_is_logged_as_failure(self):
        async def run():
            calls = 0

            async def evaluate(code):
                nonlocal calls
                calls += 1
                return {"fitness": 1.0 if calls == 1 else float("nan")}

            with tempfile.TemporaryDirectory() as directory:
                result = await evolve.evolve(
                    evolve.DemoProvider("binpacking"),
                    "binpacking",
                    evaluate,
                    population_size=1,
                    parents=1,
                    generations=1,
                    output=Path(directory),
                )
                self.assertEqual(result[0].id, 1)
                rows = [
                    json.loads(line)
                    for line in (Path(directory) / "attempts.jsonl").read_text().splitlines()
                ]
                self.assertTrue(all("error" in row["evaluation"] for row in rows[1:]))

        asyncio.run(run())

    def test_lower_bound_never_exceeds_exact_small_packing(self):
        def exact(items, capacity):
            states = {()}
            for item in sorted(items, reverse=True):
                next_states = set()
                for loads in states:
                    next_states.add(tuple(sorted((*loads, item))))
                    for i, load in enumerate(loads):
                        if load + item <= capacity:
                            updated = list(loads)
                            updated[i] += item
                            next_states.add(tuple(sorted(updated)))
                states = next_states
            return min(map(len, states))

        for items in itertools.combinations_with_replacement(range(1, 7), 5):
            self.assertLessEqual(problems.bin_lower_bound(items, 6), exact(items, 6))

    def test_problem_metrics_and_bad_data(self):
        square = [
            [0, 1, 2**0.5, 1],
            [1, 0, 1, 2**0.5],
            [2**0.5, 1, 0, 1],
            [1, 2**0.5, 1, 0],
        ]
        result = problems.evaluate(
            {"problem": "tsp", "instances": [{"distances": square, "optimum": 4}]},
            problems.tsp_penalty,
            iterations=1,
        )
        self.assertEqual(result["fitness"], 0)
        self.assertEqual(result["metric"], "negative_mean_gap_percent")
        times = [[2, 1], [1, 3], [3, 2]]
        result = problems.evaluate(
            {"problem": "flowshop", "instances": [{"times": times}]},
            problems.flow_perturb,
            iterations=1,
        )
        self.assertEqual(result["fitness"], -result["values"][0])
        for instance in (
            {"items": [0], "capacity": 10},
            {"items": [11], "capacity": 10},
            {"items": [1.5], "capacity": 10},
        ):
            with self.assertRaises(ValueError):
                problems.validate_dataset({"problem": "binpacking", "instances": [instance]})
        with self.assertRaises(ValueError):
            problems.flow_output((np.ones((3, 2)), [0, 0]), (3, 2))
        with self.assertRaises(ValueError):
            problems.flow_output((np.ones((3, 2)), [0.5]), (3, 2))

    def test_prompts_keep_five_strategies_distinct_and_code_only_hides_thoughts(self):
        async def run():
            parent = evolve.Heuristic(
                thought="UNIQUE_SECRET_THOUGHT",
                code="def score(item, bins): pass",
                id=1,
                fitness=1,
            )
            contexts = [
                await evolve.propose.render(
                    problems.TASKS["binpacking"],
                    evolve.INSTRUCTIONS[operator],
                    [parent],
                )
                for operator in evolve.OPERATORS
            ]
            self.assertEqual(len(set(contexts)), 5)
            self.assertTrue(all("UNIQUE_SECRET_THOUGHT" in context for context in contexts))
            hidden = await evolve.propose.render(
                problems.TASKS["binpacking"], evolve.INSTRUCTIONS["E1"], [parent], False
            )
            self.assertNotIn("UNIQUE_SECRET_THOUGHT", hidden)
            self.assertIn(parent.code, hidden)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                template = (prompts.TEMPLATE_ROOT / "propose.j2").read_text()
                (root / "propose.j2").write_text(template + "\nLOADED_FROM_TEMPLATE")
                with patch.object(prompts, "TEMPLATE_ROOT", root):
                    loaded = await evolve.propose.render("task", "instruction", [])
                self.assertIn("LOADED_FROM_TEMPLATE", loaded)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
