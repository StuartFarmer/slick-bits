"""Offline checks of MOEA/D search, official LO arithmetic, and Slick generation."""

import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from llm4moea import MOEAD, CandidateRejected, Individual, das_dennis, rank_weights
from tests.providers import ScriptedProvider


async def objectives(x):
    return (sum(v * v for v in x), sum((v - 1) ** 2 for v in x))


class MOEADTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "llm4moea/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)

    def agent(self, provider=None, evaluate=objectives):
        return MOEAD("Minimize two competing costs", [(0, 1)] * 2, 2, evaluate, provider)

    def test_reference_directions_and_official_weights(self):
        directions = das_dennis(3, 2)
        self.assertEqual(len(directions), 6)
        self.assertIn((0.5, 0.5, 0.0), directions)
        self.assertTrue(all(sum(w) == 1 for w in directions))
        # Worst-to-best parents have normalized ranks 1 and 1/2.
        polynomial = (0.080, 0.044875)
        expected = [v / sum(polynomial) for v in polynomial]
        for actual, value in zip(rank_weights(2, "official"), expected):
            self.assertAlmostEqual(actual, value)
        softmax = [math.exp(v) / sum(map(math.exp, polynomial)) for v in polynomial]
        for actual, value in zip(rank_weights(2, "paper"), softmax):
            self.assertAlmostEqual(actual, value)
        self.assertGreater(rank_weights(10)[-1], rank_weights(10)[0])

    async def test_lo_budget_repeatability_bounds_and_archive(self):
        agent = self.agent()
        settings = dict(
            max_evaluations=53, population_size=10, parents=4, neighborhood_size=5, seed=3
        )
        result = await agent.run(**settings)
        self.assertEqual(result.evaluations, 53)
        self.assertEqual(len(agent.measurements), 53)
        self.assertEqual(len(result.population), 10)
        self.assertEqual(result.model_calls, 0)
        self.assertTrue(all(0 <= v <= 1 for p in result.archive for v in p.x))
        measured = [r["individual"] for r in agent.measurements]
        expected = {
            p
            for p in measured
            if not any(
                all(a <= b for a, b in zip(q.objectives, p.objectives))
                and q.objectives != p.objectives
                for q in measured
            )
        }
        self.assertEqual(set(result.archive), expected)
        self.assertEqual(
            result.ideal, tuple(min(p.objectives[j] for p in measured) for j in range(2))
        )
        self.assertEqual(result, await agent.run(**settings))

    async def test_llm_retries_partial_batches_scaling_and_exact_budget(self):
        provider = ScriptedProvider(
            [
                "bad <start>nan, 0.2<end><start>0.5<end>",
                "<start>0.25,\n0.75<end><start>1e1,-1<end>",
                "<start>0.4,0.6<end>",
            ]
        )
        agent = MOEAD("Any real-valued task", [(-2, 2), (10, 20)], 2, objectives, provider)
        result = await agent.run(
            operator="llm",
            max_evaluations=7,
            population_size=4,
            parents=2,
            neighborhood_size=3,
            mutation_probability=0,
            seed=1,
        )
        self.assertEqual((result.evaluations, result.model_calls), (7, 3))
        self.assertEqual(agent.measurements[4]["individual"].x, (-1, 17.5))
        self.assertEqual(agent.measurements[5]["individual"].x, (2, 10))
        self.assertEqual(agent.attempts[0]["status"], "rejected")
        self.assertIn("nan", agent.attempts[0]["response"])
        self.assertEqual(provider.calls[0], provider.calls[1])
        self.assertIn("function values", provider.calls[0])
        for attempt in agent.attempts:
            values = [value for _, value in attempt["samples"]]
            self.assertEqual(values, sorted(values, reverse=True))

    async def test_bad_generation_is_bounded_and_provider_errors_propagate(self):
        agent = self.agent(ScriptedProvider(["no points"] * 2))
        with self.assertRaisesRegex(RuntimeError, "2 attempts"):
            await agent.run(
                operator="llm",
                max_evaluations=5,
                population_size=4,
                parents=2,
                neighborhood_size=3,
                max_attempts=2,
            )
        self.assertEqual(agent.evaluations, 4)
        self.assertEqual(len(agent.attempts), 2)
        agent.provider = ScriptedProvider([TimeoutError("provider offline")])
        with self.assertRaisesRegex(TimeoutError, "provider offline"):
            await agent.run(
                operator="llm", max_evaluations=5, population_size=4, parents=2, neighborhood_size=3
            )
        self.assertEqual(agent.attempts[0]["status"], "error")

    async def test_evaluation_failures_consume_budget_without_poisoning_search(self):
        calls = 0

        async def evaluate(x):
            nonlocal calls
            calls += 1
            return (float("nan"), 0) if calls == 5 else await objectives(x)

        agent = self.agent(evaluate=evaluate)
        result = await agent.run(
            max_evaluations=6, population_size=4, parents=2, neighborhood_size=3, seed=7
        )
        self.assertEqual(result.evaluations, 6)
        self.assertEqual(agent.measurements[4]["status"], "rejected")
        self.assertTrue(all(math.isfinite(v) for v in result.ideal))

    async def test_replacement_uses_new_ideal_and_replaces_at_most_two(self):
        agent = self.agent()
        await agent.run(
            max_evaluations=4, population_size=4, parents=2, neighborhood_size=3, seed=1
        )
        old = Individual((1, 1), (5, 5))
        agent.population = [old] * 4
        agent.ideal = (5, 5)
        child = Individual((0, 0), (1, 1))
        agent._update(child, [3, 2, 1, 0], strict=False)
        self.assertEqual(agent.ideal, (1, 1))
        self.assertEqual(agent.population, [old, old, child, child])

    async def test_fixed_coordinates_and_template_from_another_directory(self):
        agent = MOEAD("Any task", [(4, 4), (-2, 3)], 2, objectives)
        result = await agent.run(
            max_evaluations=15, population_size=4, parents=2, neighborhood_size=3, seed=4
        )
        self.assertTrue(all(p.x[0] == 4 for p in result.archive))
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                rendered = await MOEAD.propose.render(agent, [((0, 0.5), 3.0)], 2)
            finally:
                os.chdir(previous)
        self.assertIn("Any task", rendered)
        self.assertIn("<start>0,0.5<end>", rendered)
        tree = Environment().parse((self.root / "propose.j2").read_text())
        self.assertEqual(list(tree.find_all((nodes.If, nodes.CondExpr))), [])

    def test_linear_noise_is_shared_across_dimensions_and_not_renormalized(self):
        agent = self.agent()
        agent.rng = random.Random(11)
        selected = [Individual((1, 2), (9, 9)), Individual((3, 6), (1, 1))]
        oracle = random.Random(11)
        expected = sum(
            (w + 0.5 * oracle.gauss(0, 1)) * p.x[0] for w, p in zip(rank_weights(2), selected)
        )
        actual = agent._linear(selected, selected[0], 0.5, 1.0, "official")
        self.assertAlmostEqual(actual[0], expected)
        self.assertAlmostEqual(actual[1], 2 * expected)
        self.assertEqual(agent._linear(selected, selected[0], 0.5, 0, "official"), (1, 2))

    def test_polynomial_mutation_has_known_boundary_behavior(self):
        agent = self.agent()
        agent.rng = random.Random(0)
        # Gate, first site, u=0, second site, u=1: move to each bound.
        with patch.object(agent.rng, "random", side_effect=[0, 0, 0, 0, 1]):
            self.assertEqual(agent._mutate((0.25, 0.75), 1, 20), (0, 1))
        # Standard PM from official pymoo; MATLAB LO has different parentheses.
        with patch.object(agent.rng, "random", side_effect=[0, 0, 0.25, 1]):
            actual = agent._mutate((0.5, 0.5), 1, 20)
        self.assertAlmostEqual(actual[0], 0.46753180049317733)
        self.assertEqual(actual[1], 0.5)
        self.assertEqual(agent._mutate((-3, 4), 0, 20), (0, 1))

    async def test_three_objectives_custom_weights_and_failure_boundaries(self):
        async def three(x):
            return (x[0] ** 2, (x[0] - 1) ** 2, (x[0] + 1) ** 2)

        agent = MOEAD("Three costs", [(-2, 2)], 3, three)
        result = await agent.run(
            population_size=8, parents=2, neighborhood_size=3, max_evaluations=15, seed=10
        )
        self.assertEqual(len(result.population), 6)  # Complete lattice, not arbitrary truncation.
        self.assertEqual(len(result.ideal), 3)
        weights = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        result = await agent.run(
            weights=weights, max_evaluations=7, parents=2, neighborhood_size=3, seed=10
        )
        self.assertEqual(agent.weights, weights)
        self.assertEqual(len(result.population), 3)

        async def rejected(x):
            raise CandidateRejected("infeasible")

        agent.evaluate = rejected
        with self.assertRaisesRegex(RuntimeError, "Initial population"):
            await agent.run(population_size=3, max_evaluations=3)
        self.assertEqual(agent.evaluations, 1)
        self.assertIn("infeasible", agent.measurements[0]["error"])

        async def broken(x):
            raise LookupError("infrastructure error")

        agent.evaluate = broken
        with self.assertRaisesRegex(LookupError, "infrastructure error"):
            await agent.run(population_size=3, max_evaluations=3)
        self.assertEqual(agent.measurements[0]["status"], "error")


if __name__ == "__main__":
    unittest.main()
