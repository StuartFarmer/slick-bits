"""Offline checks of CCMO selection, reference GA arithmetic, and Slick offspring."""

import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from ccmo_llm import CCMOLLM, CandidateRejected, Evaluation, Individual
from ccmo_llm.agent import environmental_selection, fitness
from tests.providers import ScriptedProvider


async def evaluate(x):
    return Evaluation((sum(v * v for v in x), sum((v - 1) ** 2 for v in x)), (0.5 - x[0],))


class CCMOTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "ccmo_llm/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)

    def agent(self, provider=None, evaluator=evaluate):
        return CCMOLLM("Minimize competing costs", [(0, 1)] * 2, 2, evaluator, provider)

    def test_reference_fitness_constraints_and_truncation(self):
        points = [Individual((i,), f, 0) for i, f in enumerate([(0, 0), (1, 1), (0, 2), (2, 0)])]
        expected = [0.25, 3 + 1 / (math.sqrt(2) + 2), 3.25, 3.25]
        for actual, target in zip(fitness(points, True), expected):
            self.assertAlmostEqual(actual, target)
        points = [Individual((0,), (0, 0), 1), Individual((1,), (5, 5), 0)]
        self.assertEqual(environmental_selection(points, 1, True)[0], [points[1]])
        self.assertEqual(environmental_selection(points, 1, False)[0], [points[0]])
        points = [Individual((i,), (i, 4 - i), 0) for i in (0, 1, 2, 4)]
        selected, scores = environmental_selection(points, 2, True)
        self.assertEqual(set(selected), {points[0], points[-1]})
        # Mating fitness is retained from the union, as in EnvironmentalSelection.m.
        self.assertEqual(scores, [fitness(points, True)[points.index(p)] for p in selected])

    async def test_hybrid_split_shared_offspring_and_exact_budget(self):
        provider = ScriptedProvider(["<start>0.9,0.1<end>", "<start>0.8,0.2<end>"])
        agent = self.agent(provider)
        result = await agent.run(population_size=20, max_evaluations=63, seed=4)
        self.assertEqual((result.evaluations, result.model_calls), (63, 2))
        self.assertEqual((len(result.population), len(result.auxiliary)), (20, 20))
        self.assertEqual(sum(r["source"] == "llm" for r in agent.measurements), 2)
        self.assertEqual(sum(r["source"] == "ga" for r in agent.measurements), 21)
        self.assertEqual([len(r["samples"]) for r in agent.attempts], [2, 2])
        self.assertTrue(all(p.cv == 0 for p in result.front))
        self.assertEqual(len(agent.generations[0]["offspring"]), 20)
        initial = [r["individual"] for r in agent.measurements[:40]]
        offspring = list(agent.generations[0]["offspring"])
        self.assertEqual(
            agent.generations[0]["population"],
            tuple(environmental_selection(initial[:20] + offspring, 20, True)[0]),
        )
        self.assertEqual(
            agent.generations[0]["auxiliary"],
            tuple(environmental_selection(initial[20:] + offspring, 20, False)[0]),
        )
        self.assertIn("constraint violation degree", provider.calls[1])

    async def test_invalid_batches_retry_without_evaluating_or_silent_fallback(self):
        invalid = [
            "<start>nan,0<end>",
            "<start>0.4<end>",
            "<start>2,0<end>",
            "<start>0.1,0.2<end><start>0.2,0.3<end>",
        ]
        agent = self.agent(
            ScriptedProvider([*invalid, "<start>0.9,0.1<end>", "<start>0.8,0.2<end>"])
        )
        result = await agent.run(population_size=20, max_evaluations=60, max_attempts=5, seed=5)
        self.assertEqual((result.evaluations, result.model_calls), (60, 6))
        self.assertEqual([r["response"] for r in agent.attempts[:4]], invalid)
        self.assertTrue(all(r["status"] == "rejected" for r in agent.attempts[:4]))
        agent = self.agent(ScriptedProvider(["bad"] * 2))
        with self.assertRaisesRegex(RuntimeError, "2 attempts"):
            await agent.run(population_size=20, max_evaluations=60, max_attempts=2)
        self.assertEqual(len(agent.attempts), 2)
        self.assertEqual(agent.evaluations, 40)

    async def test_duplicates_wrong_count_and_raw_failure_records(self):
        agent = self.agent(ScriptedProvider(["<start>0.1,0.2<end>" * 2]))
        agent.rng = random.Random(0)
        samples = [Individual((0, 0), (1, 2), 0), Individual((1, 1), (2, 1), 1)]
        with self.assertRaises(RuntimeError):
            await agent._llm_offspring(samples, 2, 0, set(), 1)
        self.assertIn("distinct", agent.attempts[0]["error"])
        agent.provider = ScriptedProvider([TimeoutError("offline")])
        with self.assertRaisesRegex(TimeoutError, "offline"):
            await agent.run(population_size=20, max_evaluations=60)
        self.assertEqual(agent.attempts[0]["status"], "error")

    async def test_measurements_equality_tolerance_and_error_boundaries(self):
        async def equality(x):
            return Evaluation((1, 2), (-1, 0.3), (-0.2, 0.05))

        agent = self.agent(evaluator=equality)
        result = await agent.run(population_size=2, max_evaluations=4, equality_tolerance=0.1)
        self.assertAlmostEqual(result.population[0].cv, 0.4)
        self.assertEqual(result.front, ())
        calls = 0

        async def sometimes_bad(x):
            nonlocal calls
            calls += 1
            if calls == 5:
                return Evaluation((float("nan"), 0))
            return await evaluate(x)

        agent = self.agent(evaluator=sometimes_bad)
        result = await agent.run(population_size=2, max_evaluations=7, llm_fraction=0)
        self.assertEqual(result.evaluations, 7)
        self.assertEqual(agent.measurements[4]["status"], "rejected")

        async def broken(x):
            raise LookupError("evaluator bug")

        agent = self.agent(evaluator=broken)
        with self.assertRaisesRegex(LookupError, "evaluator bug"):
            await agent.run(population_size=2, max_evaluations=4)
        self.assertEqual(agent.measurements[0]["status"], "error")

        async def rejected(x):
            raise CandidateRejected("unmeasurable")

        agent = self.agent(evaluator=rejected)
        with self.assertRaisesRegex(RuntimeError, "initial"):
            await agent.run(population_size=2, max_evaluations=4)

    async def test_baseline_seed_bounds_and_short_initialization_budget(self):
        agent = CCMOLLM("Fixed coordinate", [(4, 4), (-2, 3)], 2, evaluate)
        kwargs = dict(population_size=6, max_evaluations=31, llm_fraction=0, seed=8)
        result = await agent.run(**kwargs)
        self.assertEqual(result, await agent.run(**kwargs))
        self.assertTrue(all(r["x"][0] == 4 and -2 <= r["x"][1] <= 3 for r in agent.measurements))
        self.assertEqual(result.model_calls, 0)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            await agent.run(population_size=6, max_evaluations=3, llm_fraction=0)
        self.assertEqual(agent.evaluations, 3)

    def test_reference_ga_mutation_arithmetic(self):
        agent = self.agent()
        agent.rng = random.Random(0)
        # Direct numerical values from OperatorGAhalf's real-valued mutation formula.
        with patch.object(agent.rng, "random", side_effect=[0, 0.25, 1]):
            self.assertAlmostEqual(agent._mutate((0.5, 0.5))[0], 0.46753180049317733)
        with patch.object(agent.rng, "random", side_effect=[0, 0.75, 1]):
            self.assertAlmostEqual(agent._mutate((0.5, 0.5))[0], 0.5324681995068227)
        with patch.object(agent.rng, "random", side_effect=[0, 0, 0, 1]):
            self.assertEqual(agent._mutate((0.5, 0.5)), (0, 1))

    def test_reference_sbx_first_child(self):
        agent = self.agent()
        agent.rng = random.Random(0)
        parents = [Individual((0, 0), (0, 0), 0), Individual((1, 1), (1, 1), 0)]
        with patch.object(agent.rng, "randrange", side_effect=[0, 0, 1, 1, 0, 0]):
            with patch.object(agent.rng, "random", side_effect=[0.25, 0.9, 0.25, 0.9, 1, 1]):
                child = agent._ga_offspring(parents, [0, 1], 1)[0]
        expected = (1 - 0.5 ** (1 / 21)) / 2
        self.assertAlmostEqual(child[0], expected)
        self.assertEqual(child[0], child[1])

    async def test_paper_sized_batches_and_three_objectives(self):
        async def three(x):
            return Evaluation((x[0], x[1], -sum(x)), (0.5 - sum(x),))

        replies = [
            "".join(f"<start>{i / 100},{(20 - i) / 100}<end>" for i in range(k, k + 5))
            for k in (1, 6)
        ]
        agent = CCMOLLM("Three objectives", [(0, 1)] * 2, 3, three, ScriptedProvider(replies))
        result = await agent.run(population_size=100, max_evaluations=300, seed=9)
        self.assertEqual((result.evaluations, result.model_calls), (300, 2))
        self.assertEqual([len(r["samples"]) for r in agent.attempts], [10, 10])
        self.assertEqual(sum(r["source"] == "ga" for r in agent.measurements), 90)
        self.assertEqual(sum(r["source"] == "llm" for r in agent.measurements), 10)

    async def test_template_binding_and_external_working_directory(self):
        agent = self.agent()
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                rendered = await CCMOLLM.propose.render(
                    agent, "no feasible solution", "no infeasible solution", 3
                )
            finally:
                os.chdir(previous)
        self.assertIn(agent.task, rendered)
        self.assertIn("3 new solutions", rendered)
        self.assertIn("<start>", rendered)
        tree = Environment().parse((self.root / "propose.j2").read_text())
        self.assertEqual(list(tree.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
