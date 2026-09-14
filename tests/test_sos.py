"""Check local vector neighborhoods and exhaustive objective transitions."""

import unittest
from pathlib import Path
from unittest.mock import patch

from sos import Evaluation, Individual, SoS
from tests.providers import ScriptedProvider


class SoSTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "sos/prompts")
        root.start()
        self.addCleanup(root.stop)

    def test_local_selection_is_not_weighted_ranking(self):
        agent = SoS("Task", ScriptedProvider([]), None, ["a", "b"])
        population = [
            Individual(str(i), Evaluation(scores, ("", "")))
            for i, scores in enumerate([(1, 0), (0, 1), (0.6, 0.6), (0.59, 0.59)])
        ]
        result = agent._local_optima(population, 0.05)
        self.assertEqual(result, population[:3])

    async def test_exhaustive_objectives_and_crossover(self):
        async def evaluate(text):
            return Evaluation((1, 1), ("a error", "b error"))

        provider = ScriptedProvider(
            ['{"text":"feedback"}', '{"text":"seed"}', '{"text":"feedback"}', '{"text":"seed"}']
        )
        result = await SoS("Task", provider, evaluate, ["first", "second"]).run(
            "seed", weights=(0.5, 0.5), variants=0, max_rounds=3
        )
        self.assertEqual(result["objectives"], [0, 1])
        self.assertEqual(result["evaluations"], 3)
        self.assertIn("a error", provider.calls[0])
        self.assertIn("b error", provider.calls[2])
        agent = SoS("Task", provider, evaluate, ["first", "second"])
        for method, args in ((SoS.semantic, ("seed",)), (SoS.crossover, ("a", "b"))):
            self.assertIn("Task", await method.render(agent, *args))

    async def test_nonfinite_and_generated_failure(self):
        async def evaluate(text):
            return Evaluation((float("nan"), 0), ("", ""))

        agent = SoS("Task", ScriptedProvider([]), evaluate, ["a", "b"])
        with self.assertRaises(ValueError):
            await agent.run("seed", weights=(0.5, 0.5), variants=0)
        with self.assertRaises(ValueError):
            await agent.semantic("seed", provider=ScriptedProvider(['{"text":" "}']))
