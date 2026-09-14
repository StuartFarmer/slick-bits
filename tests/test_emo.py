"""Check EMO genotype-to-phenotype evaluation and both survivor strategies."""

import unittest
from pathlib import Path
from unittest.mock import patch

from emo import EMO, Individual
from emo.agent import GeneratedText, hypervolume_2d
from tests.providers import ScriptedProvider


class EMOTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "emo/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_crossover_mutation_phenotype_and_vector_fitness(self):
        evaluated = []

        async def evaluate(text):
            evaluated.append(text)
            return {"text a": (1, 0), "text b": (0, 1), "text c": (1, 1)}[text]

        provider = ScriptedProvider(
            [GeneratedText(text=x) for x in ("text a", "text b", "cross", "mutant", "text c")]
        )
        result = await EMO("Task", provider, evaluate, ["a", "b"]).run(
            ["prompt a", "prompt b"], generations=1, offspring_size=1
        )
        self.assertEqual(evaluated, ["text a", "text b", "text c"])
        self.assertEqual(result["pareto"][0].prompt, "mutant")
        self.assertIn("cross", provider.calls[3])
        self.assertIn("mutant", provider.calls[4])
        agent = EMO("Task", provider, evaluate, ["a", "b"])
        for method in (EMO.change, EMO.modify, EMO.paraphrase):
            self.assertIn("Task", await method.render(agent, "instruction"))

    def test_hypervolume_and_sms_deletion(self):
        self.assertAlmostEqual(hypervolume_2d([(1, 0.2), (0.2, 1), (0.6, 0.6)], (0, 0)), 0.52)
        agent = EMO("Task", ScriptedProvider([]), None, ["a", "b"])
        population = [
            Individual(str(i), "text", p)
            for i, p in enumerate([(1, 0.2), (0.2, 1), (0.6, 0.6), (0.1, 0.1)])
        ]
        selected = agent._select_sms(population, 3, (0, 0))
        self.assertNotIn(population[-1], selected)
        selected = agent._select_sms(population[:3], 2, (0, 0))
        self.assertIn(population[2], selected)

    async def test_generation_rejection(self):
        agent = EMO("Task", ScriptedProvider([]), None, ["a", "b"])
        with self.assertRaises(ValueError):
            await agent.generate("prompt", provider=ScriptedProvider(['{"text":" "}']))
