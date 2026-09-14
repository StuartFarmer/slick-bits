import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ordered_prompts import Example, OrderedPrompts
from tests.providers import ScriptedProvider


class OrderedPromptsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "ordered_prompts/prompts"
        )
        root.start()
        self.addCleanup(root.stop)

    async def test_global_entropy_selects_balanced_order_without_gold_selection(self):
        calls = []

        async def classify(demos, inputs):
            calls.append(inputs)
            return np.array([[0.9, 0.1], [0.9, 0.1]]) if demos[0].input == "A" else np.eye(2)

        async def evaluate(demos):
            self.assertEqual(demos[0].input, "B")
            return -3.0

        provider = ScriptedProvider(['{"inputs":["x","y"]}', '{"inputs":["x"]}', "label"])
        agent = OrderedPrompts("task", provider, evaluate, classify)
        result = await agent.run([Example("A", "0"), Example("B", "1")], classes=2)
        self.assertEqual(result["selected"][0]["order"], (1, 0))
        self.assertEqual(result["synthetic"], ["x", "y"])
        self.assertAlmostEqual(result["selected"][0]["entropy"], np.log(2))
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["classifications"], 2)
        await agent.answer("q", provider=provider)
        self.assertLess(provider.calls[-1].index("Input: B"), provider.calls[-1].index("Input: A"))

    async def test_bad_synthetic_json_fails_before_classification(self):
        agent = OrderedPrompts("task", ScriptedProvider(['{"inputs":[" "]}']), None, None)
        with self.assertRaises(ValueError):
            await agent.run([Example("a", "0")], classes=2)
        self.assertEqual(agent.classifications, 0)
