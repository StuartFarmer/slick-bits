import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from capo import CAPO
from capo.agent import significantly_better
from tests.providers import ScriptedProvider


class CAPOTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "capo/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_cost_adjusted_race_eliminates_before_later_blocks(self):
        calls = []

        async def evaluate(text, block):
            calls.append((text, block))
            return (
                np.array([0.8, 0.81, 0.79])
                if text == "a"
                else np.array([0.7, 0.71, 0.69])
                if text == "b"
                else np.array([0.9, 0.91, 0.89])
            )

        async def demonstrate(instruction, example):
            self.fail("no demonstrations requested")

        provider = ScriptedProvider(["<prompt>cross</prompt>", "<prompt>very long child</prompt>"])
        result = await CAPO("Task", provider, evaluate, demonstrate).run(
            ["a", "b"],
            ["first", "later"],
            iterations=1,
            crossovers=1,
            length_penalty=0.5,
            shuffle_blocks=False,
        )
        self.assertEqual([item.instruction for item in result["population"]], ["a", "b"])
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(result["example_evaluations"], 9)
        self.assertTrue(all(block == "first" for _, block in calls))
        self.assertEqual(result["history"][0]["beaten"][-1], 2)

    async def test_demonstration_reasoning_only_when_answer_correct(self):
        async def evaluate(text, block):
            return np.ones(2)

        async def demonstrate(instruction, example):
            return (
                ("yes", "correct reasoning")
                if example["input"] == "good"
                else ("no", "wrong reasoning")
            )

        agent = CAPO("Task", ScriptedProvider([]), evaluate, demonstrate)
        import random

        agent.rng = random.Random(0)
        agent.demonstrations = [
            {"input": "good", "target": "yes"},
            {"input": "bad", "target": "yes"},
        ]
        agent.demonstration_calls = 0
        examples = await agent._examples("instruction", 2)
        self.assertTrue(any("correct reasoning" in text for text in examples))
        self.assertFalse(any("wrong reasoning" in text for text in examples))
        self.assertTrue(significantly_better([1, 1, 1], [0, 0, 0], 0.05))
        self.assertFalse(significantly_better([1], [0], 0.05))

    async def test_cached_block_scores_snapshot_reused_evaluator_buffer(self):
        buffer = np.zeros(3)

        async def evaluate(text, block):
            buffer[:] = {"a": 0.8, "b": 0.6, "child": 0.1}[text]
            return buffer

        async def demonstrate(instruction, example):
            self.fail("no demonstrations")

        provider = ScriptedProvider(["<prompt>cross</prompt>", "<prompt>child</prompt>"])
        result = await CAPO("Task", provider, evaluate, demonstrate).run(
            ["a", "b"], ["block"], iterations=1, crossovers=1, length_penalty=0
        )
        np.testing.assert_allclose(result["cache"][("a", "block")], 0.8)
        np.testing.assert_allclose(result["cache"][("b", "block")], 0.6)
        self.assertEqual([item.instruction for item in result["population"]], ["a", "b"])
