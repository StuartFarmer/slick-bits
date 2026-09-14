"""Check vision-conditioned episodic memory and class-token preservation."""

import unittest
from pathlib import Path
from unittest.mock import patch

from ipo import IPO, Evaluation
from tests.providers import ScriptedProvider


class IPOTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "ipo/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_visual_memory_baseline_and_loss_ties(self):
        results = {
            "base <CLASS>": Evaluation(1, 3),
            "a <CLASS>": Evaluation(2, 2),
            "b <CLASS>": Evaluation(2, 1),
        }

        async def evaluate(text):
            return results[text]

        provider = ScriptedProvider(
            ['{"prompts":["a <CLASS>","b <CLASS>"]}', '{"prompts":["a <CLASS>","b <CLASS>"]}']
        )
        agent = IPO("Task", provider, evaluate, ["striped object"])
        result = await agent.run("base <CLASS>", rounds=2, candidates=2, memory_size=1)
        self.assertEqual(result["best"].prompt, "b <CLASS>")
        self.assertEqual(result["evaluations"], 3)
        for term in (
            "striped object",
            r"base \u003cCLASS\u003e",
            r"b \u003cCLASS\u003e",
            "loss",
            "accuracy",
        ):
            self.assertIn(term, provider.calls[1])

    async def test_missing_class_token_and_nonfinite_loss(self):
        async def evaluate(text):
            return Evaluation(0, float("inf"))

        agent = IPO("Task", ScriptedProvider([]), evaluate)
        with self.assertRaises(ValueError):
            await agent.run(rounds=0)
        with self.assertRaises(ValueError):
            await agent.propose([], 1, provider=ScriptedProvider(['{"prompts":["no token"]}']))

    async def test_evaluator_exception_is_not_cached(self):
        async def evaluate(text):
            raise RuntimeError("model unavailable")

        agent = IPO("Task", ScriptedProvider([]), evaluate)
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            await agent.run()
        self.assertEqual(agent.archive, {})
