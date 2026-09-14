"""BPO inference preserves its source prompt and scores optional alternatives."""

import unittest
from pathlib import Path
from unittest.mock import patch

from bpo import BPO
from tests.providers import ScriptedProvider


class BPOTests(unittest.IsolatedAsyncioTestCase):
    async def test_frozen_input_and_selection(self):
        async def evaluate(text):
            return len(text)

        provider = ScriptedProvider(["short", "the best rewrite", "other"])
        with patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "bpo/prompts"):
            result = await BPO("original task", provider, evaluate).run(samples=3)
        self.assertEqual(result["best"]["prompt"], "the best rewrite")
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(provider.calls[0], provider.calls[1])
        self.assertIn("original task [/INST]", provider.calls[0])

    async def test_generated_failure_precedes_evaluation(self):
        async def evaluate(text):
            self.fail("must not score empty generation")

        with patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "bpo/prompts"):
            with self.assertRaisesRegex(ValueError, "response="):
                await BPO("Task", ScriptedProvider([" "]), evaluate).run()
