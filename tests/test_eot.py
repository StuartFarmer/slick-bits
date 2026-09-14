"""Check per-instance EoT routing and downstream answer evaluation."""

import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

from eot import EoT
from tests.providers import ScriptedProvider


class EoTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "eot/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_full_pipeline_scores_only_answer(self):
        seen = []

        async def evaluate(text):
            seen.append(text)
            return 1.0

        provider = ScriptedProvider(
            [
                json.dumps({"text": "combined"}),
                json.dumps({"text": "mutated"}),
                '{"index": 3}',
                json.dumps({"text": "rewritten problem"}),
                json.dumps({"text": "solution"}),
                json.dumps({"text": "42"}),
            ]
        )
        result = await EoT("Reason", provider, evaluate).run("What is the answer?")
        self.assertEqual(result["instruction"], "mutated")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(seen, ["42"])
        self.assertIn("combined", provider.calls[1])
        self.assertIn("mutated", provider.calls[3])
        self.assertIn("rewritten problem", provider.calls[4])
        self.assertIn("solution", provider.calls[5])

    async def test_bad_selection_and_failure_propagation(self):
        async def evaluate(text):
            return math.nan

        agent = EoT("Task", ScriptedProvider([]), evaluate)
        with self.assertRaises(ValueError):
            await agent.select("Problem", ["only"], provider=ScriptedProvider(['{"index": 1}']))
        with self.assertRaises(ValueError):
            await agent.mutate("a", provider=ScriptedProvider(['{"text": " "}']))
        with self.assertRaisesRegex(RuntimeError, "transport"):
            await agent.mutate("a", provider=ScriptedProvider([RuntimeError("transport")]))
        agent.provider = ScriptedProvider(
            [
                '{"text":"cross"}',
                '{"text":"mutate"}',
                '{"index":0}',
                '{"text":"rewrite"}',
                '{"text":"solve"}',
                '{"text":"answer"}',
            ]
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run("Problem")
