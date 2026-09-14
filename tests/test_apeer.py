import unittest
from pathlib import Path
from unittest.mock import patch

from apeer import APEER, Example
from tests.providers import ScriptedProvider


class APEERTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_baseline_history_and_two_step_search(self):
        provider = ScriptedProvider(["feedback", "better", "best", "feedback2", "mid", "tie"])

        async def respond(instruction, example):
            self.assertEqual(example, "input")
            return "observed response"

        async def evaluate(instruction):
            return {"initial": 1, "negative": 0, "better": 2, "best": 4, "mid": 3, "tie": 1}[
                instruction
            ]

        with patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "apeer/prompts"):
            result = await APEER("Any task", provider, respond, evaluate).run(
                "initial", "negative", [Example("input", "reference")], iterations=2
            )
        self.assertEqual(result["best"]["prompt"], "best")
        self.assertIn("mid", [item["prompt"] for item in result["positive"]])
        self.assertIn("tie", [item["prompt"] for item in result["negative"]])
        self.assertIn("Instruction: best", provider.calls[3])
        self.assertEqual(result["evaluations"], 6)
