import unittest
from pathlib import Path
from unittest.mock import patch

from lcp import LCP
from tests.providers import ScriptedProvider


class LCPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "lcp/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_diverse_pool_contrast_and_best_ever(self):
        observed = []

        async def failures(text):
            observed.append(text)
            return [{"input": "case", "source_correct": True}]

        async def score(text):
            return {"seed": 3, "high": 5, "low": 1, "next": 2}[text]

        provider = ScriptedProvider(
            ["reason", "high", "low", "next", "reason2", "high", "low", "next"]
        )
        result = await LCP("Task", provider, score, failures).run(
            "seed", iterations=2, diversity=2, top_k=1
        )
        self.assertEqual(result["best"].prompt, "high")
        self.assertEqual(result["current"].prompt, "next")
        self.assertEqual(observed, ["seed", "next"])
        self.assertEqual(result["evaluations"], 4)
        self.assertEqual(result["optimizer_calls"], 8)
        self.assertIn("Score 5.0: high", provider.calls[3])
        self.assertIn("Score 1.0: low", provider.calls[3])
        self.assertEqual(result["history"][1]["good"][0].prompt, "high")

    async def test_adaptation_filters_source_errors_and_blank_rejected(self):
        async def failures(text):
            return [{"input": "source wrong", "source_correct": False}]

        async def score(text):
            return 1

        result = await LCP("Task", ScriptedProvider([]), score, failures).run(
            "seed", adaptation=True
        )
        self.assertEqual(result["optimizer_calls"], 0)
        with self.assertRaisesRegex(ValueError, "empty generated"):
            await LCP("Task", ScriptedProvider([" "]), score, failures).run("seed")

    async def test_collapsed_generation_does_not_contrast_a_prompt_with_itself(self):
        async def failures(text):
            return [{"input": "x"}]

        async def score(text):
            return 1

        provider = ScriptedProvider(["reason", "seed", "seed"])
        result = await LCP("Task", provider, score, failures).run("seed", diversity=2)
        self.assertEqual(result["optimizer_calls"], 3)
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["history"], [])


if __name__ == "__main__":
    unittest.main()
