import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gpo import GPO
from tests.providers import ScriptedProvider


def tagged(text):
    return f"<START>{text}<END>"


class GPOTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "gpo/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_momentum_gains_novelty_selection_and_schedule(self):
        async def failures(text):
            return [{"input": "x", "output": "wrong"}]

        async def score(text):
            return {"seed": 10, "weak": 2, "better": 4}[text]

        provider = ScriptedProvider(
            list(map(tagged, ["g1", "seed", "weak", "g2", "seed", "better"]))
        )
        result = await GPO("Task", provider, score, failures).run(
            "seed",
            iterations=2,
            candidates=2,
            schedule="linear",
            initial_words=10,
            final_words=2,
            recipe="feedback",
            selection="importance",
            momentum="feedback",
        )
        self.assertEqual(result["best"].prompt, "seed")
        self.assertEqual(result["current"].prompt, "better")
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual([item.gain for item in result["gradients"]], [-8, 2])
        self.assertEqual([item["words"] for item in result["history"]], [6, 2])
        self.assertEqual(result["history"][1]["feedback"], ["g2", "g1"])
        self.assertIn("g1", provider.calls[4])

    async def test_parameter_momentum_and_visited_fallback(self):
        async def failures(text):
            return []

        async def score(text):
            return 4

        provider = ScriptedProvider([tagged("gradient"), tagged("seed")])
        result = await GPO("Task", provider, score, failures).run(
            "seed",
            iterations=1,
            candidates=1,
            momentum="parameter",
            schedule="fixed",
            recipe="feedback",
            selection="importance",
        )
        self.assertEqual(result["current"].score, 4)
        self.assertEqual(result["evaluations"], 1)
        self.assertIn("Score 4.0: seed", provider.calls[1])

    async def test_missing_delimiter_rejected(self):
        async def failures(text):
            return []

        async def score(text):
            return 1

        with self.assertRaisesRegex(ValueError, "START/END"):
            await GPO("Task", ScriptedProvider(["untagged"]), score, failures).run("seed")

    async def test_recommended_recipe_retrieves_relevance_without_feedback_calls(self):
        async def score(text):
            return {"seed": 0, "east": 1, "north": 2, "final": 3}[text]

        async def embed(texts):
            return np.array([{"seed": [0, 1], "east": [1, 0], "north": [0, 2]}[t] for t in texts])

        async def failures(text):
            self.fail("recommended GPO does not generate a feedback gradient")

        provider = ScriptedProvider(list(map(tagged, ["east", "north", "final"])))
        result = await GPO("Task", provider, score, failures, embed=embed).run(
            "seed", iterations=3, candidates=1, memory=2, initial_words=10, final_words=2
        )
        self.assertEqual(result["optimizer_calls"], 3)
        self.assertEqual(result["gradients"], [])
        self.assertEqual([p.prompt for p in result["history"][2]["parameters"]], ["north", "seed"])
        self.assertEqual(result["history"][-1]["words"], 2)
        self.assertEqual(result["best"].prompt, "final")


if __name__ == "__main__":
    unittest.main()
