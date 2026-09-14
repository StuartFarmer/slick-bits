"""Check PE2 context reasoning, momentum ancestry, and beam decisions."""

import unittest
from pathlib import Path
from unittest.mock import patch

from pe2 import PE2
from tests.providers import ScriptedProvider


class PE2Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "pe2/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_context_two_stages_history_and_hard_examples(self):
        provider = ScriptedProvider(
            ["analysis one", "better", "edit one", "analysis two", "best", "edit two"]
        )
        observed = []

        async def observe(text):
            observed.append(text)
            return [
                {"input": "bad case", "output": "wrong", "label": "right", "score": 0},
                {"input": "good case", "output": "right", "label": "right", "score": 1},
            ]

        async def evaluate(text):
            return {"seed": 0, "better": 1, "best": 2}[text]

        result = await PE2(
            "Teach geometry",
            provider,
            evaluate,
            observe,
            full_template="Question: {input}\nAnswer rule: {instruction}",
        ).run(["seed"], iterations=2, beam_size=1, children=1, momentum=True)
        self.assertEqual(result["best"].prompt, "best")
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(result["optimizer_calls"], 6)
        self.assertEqual(observed, ["seed", "better"])
        self.assertIn("Answer rule: seed", provider.calls[0])
        self.assertIn("bad case", provider.calls[0])
        self.assertNotIn("good case", provider.calls[0])
        self.assertIn("analysis one", provider.calls[1])
        self.assertIn("edit one", provider.calls[4])
        for context in provider.calls:
            self.assertIn("Teach geometry", context)

    async def test_initial_induction_duplicate_skip_and_archive_backtracking(self):
        async def observe(text):
            return [{"input": "x", "output": "no", "label": "yes", "score": 0}]

        async def evaluate(text):
            return 10 if text == "seed" else 1

        provider = ScriptedProvider(["seed", "analysis", "weak", "analysis again", "seed"])
        result = await PE2("Task", provider, evaluate, observe).run(
            [],
            demonstrations=[{"input": "x", "label": "yes"}],
            iterations=2,
            beam_size=1,
            children=1,
        )
        self.assertEqual(result["best"].prompt, "seed")
        self.assertEqual(result["evaluations"], 2)
        self.assertTrue(result["history"][-1]["duplicate"])
        self.assertIn("Current instruction: seed", provider.calls[3])

    async def test_blank_revision_fails_before_scoring(self):
        async def observe(text):
            return [{"score": 0}]

        async def evaluate(text):
            return 1

        agent = PE2("Task", ScriptedProvider(["analysis", " "]), evaluate, observe)
        with self.assertRaisesRegex(ValueError, "empty generated"):
            await agent.run(["seed"], children=1)
        self.assertEqual(agent.evaluations, 1)


if __name__ == "__main__":
    unittest.main()
