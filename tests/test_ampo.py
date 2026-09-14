import unittest
from pathlib import Path
from unittest.mock import patch

from ampo import AMPO, Pattern, Patterns
from tests.providers import ScriptedProvider


class AMPOTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "ampo/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_ranked_patterns_independent_branches_and_pruning(self):
        async def failures(text):
            return [{"input": "x"}]

        async def score(text):
            return {"seed": 1, "pruned-high": 3, "pruned-middle": 2}[text]

        provider = ScriptedProvider(
            [
                "reason",
                Patterns(
                    patterns=[
                        Pattern(description="low", importance=1),
                        Pattern(description="high", importance=9),
                        Pattern(description="middle", importance=5),
                    ]
                ),
                "expanded-high",
                "pruned-high",
                "expanded-middle",
                "pruned-middle",
            ]
        )
        result = await AMPO("Task", provider, score, failures).run("seed", iterations=1, branches=2)
        self.assertEqual(result["best"].prompt, "pruned-high")
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(result["optimizer_calls"], 6)
        self.assertIn("Selected pattern: high", provider.calls[2])
        self.assertIn("Instruction: seed", provider.calls[4])
        self.assertIn("expanded-middle", provider.calls[5])

    async def test_pre_pruning_stops_losing_trajectory_and_keeps_best(self):
        async def failures(text):
            return [{"input": "x"}]

        async def score(text):
            return 5 if text == "seed" else 1

        provider = ScriptedProvider(
            [
                "reason",
                Patterns(patterns=[Pattern(description="pattern", importance=2)]),
                "expanded",
                "weak",
            ]
        )
        result = await AMPO("Task", provider, score, failures).run("seed", iterations=8, branches=1)
        self.assertEqual(result["best"].prompt, "seed")
        self.assertEqual(result["current"].prompt, "weak")
        self.assertEqual(result["evaluations"], 2)

    async def test_invalid_pattern_rejected_before_revision(self):
        async def failures(text):
            return [{"input": "x"}]

        async def score(text):
            return 1

        provider = ScriptedProvider(
            ["reason", Patterns(patterns=[Pattern(description=" ", importance=1)])]
        )
        with self.assertRaisesRegex(ValueError, "invalid generated pattern"):
            await AMPO("Task", provider, score, failures).run("seed")


if __name__ == "__main__":
    unittest.main()
