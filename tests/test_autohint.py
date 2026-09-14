"""Check residual-only hint induction, sampling order, and iterative enrichment."""

import unittest
from pathlib import Path
from unittest.mock import patch

from autohint import AutoHint, Example
from tests.providers import ScriptedProvider


class AutoHintTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "autohint/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_all_residual_hints_generated_before_sampling_and_validation(self):
        actor = ScriptedProvider(["yes", "wrong", "wrong"])
        provider = ScriptedProvider(["hint b", "hint c", "combined"])

        async def evaluate(text):
            return len(text)

        result = await AutoHint("Classify plants", provider, evaluate, actor_provider=actor).run(
            "seed", [Example("a", "yes"), Example("b", "yes"), Example("c", "no")], sample_size=1
        )
        self.assertEqual(result["inference_calls"], 3)
        self.assertEqual(result["optimizer_calls"], 3)
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(len(result["history"][0]["hints"]), 2)
        self.assertEqual(len(result["history"][0]["selected"]), 1)
        self.assertEqual(result["best"]["prompt"], "seed\n\nHint: combined")
        self.assertIn("Input: b", provider.calls[0])
        for context in provider.calls + actor.calls:
            self.assertIn("Classify plants", context)

    async def test_balanced_and_cluster_sample_each_group(self):
        async def evaluate(text):
            return 1

        examples = [Example("a", "yes"), Example("b", "yes"), Example("c", "no")]
        for sampling in ("balanced", "cluster"):
            clustered = []

            def cluster(hints):
                clustered.extend(hint.hint for hint in hints)
                return [0, 0, 1]

            result = await AutoHint(
                "Task",
                ScriptedProvider(["h1", "h2", "h3", "summary"]),
                evaluate,
                cluster=cluster,
                actor_provider=ScriptedProvider(["bad"] * 3),
            ).run("seed", examples, sampling=sampling, sample_size=1)
            self.assertEqual(len(result["history"][0]["selected"]), 2)
            if sampling == "cluster":
                self.assertEqual(clustered, ["h1", "h2", "h3"])

    async def test_no_errors_stops_and_blank_summary_propagates(self):
        async def evaluate(text):
            return 1

        result = await AutoHint("Task", ScriptedProvider(["yes"]), evaluate).run(
            "seed", [Example("x", "yes")], iterations=4
        )
        self.assertEqual(result["optimizer_calls"], 0)
        self.assertEqual(result["evaluations"], 1)
        agent = AutoHint("Task", ScriptedProvider(["wrong", "hint", " "]), evaluate)
        with self.assertRaisesRegex(ValueError, "empty hint summary"):
            await agent.run("seed", [Example("x", "yes")])
        self.assertEqual(agent.evaluations, 1)


if __name__ == "__main__":
    unittest.main()
