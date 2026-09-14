import unittest
from pathlib import Path
from unittest.mock import patch

from maps import MAPS, Cluster, Evaluation, Suggestions
from maps.agent import distance
from tests.providers import ScriptedProvider


class MAPSTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "maps/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_rule_acceptance_remeasures_pool_under_shared_rules(self):
        scored = []

        async def evaluate(text):
            scored.append(text)
            value = {
                "seed": 2,
                "seed\n\nEnsure that good": 4,
                "seed\n\nEnsure that bad": 1,
                "child\n\nEnsure that good": 8,
            }[text]
            return Evaluation(value, [{"error": "compile"}])

        def cluster(rows):
            return [Cluster("compile", len(rows), list(rows))]

        provider = ScriptedProvider(
            [
                Suggestions(suggestions=["mutation"]),
                "child",
                "reflection",
                "Ensure that good",
                "Ensure that bad",
            ]
        )
        result = await MAPS("Task", provider, evaluate, cluster).run(
            ["seed"], iterations=1, mutations=1, rule_candidates=2
        )
        self.assertEqual(result["best"].instruction, "child")
        self.assertEqual(result["rules"], ["Ensure that good"])
        self.assertEqual(result["evaluations"], 4)
        self.assertNotIn("child", scored)
        self.assertTrue(result["history"][0]["accepted"])
        self.assertEqual(result["tried"], ["compile", "compile"])

    async def test_rejected_rule_still_reduces_cluster_sampling_weight(self):
        async def evaluate(text):
            return Evaluation(0 if "rule" in text else 2, [{"error": "same"}])

        def cluster(rows):
            return [Cluster("same", 3, list(rows))]

        provider = ScriptedProvider(
            [
                Suggestions(suggestions=["change"]),
                "child",
                "reason",
                "rule",
                Suggestions(suggestions=["change again"]),
                "child2",
            ]
        )
        result = await MAPS("Task", provider, evaluate, cluster).run(
            ["seed"], iterations=2, mutations=1, rule_candidates=1
        )
        self.assertEqual(result["rules"], [])
        self.assertFalse(result["history"][0]["accepted"])
        self.assertEqual(result["optimizer_calls"], 6)
        self.assertEqual(distance("kitten", "sitting"), 3)

    async def test_duplicate_suggestions_rejected(self):
        async def evaluate(text):
            return Evaluation(1)

        provider = ScriptedProvider([Suggestions(suggestions=["same", "same"])])
        with self.assertRaisesRegex(ValueError, "distinct mutation"):
            await MAPS("Task", provider, evaluate, lambda rows: []).run(["seed"], mutations=2)


if __name__ == "__main__":
    unittest.main()
