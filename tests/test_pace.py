"""Check paper-based PACE actor/critic evidence and candidate selection."""

import unittest
from pathlib import Path
from unittest.mock import patch

from pace import PACE, Example
from tests.providers import ScriptedProvider


class PACETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "pace/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_independent_actor_critic_pairs_and_frozen_candidate_parent(self):
        actor = ScriptedProvider(["a1", "a2", "b1", "b2"])
        provider = ScriptedProvider(
            ["critique a1", "critique a2", "weak", "critique b1", "critique b2", "strong"]
        )

        async def evaluate(text):
            return {"seed": 1, "weak": 0, "strong": 2}[text]

        result = await PACE("Teach chemistry", provider, evaluate, actor).run(
            "seed", [Example("input", "answer")], actors=2, candidates=2
        )
        self.assertEqual(result["best"]["prompt"], "strong")
        self.assertEqual(result["actor_calls"], 4)
        self.assertEqual(result["optimizer_calls"], 6)
        self.assertEqual(result["evaluations"], 3)
        self.assertIn("Prediction: a1", provider.calls[0])
        self.assertIn("Ground truth: answer", provider.calls[0])
        self.assertIn("critique a2", provider.calls[2])
        self.assertIn("Instruction: seed", provider.calls[-1])
        for context in actor.calls + provider.calls:
            self.assertIn("Teach chemistry", context)

    async def test_trajectory_can_lose_while_global_best_survives(self):
        provider = ScriptedProvider(["wrong", "advice", "weak", "wrong", "advice", "worse"])

        async def evaluate(text):
            return {"seed": 3, "weak": 2, "worse": 1}[text]

        result = await PACE("Task", provider, evaluate).run(
            "seed", [Example("input", "answer")], iterations=2, actors=1, candidates=1
        )
        self.assertEqual(result["best"]["prompt"], "seed")
        self.assertEqual(result["current"]["prompt"], "worse")
        self.assertIn("Instruction: weak", provider.calls[3])

    async def test_blank_update_is_rejected_before_evaluation(self):
        async def evaluate(text):
            return 1

        agent = PACE("Task", ScriptedProvider(["wrong", "advice", " "]), evaluate)
        with self.assertRaisesRegex(ValueError, "empty instruction"):
            await agent.run("seed", [Example("input", "answer")], actors=1, candidates=1)
        self.assertEqual(agent.evaluations, 1)


if __name__ == "__main__":
    unittest.main()
