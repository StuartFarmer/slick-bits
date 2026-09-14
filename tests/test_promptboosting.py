import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from promptboosting import PromptBoosting
from tests.providers import ScriptedProvider


class PromptBoostingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "promptboosting/prompts"
        )
        root.start()
        self.addCleanup(root.stop)

    async def test_weighted_verbalizer_search_boosts_errors_and_keeps_best_prefix(self):
        async def token_scores(provider, contexts):
            predictions = [0, 0, 0, 1] if "first" in contexts[0] else [0, 1, 1, 1]
            return np.eye(2)[predictions] * 0.8 + 0.1

        evaluations = []

        async def evaluate(predictions):
            evaluations.append(predictions.copy())
            return 1.0 if len(evaluations) == 1 else 0.5

        agent = PromptBoosting("task", ScriptedProvider([]), evaluate, token_scores)
        result = await agent.run(
            ["first {input}", "second {input}"],
            list("abcd"),
            np.array([0, 0, 1, 1]),
            list("abcd"),
            classes=2,
            rounds=2,
            top_tokens=1,
            random_templates=False,
        )
        self.assertEqual(len(result["all_learners"]), 2)
        self.assertEqual(len(result["learners"]), 1)
        np.testing.assert_allclose(
            result["history"][0]["updated_weights"], [1 / 6, 1 / 6, 1 / 2, 1 / 6]
        )
        self.assertAlmostEqual(result["history"][1]["error"], 1 / 6)
        self.assertEqual(result["model_calls"], 4)
        np.testing.assert_array_equal(await agent.predict(list("abcd")), [0, 0, 0, 1])

    async def test_perfect_learner_has_finite_weight_and_stops(self):
        async def scores(provider, contexts):
            return np.eye(2)

        async def evaluate(predictions):
            return 1.0

        result = await PromptBoosting("t", None, evaluate, scores).run(
            ["{input}"],
            ["a", "b"],
            np.array([0, 1]),
            ["a", "b"],
            classes=2,
            rounds=10,
            top_tokens=1,
        )
        self.assertEqual(result["evaluations"], 1)
        self.assertTrue(np.isfinite(result["learners"][0].weight))

    async def test_chance_only_candidates_are_rejected(self):
        async def scores(provider, contexts):
            return np.ones((2, 2))

        agent = PromptBoosting("t", None, None, scores)
        with self.assertRaisesRegex(ValueError, "chance"):
            await agent.run(
                ["{input}"],
                ["a", "b"],
                np.array([0, 1]),
                ["a", "b"],
                classes=2,
                rounds=1,
                top_tokens=1,
            )
        self.assertEqual(agent.evaluations, 0)
