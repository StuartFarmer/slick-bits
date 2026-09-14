"""APOHF pair acquisition and locally fitted Bradley-Terry preferences."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from apohf import APOHF
from tests.providers import ScriptedProvider


def linear_surrogate(theta, contexts):
    return contexts @ theta, contexts


class APOHFTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "apohf/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_duels_fit_preference_model_and_rebuild_diagonal(self):
        values = {"weak": 0, "good": 1, "best": 2}

        async def evaluate(left, right):
            return float(values[left] > values[right])

        async def embed(prompts):
            return np.array([[values[p], 1] for p in prompts], dtype=float)

        agent = APOHF("Task", ScriptedProvider(["best"]), evaluate, embed, linear_surrogate)
        result = await agent.run(
            ["weak", "good"],
            np.zeros(2),
            examples=["demo"],
            additional_candidates=1,
            initial_pairs=3,
            iterations=2,
            training_steps=30,
            learning_rate=0.1,
        )
        self.assertGreater(result["theta"][0], 0)
        self.assertEqual(result["best"]["prompt"], "best")
        self.assertEqual(result["comparisons"], 5)
        self.assertEqual(result["training_runs"], 3)
        self.assertEqual(result["optimizer_calls"], 1)
        self.assertTrue(all(left != right for left, right in result["pairs"]))
        pairs = np.array(result["pairs"][:3])
        differences = agent.contexts[pairs[:, 0]] - agent.contexts[pairs[:, 1]]
        np.testing.assert_allclose(
            result["acquisitions"][0]["u"], 1 + np.sum(differences**2, axis=0)
        )
        self.assertEqual(result["acquisitions"][0]["pair"][0], 2)
        self.assertIn("demo", agent.provider.calls[0])

    async def test_paraphrase_generation_and_tie_feedback(self):
        async def evaluate(left, right):
            return 0.5

        async def embed(prompts):
            return np.eye(len(prompts))

        provider = ScriptedProvider(["alternative"])
        result = await APOHF("Task", provider, evaluate, embed, linear_surrogate).run(
            ["seed"],
            np.zeros(2),
            additional_candidates=1,
            candidate_method="rephrase",
            initial_pairs=1,
            iterations=1,
            training_steps=2,
        )
        np.testing.assert_allclose(result["theta"], 0)
        self.assertIn("Rephrase", provider.calls[0])
        self.assertEqual(result["preferences"], [0.5, 0.5])

    async def test_invalid_preference_fails_before_training(self):
        async def evaluate(left, right):
            return 2

        async def embed(prompts):
            return np.eye(2)

        agent = APOHF("Task", ScriptedProvider([]), evaluate, embed, linear_surrogate)
        with self.assertRaisesRegex(ValueError, "preference"):
            await agent.run(["a", "b"], np.zeros(2), initial_pairs=1)
        self.assertEqual(agent.comparisons, 1)
        self.assertEqual(agent.training_runs, 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            await agent.paraphrase("seed", provider=ScriptedProvider([" "]))
