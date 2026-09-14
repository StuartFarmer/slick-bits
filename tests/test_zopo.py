"""ZOPO local posterior derivatives, nearest projection and bounded exploration."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests.providers import ScriptedProvider
from zopo import ZOPO


def linear_features(points):
    features = np.column_stack((points[:, 0], np.ones(len(points))))
    derivatives = np.zeros((len(points), 2, 1))
    derivatives[:, 0, 0] = 1
    return features, derivatives


class ZOPOTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "zopo/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_local_gp_gradient_ascent_projection_and_budget(self):
        values = {"low": 0, "mid": 1, "good": 2, "best": 3}

        async def embed(prompts):
            return np.array([[values[p]] for p in prompts], dtype=float)

        async def evaluate(text):
            return values[text]

        agent = ZOPO("Task", ScriptedProvider(["best"]), evaluate, embed, linear_features)
        result = await agent.run(
            ["low", "mid", "good"],
            additional_candidates=1,
            examples=["demo"],
            initial_samples=2,
            max_evaluations=4,
            learning_rate=0.5,
            uncertainty_threshold=100,
            max_steps=1,
        )
        self.assertEqual(result["best"]["prompt"], "best")
        self.assertEqual(result["evaluations"], 4)
        self.assertEqual(len({r["arm"] for r in result["records"]}), 4)
        self.assertTrue(result["steps"])
        self.assertGreater(result["steps"][0]["update"][0], 0)
        point = np.array([1.2])
        mean, variance, gradient = agent.posterior(point)
        delta = 1e-5
        numerical = (agent.posterior(point + delta)[0] - agent.posterior(point - delta)[0]) / (
            2 * delta
        )
        self.assertAlmostEqual(gradient[0], numerical, places=6)
        self.assertTrue(np.isfinite(mean))
        self.assertGreaterEqual(variance, 0)
        self.assertIn("demo", agent.provider.calls[0])

    async def test_zero_gradient_exploration_terminates_at_pool_exhaustion(self):
        async def embed(prompts):
            return np.arange(len(prompts), dtype=float)[:, None]

        async def evaluate(text):
            return 0

        agent = ZOPO("Task", ScriptedProvider([]), evaluate, embed, linear_features)
        result = await agent.run(
            ["a", "b", "c"], initial_samples=1, max_evaluations=20, neighbors=1
        )
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(
            [r["phase"] for r in result["records"]], ["initial", "exploration", "exploration"]
        )
        self.assertEqual(result["steps"], [])

    async def test_failures_propagate(self):
        async def embed(prompts):
            return np.ones((len(prompts), 1))

        async def evaluate(text):
            return np.nan

        agent = ZOPO("Task", ScriptedProvider([]), evaluate, embed, linear_features)
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run(["seed"], initial_samples=1, max_evaluations=1)
        self.assertEqual(agent.evaluations, 1)
        with self.assertRaisesRegex(ValueError, "empty"):
            await agent.induce([], provider=ScriptedProvider([" "]))
