import itertools
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from automate_cot import AutomateCoT
from automate_cot.agent import policy_gradient, project_simplex
from tests.providers import ScriptedProvider


class AutomateCoTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "automate_cot/prompts"
        )
        root.start()
        self.addCleanup(root.stop)

    def test_expected_signed_estimator_matches_twice_directional_loss_derivative(self):
        probabilities = np.array([[0.5, 0.5]])
        losses = np.array([1.0, 3.0])
        gradient = (
            sum(
                policy_gradient(probabilities, np.array(pair)[:, None], losses[list(pair)])
                for pair in itertools.product(range(2), repeat=2)
            )
            / 4
        )
        direction = np.array([[1.0, -1.0]])
        h = 1e-6
        numerical = (
            (probabilities + h * direction) @ losses - (probabilities - h * direction) @ losses
        ) / (2 * h)
        self.assertAlmostEqual(float(np.sum(gradient * direction)), 2 * numerical[0], places=6)

    async def test_policy_learns_lower_loss_and_counts_all_samples(self):
        async def evaluate(demos):
            return sum(item == "bad" for item in demos)

        provider = ScriptedProvider(["answer"])
        agent = AutomateCoT("task", provider, evaluate)
        result = await agent.run(
            ["good", "bad"], slots=2, steps=3, samples_per_step=20, learning_rate=0.1
        )
        self.assertEqual(result["demonstrations"], ("good", "good"))
        self.assertEqual(result["evaluations"], 61)
        self.assertTrue(np.all(result["probabilities"][:, 0] > 0.5))
        await agent.answer("q", provider=provider)
        self.assertIn("good", provider.calls[0])

    def test_projection_handles_rows_below_unit_mass(self):
        projected = project_simplex(np.array([-0.5, 0.1, 0.2]))
        self.assertAlmostEqual(projected.sum(), 1.0)
        self.assertGreaterEqual(projected.min(), 0.0001)

    async def test_bad_measured_loss_propagates(self):
        async def evaluate(demos):
            return float("inf")

        with self.assertRaisesRegex(ValueError, "finite"):
            await AutomateCoT("task", None, evaluate).run(["a", "b"], slots=1, steps=1)
