import itertools
import unittest

import numpy as np

from blackbox_prompt_learning import (
    BlackBoxPromptLearning,
    project_simplex,
    variance_reduced_gradient,
)


class BlackBoxPromptLearningTests(unittest.IsolatedAsyncioTestCase):
    def test_expected_centered_gradient_matches_tangent_derivative(self):
        probabilities = np.array([[0.3, 0.7]])
        losses = np.array([2.0, -1.0])
        expected = np.zeros_like(probabilities)
        for pair in itertools.product(range(2), repeat=2):
            samples = np.array(pair)[:, None]
            weight = np.prod(probabilities[0, list(pair)])
            expected += weight * variance_reduced_gradient(
                probabilities, samples, losses[list(pair)]
            )
        # Release's +/- score formula doubles the standard score-function gradient.
        self.assertAlmostEqual((expected[0, 0] - expected[0, 1]) / 2, losses[0] - losses[1])
        np.testing.assert_allclose(project_simplex(np.array([-2.0, -3.0])), [1.0, 0.0])

    async def test_learning_concentrates_on_low_loss_and_handles_zero_mass(self):
        async def evaluate(prompt, batch):
            return float(prompt[0] == "bad")

        result = await BlackBoxPromptLearning("task", evaluate).run(
            ["bad", "good"],
            ["batch"],
            prompt_length=1,
            epochs=40,
            samples_per_batch=20,
            learning_rate=0.1,
        )
        self.assertEqual(result["prompt"], ("good",))
        self.assertEqual(result["evaluations"], 800)
        np.testing.assert_allclose(result["probabilities"].sum(axis=1), [1.0])
        self.assertTrue(np.isfinite(result["probabilities"]).all())
