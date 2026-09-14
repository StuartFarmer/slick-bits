import unittest

import numpy as np

from concentrate_attention import (
    ConcentrateAttention,
    Observation,
    concentration_loss,
    global_score,
)


class ConcentrationTests(unittest.IsolatedAsyncioTestCase):
    def test_gradient_below_normalization_floor(self):
        features = np.array([[1e-13, 2e-13], [0.1, 0.2], [0.4, 0.1]])
        labels = (0, 0, 1)
        analytic = concentration_loss(features, labels)[3]
        for column in range(2):
            plus, minus = features.copy(), features.copy()
            plus[0, column] += 1e-17
            minus[0, column] -= 1e-17
            numeric = (
                concentration_loss(plus, labels)[1] - concentration_loss(minus, labels)[1]
            ) / 2e-17
            np.testing.assert_allclose(analytic[0, column], numeric, rtol=1e-5)

    async def test_objective_derivatives_and_optimization(self):
        features = np.array([[0.1, 0.4], [0.2, 0.3], [0.4, 0.1]])
        labels = (0, 0, 1)
        _, _, ds, dc = concentration_loss(features, labels)
        for index in np.ndindex(features.shape):
            plus, minus = features.copy(), features.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numeric = (
                sum(concentration_loss(plus, labels)[:2])
                - sum(concentration_loss(minus, labels)[:2])
            ) / 2e-6
            self.assertAlmostEqual(numeric, (ds + dc)[index], places=5)

        async def observe(parameters):
            return Observation(
                float(parameters[0] ** 2),
                2 * parameters,
                features,
                np.zeros((*features.shape, 1)),
                labels,
            )

        result = await ConcentrateAttention("Any differentiable task", observe).run(
            np.array([1.0]), iterations=3, learning_rate=0.1
        )
        self.assertLess(result["best"]["parameters"][0], 1)
        self.assertEqual(result["model_calls"], 4)
        stable = global_score([[0.1, 0.4], [0.1, 0.4]], (0, 0), [0, 0])
        unstable = global_score([[0.1, 0.4], [0.4, 0.1]], (0, 0), [0, 0])
        self.assertGreater(stable, unstable)
