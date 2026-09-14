import unittest

import numpy as np

from blackvip import BlackVIP


class BlackVIPTests(unittest.IsolatedAsyncioTestCase):
    async def test_spsa_derivative_and_gradient_correction(self):
        async def coordinate(parameters, images):
            return np.full_like(images, parameters[0])

        async def forward(images):
            return np.column_stack([images[:, 0], np.zeros(len(images))])

        agent = BlackVIP("Task", coordinate, forward, clip=lambda image: image)
        result = await agent.run(
            np.array([0.0]),
            [(np.zeros((1, 1)), np.array([0]))],
            epochs=2,
            learning_rate=0.1,
            perturbation=1e-5,
            alpha=0,
            gamma=0,
            momentum=0.5,
        )
        first, second = result["history"]
        self.assertAlmostEqual(first["gradient"][0], -0.5, places=8)
        self.assertAlmostEqual(first["parameters"][0], 0.075, places=8)
        expected_second_gradient = -1 / (1 + np.exp(0.075))
        self.assertAlmostEqual(second["gradient"][0], expected_second_gradient, places=8)
        self.assertAlmostEqual(second["velocity"][0], -0.25 + expected_second_gradient, places=8)
        self.assertEqual(result["model_calls"], 4)

    async def test_input_dependence_and_averaging_call_budget(self):
        inputs = []

        async def coordinate(parameters, images):
            inputs.append(images.copy())
            return parameters[0] * images

        async def forward(images):
            return np.column_stack([images[:, 0], -images[:, 0]])

        agent = BlackVIP("Task", coordinate, forward)
        result = await agent.run(
            np.zeros(1), [(np.array([[0.2], [0.4]]), np.array([0, 1]))], averages=3
        )
        self.assertEqual(result["coordinator_calls"], 6)
        self.assertEqual(len(result["history"][0]["loss_pairs"]), 3)
        np.testing.assert_array_equal(inputs[0], [[0.2], [0.4]])
