"""PEZ applies projected gradients to latent rather than projected embeddings."""

import unittest

import numpy as np

from pez import PEZ


class PEZTests(unittest.IsolatedAsyncioTestCase):
    async def test_cosine_projection_and_latent_adam_step(self):
        points = []

        async def gradient(projected):
            points.append(projected.copy())
            return np.array([[1.0, -1.0]])

        async def evaluate(ids):
            return 0.0 if ids == (1,) else 1.0

        result = await PEZ("Task", np.eye(2), gradient, evaluate).run(
            np.array([[2.0, 0.0]]), iterations=2, learning_rate=1.0, weight_decay=0
        )
        np.testing.assert_array_equal(points[0], [[1.0, 0.0]])
        self.assertEqual(result["best"]["tokens"], (1,))
        self.assertEqual(result["gradient_calls"], 2)
        self.assertEqual(result["evaluations"], 3)
        self.assertLess(result["latent"][0, 0], 0.01)
