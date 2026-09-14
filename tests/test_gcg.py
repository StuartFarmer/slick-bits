"""GCG follows gradient descent coordinates while leaving model work to callers."""

import unittest

import numpy as np

from gcg import GCG


class GCGTests(unittest.IsolatedAsyncioTestCase):
    async def test_lowest_gradient_tokens_and_best_loss(self):
        seen = []

        async def gradient(tokens):
            return np.array([[0.0, 4.0, -2.0], [0.0, -3.0, 5.0]])

        async def loss(tokens):
            seen.append(tuple(tokens))
            return -sum(tokens)

        result = await GCG(gradient, loss).run([0, 0], iterations=2, batch_size=2, top_k=1)
        self.assertEqual(result["best"]["tokens"], (2, 1))
        self.assertEqual(seen[:3], [(0, 0), (2, 0), (0, 1)])
        self.assertEqual(result["evaluations"], 5)

    async def test_disallowed_and_roundtrip_filter(self):
        async def gradient(tokens):
            return np.array([[2.0, -2.0, -1.0]])

        async def loss(tokens):
            return -sum(tokens)

        result = await GCG(gradient, loss, accept=lambda tokens: tokens != (2,)).run(
            [0], iterations=2, top_k=1, batch_size=2, forbidden_tokens=[1]
        )
        self.assertEqual(result["best"]["tokens"], (0,))
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["rejected_candidates"], 4)
