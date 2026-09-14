import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from propane import PROPANE
from tests.providers import ScriptedProvider


class PROPANETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "propane/prompts")
        root.start()
        self.addCleanup(root.stop)

    def model(self, provider):
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])

        async def forward(weights, documents):
            logits = weights.sum(axis=0) @ matrix
            return np.broadcast_to(logits, (*documents.shape, 3)).copy()

        async def backward(weights, documents, cotangent):
            gradient = cotangent.sum(axis=(0, 1)) @ matrix.T
            return np.broadcast_to(gradient, weights.shape).copy()

        return PROPANE(
            "Match documents",
            provider,
            forward,
            backward,
            lambda text: np.array([0, 0]),
            lambda tokens: " ".join(map(str, tokens)),
            3,
        )

    async def test_warm_start_gcg_one_coordinate_and_likelihood(self):
        provider = ScriptedProvider(["initial instruction"])
        result = await self.model(provider).run(np.array([[1], [1]]), iterations=2, top_k=1)
        np.testing.assert_array_equal(result["tokens"], [1, 1])
        self.assertEqual(result["forward_calls"], 7)
        self.assertEqual(result["backward_calls"], 2)
        self.assertEqual(result["optimizer_calls"], 1)
        first = result["history"][0]
        self.assertTrue(all(np.count_nonzero(tokens) <= 1 for tokens, _ in first["proposals"]))
        self.assertLess(result["loss"], 0.01)

    async def test_local_likelihood_cotangent_matches_finite_difference(self):
        agent = self.model(ScriptedProvider([]))
        agent.forward_calls = agent.backward_calls = 0
        tokens, documents = np.array([0, 2]), np.array([[1], [2]])
        _, _, gradient = await agent._objective(tokens, documents, gradient=True)
        weights = np.eye(3)[tokens]
        matrix = np.diag([1.0, 3.0, 1.0])

        def objective(w):
            logits = w.sum(axis=0) @ matrix
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            return -np.log(probs[documents]).mean()

        delta = np.zeros_like(weights)
        delta[0, 1] = 1e-5
        numeric = (objective(weights + delta) - objective(weights - delta)) / 2e-5
        self.assertAlmostEqual(gradient[0, 1], numeric, places=8)
