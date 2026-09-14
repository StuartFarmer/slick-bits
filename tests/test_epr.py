import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from epr import EPR, Example
from epr.agent import contrastive_loss
from tests.providers import ScriptedProvider


class EPRTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "epr/prompts")
        root.start()
        self.addCleanup(root.stop)

    def test_query_and_context_gradients_match_finite_differences(self):
        q, c = np.array([[0.2, 0.7], [-0.5, 0.1]]), np.array([[0.1, 0.3], [0.8, -0.4], [-0.3, 0.2]])
        positives = np.array([1, 0])
        _, dq, dc = contrastive_loss(q, c, positives)
        h = 1e-6
        for matrix, gradient in [(q, dq), (c, dc)]:
            for index in np.ndindex(matrix.shape):
                matrix[index] += h
                plus = contrastive_loss(q, c, positives)[0]
                matrix[index] -= 2 * h
                minus = contrastive_loss(q, c, positives)[0]
                matrix[index] += h
                self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * h), places=7)

    async def test_lm_ranking_supervises_real_encoder_updates_and_retrieval(self):
        async def evaluate(query, demo):
            return float(query.output != demo.output)

        async def encode(theta, texts, role):
            base = np.array([[1.0, 0.2] if "a" in text else [0.2, 1.0] for text in texts])
            jac = np.array([np.diag(row) for row in base])
            return base * theta, jac

        provider = ScriptedProvider(["answer"])
        agent = EPR("task", provider, evaluate, encode)
        pool = [
            Example("a one", "X"),
            Example("a two", "X"),
            Example("b one", "Y"),
            Example("b two", "Y"),
        ]
        result = await agent.run(
            pool,
            np.ones(2),
            candidate_count=3,
            positive_count=1,
            steps=3,
            batch_size=4,
            learning_rate=0.05,
        )
        self.assertEqual(result["evaluations"], 12)
        self.assertEqual(result["supervision"][0]["positives"], [1])
        self.assertEqual(result["supervision"][2]["positives"], [3])
        self.assertFalse(np.allclose(result["parameters"], 1))
        demos = await agent.retrieve("a query", count=1)
        self.assertEqual(demos[0].output, "X")
        await agent.answer("q", demos, provider=provider)
        self.assertIn("a one", provider.calls[0])

    async def test_nonfinite_teacher_loss_stops_training(self):
        async def evaluate(query, demo):
            return float("nan")

        agent = EPR("t", None, evaluate, None)
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run([Example("a", "x"), Example("b", "y")], np.ones(2))
        self.assertEqual(agent.encoder_calls, 0)
