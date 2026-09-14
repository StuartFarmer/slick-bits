import unittest

import numpy as np

from prompt_oirl import Observation, PromptOIRL, logistic_derivatives


class PromptOIRLTests(unittest.IsolatedAsyncioTestCase):
    def test_logistic_newton_derivatives(self):
        logits, labels = np.array([-0.7, 1.3]), np.array([1.0, 0.0])
        gradient, hessian = logistic_derivatives(logits, labels)
        eps = 1e-5
        plus = np.logaddexp(0, logits + eps) - labels * (logits + eps)
        minus = np.logaddexp(0, logits - eps) - labels * (logits - eps)
        np.testing.assert_allclose(gradient, (plus - minus) / (2 * eps), atol=1e-9)
        gp, _ = logistic_derivatives(logits + eps, labels)
        gm, _ = logistic_derivatives(logits - eps, labels)
        np.testing.assert_allclose(hessian, (gp - gm) / (2 * eps), atol=1e-9)

    async def test_query_specific_selection_without_target_model(self):
        async def embed(text):
            return np.array([{"q0": 0, "q1": 1, "p0": 0, "p1": 1}[text]])

        # Different query margins make the root split useful before conditional prompt splits.
        observations = [Observation("q0", "p0", 1.0)] * 8 + [Observation("q0", "p1", 0.0)] * 2
        observations += [Observation("q1", "p0", 0.0)] * 8 + [Observation("q1", "p1", 1.0)] * 2
        result = await PromptOIRL("task", embed).run(
            observations,
            ["q0", "q1"],
            ["p0", "p1"],
            rounds=30,
            learning_rate=0.3,
            min_child_weight=0.01,
        )
        self.assertEqual([s["prompt"] for s in result["selections"]], ["p0", "p1"])
        self.assertLess(result["losses"][-1], result["losses"][0])
        self.assertEqual(len(result["trees"]), 30)
