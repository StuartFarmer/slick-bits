import unittest

import numpy as np

from dmo import DMO, matching_loss


class DMOTests(unittest.IsolatedAsyncioTestCase):
    def test_relative_matching_gradient(self):
        coefficients = np.array([0.2, -0.3])
        metrics = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
        target = np.array([2.0, 3.0])
        _, gradient, _, _ = matching_loss(coefficients, metrics, target)
        for i in range(2):
            plus, minus = coefficients.copy(), coefficients.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            numerical = (
                matching_loss(plus, metrics, target)[0] - matching_loss(minus, metrics, target)[0]
            ) / 2e-6
            self.assertAlmostEqual(gradient[i], numerical, places=8)

    async def test_fits_targets_then_resamples_and_reports_budget(self):
        count = 0

        async def sample(prefix, temperature, seed):
            nonlocal count
            count += 1
            return "high" if count % 2 else "low"

        async def metrics(prefix, text):
            return np.array([{"reference": 1.4, "low": 1.0, "high": 3.0}[text]])

        result = await DMO("match", sample, metrics).run(
            [("dev", "reference")],
            ["test"],
            fit_samples_per_prefix=4,
            iterations=500,
            learning_rate=0.02,
            particles=4,
        )
        self.assertLess(result["error"], 0.003)
        self.assertEqual(result["generations"], 8)
        self.assertEqual(result["evaluations"], 9)
        self.assertGreater(result["outputs"][0]["weights"][1], result["outputs"][0]["weights"][0])

    async def test_zero_measured_target_is_explicit_failure(self):
        async def sample(prefix, temperature, seed):
            return "x"

        async def evaluate(prefix, text):
            return np.zeros(1)

        with self.assertRaisesRegex(ValueError, "zero target"):
            await DMO("match", sample, evaluate).run([("q", "r")], [], fit_samples_per_prefix=1)
