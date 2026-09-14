import unittest

import numpy as np

from bbt.cma import CMA


class CMATests(unittest.TestCase):
    def test_mean_is_rank_weighted_and_covariance_learns(self):
        cma = CMA(np.zeros(2), 1, 4, np.random.default_rng(0))
        points = np.array([[2, 0], [1, 0], [-1, 0], [0, 1]])
        cma.tell(points, np.array([0, 1, 3, 4]))
        first_weight = np.log(2.5) / (np.log(2.5) + np.log(1.25))
        np.testing.assert_allclose(cma.mean, [1 + first_weight, 0])
        self.assertGreater(cma.covariance[0, 0], cma.covariance[1, 1])
        self.assertGreater(np.linalg.eigvalsh(cma.covariance).min(), 0)
        np.testing.assert_array_equal(cma.best, [2, 0])

    def test_rotated_quadratic_search_reduces_loss(self):
        cma = CMA(np.array([4.0, -4.0]), 1, 8, np.random.default_rng(4))
        matrix = np.array([[5, 4], [4, 5]])
        for _ in range(60):
            points = cma.ask()
            losses = np.einsum("bi,ij,bj->b", points, matrix, points)
            cma.tell(points, losses)
        self.assertLess(cma.best_loss, 1e-5)
        self.assertLess(np.linalg.norm(cma.mean), 0.01)

    def test_nonfinite_observation_rejected_before_state_change(self):
        cma = CMA(np.zeros(2), 1, 4, np.random.default_rng(0))
        with self.assertRaisesRegex(ValueError, "finite"):
            cma.tell(cma.ask(), np.array([1, 2, np.nan, 0]))
        self.assertEqual(cma.generation, 0)
