import unittest

import numpy as np

from bbt import BBT


class BBTTests(unittest.IsolatedAsyncioTestCase):
    async def test_intrinsic_projection_search_and_budget(self):
        seen = []

        async def evaluate(prompt):
            seen.append(prompt.copy())
            self.assertAlmostEqual(prompt[1], 2 * prompt[0])
            return float(np.sum((prompt - [1, 2]) ** 2))

        result = await BBT("Task", evaluate, np.zeros(2), projection=np.array([[1], [2]])).run(
            budget=41, population=4, sigma=1, seed=4
        )
        self.assertEqual(result["evaluations"], 40)
        self.assertEqual(len(seen), 40)
        self.assertLess(result["loss"], 0.1)
        self.assertAlmostEqual(result["loss"], min(np.sum((p - [1, 2]) ** 2) for p in seen))

    async def test_insufficient_population_budget_makes_no_queries(self):
        async def evaluate(prompt):
            self.fail("no full population available")

        result = await BBT("Task", evaluate, np.zeros(2), projection=np.eye(2)).run(
            budget=3, population=4
        )
        self.assertEqual(result["evaluations"], 0)
        self.assertIsNone(result["loss"])
