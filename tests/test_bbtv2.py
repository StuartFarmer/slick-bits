import unittest

import numpy as np

from bbtv2 import BBTv2


class BBTv2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_layerwise_context_changes_and_persistent_strategies(self):
        seen = []

        async def evaluate(prompts):
            seen.append(prompts.copy())
            return float(np.sum((prompts - 1) ** 2))

        agent = BBTv2("Task", evaluate, np.zeros((2, 1)), projections=[np.eye(1), np.eye(1)])
        result = await agent.run(budget=16, population=4, sigma=1)
        self.assertEqual(result["evaluations"], 17)
        self.assertEqual([item["layer"] for item in result["history"]], [0, 1, 0, 1])
        self.assertTrue(all(item[1, 0] == 0 for item in seen[:4]))
        first_selected = result["history"][0]["prompts"][0, 0]
        self.assertTrue(all(item[0, 0] == first_selected for item in seen[4:8]))
        self.assertEqual([state.generation for state in agent.strategies], [2, 2])
        self.assertAlmostEqual(result["loss"], np.sum((result["prompts"] - 1) ** 2))
