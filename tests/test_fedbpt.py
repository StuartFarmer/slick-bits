import unittest

import numpy as np

from fedbpt import FedBPT


class FedBPTTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_ratio_regularization_and_server_variance_scaling(self):
        calls = []

        async def evaluate(client, prompt, perturbed):
            calls.append((client, perturbed))
            return 2.0 if perturbed else float(1 + (prompt[0] - {"a": 1, "b": -1}[client]) ** 2)

        agent = FedBPT("Task", evaluate, ["a", "b"], np.zeros(1), np.eye(1))
        result = await agent.run(rounds=1, local_steps=2, population=4, sigma=1)
        self.assertEqual(result["evaluations"], 36)
        self.assertEqual(sum(perturbed for _, perturbed in calls), 18)
        history = result["history"][0]
        expected = np.sqrt(
            sum(sum(s * s for s in report["sigmas"]) / 2 for report in history["reports"]) / 2
        )
        self.assertAlmostEqual(history["aggregate_sigma"], expected)
        self.assertAlmostEqual(history["sigma"], history["adaptation_ratio"])
        self.assertEqual(agent.server.generation, 1)

    async def test_no_regularization_does_not_query_perturbed_data(self):
        async def evaluate(client, prompt, perturbed):
            self.assertFalse(perturbed)
            return float((prompt[0] - 1) ** 2)

        result = await FedBPT("Task", evaluate, ["a", "b", "c"], np.zeros(1), np.eye(1)).run(
            rounds=2, clients_per_round=2, local_steps=1, population=4, regularize=False
        )
        self.assertEqual(result["evaluations"], 20)
        self.assertTrue(all(len(item["reports"]) == 2 for item in result["history"]))
