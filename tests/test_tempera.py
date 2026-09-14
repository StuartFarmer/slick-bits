import unittest

import numpy as np

from tempera import TEMPERA, Edit, PromptState, generalized_advantages, ppo_loss


class TEMPERATests(unittest.IsolatedAsyncioTestCase):
    def test_ppo_gradient_and_clipped_branches(self):
        logits = np.array([0.3, -0.2])
        args = (0, -0.5, 0.2, 0.7, 1.0, 0.2, 0.5, 0.03)
        _, dl, dv = ppo_loss(logits, 0.3, *args)
        for i in range(2):
            plus, minus = logits.copy(), logits.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            expected = (ppo_loss(plus, 0.3, *args)[0] - ppo_loss(minus, 0.3, *args)[0]) / 2e-6
            self.assertAlmostEqual(dl[i], expected, places=7)
        expected = (
            ppo_loss(logits, 0.300001, *args)[0] - ppo_loss(logits, 0.299999, *args)[0]
        ) / 2e-6
        self.assertAlmostEqual(dv, expected, places=7)
        _, clipped, _ = ppo_loss(np.array([4.0, 0.0]), 0.0, 0, -2.0, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0)
        np.testing.assert_array_equal(clipped, [0, 0])

    def test_gae_uses_future_rewards(self):
        advantage, target = generalized_advantages(
            np.array([1.0, 2.0]), np.array([0.3, 0.4]), 0.9, 1.0
        )
        np.testing.assert_allclose(target, [2.8, 2.0])
        np.testing.assert_allclose(advantage, [2.5, 1.6])

    async def test_edits_training_and_label_free_inference(self):
        calls = []

        async def encode(query, state, actions):
            return np.array([float(state.examples[0])])

        async def forward(parameters, features):
            return parameters[:4], np.eye(5)[:4], parameters[4], np.eye(5)[4]

        async def evaluate(query, state):
            calls.append(query)
            return float(state.examples[0])

        agent = TEMPERA("task", encode, forward, evaluate, pool_size=2, verbalizer_count=2)
        initial = PromptState((0,), (0,))
        result = await agent.run(
            np.array([0.0, 2.0, 0.0, 0.0, 0.0]),
            ["training"],
            initial,
            iterations=1,
            max_edits=2,
            epochs=1,
        )
        self.assertGreater(result["evaluations"], 1)
        count = len(calls)
        agent.parameters[:4] = [0, 10, 0, 0]
        edited = await agent.edit("held-out", initial, max_edits=1)
        self.assertEqual(edited.examples, (1,))
        self.assertEqual(len(calls), count)
        swapped = agent.apply(PromptState((0, 1), (0, 1)), Edit("swap", 0, 1))
        self.assertEqual(swapped, PromptState((1, 0), (1, 0)))
