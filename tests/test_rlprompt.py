import unittest

import numpy as np

from rlprompt import RLPrompt, soft_q_loss


class RLPromptTests(unittest.IsolatedAsyncioTestCase):
    def test_compound_soft_q_gradient_matches_finite_difference(self):
        logits = np.array([[0.2, -0.4], [0.6, 0.1], [-0.2, 0.5]])
        target = logits + 0.13
        actions = np.array([0, 1, 1])
        _, gradient = soft_q_loss(logits, target, actions, 1.2)
        numerical = np.zeros_like(logits)
        for index in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numerical[index] = (
                soft_q_loss(plus, target, actions, 1.2)[0]
                - soft_q_loss(minus, target, actions, 1.2)[0]
            ) / 2e-6
        np.testing.assert_allclose(gradient, numerical, atol=1e-8)

    async def test_local_training_improves_rewarded_token_and_counts(self):
        async def forward(parameters, query, prefix):
            return parameters.copy(), np.eye(2)

        async def evaluate(query, tokens):
            return float(tokens[0] == 1)

        result = await RLPrompt("choose", forward, evaluate).run(
            np.zeros(2),
            ["query"],
            iterations=30,
            prompt_length=1,
            samples_per_query=8,
            learning_rate=0.05,
            seed=3,
        )
        self.assertEqual(result["prompts"], [(1,)])
        self.assertEqual(result["evaluations"], 240)
        self.assertGreater(result["parameters"][1], result["parameters"][0])

    async def test_nonfinite_reward_propagates(self):
        async def forward(parameters, query, prefix):
            return np.zeros(2), np.eye(2)

        async def evaluate(query, tokens):
            return np.nan

        with self.assertRaisesRegex(ValueError, "reward"):
            await RLPrompt("task", forward, evaluate).run(
                np.zeros(2), ["q"], iterations=1, prompt_length=1
            )
