import unittest
from pathlib import Path

import numpy as np
from scipy.special import log_softmax, softmax
from slick import prompts

from retroformer import Episode, Retroformer, preference_loss, sequence_ppo_loss


class RetroformerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.old_root = prompts.TEMPLATE_ROOT
        prompts.TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "retroformer" / "prompts"

    def tearDown(self):
        prompts.TEMPLATE_ROOT = self.old_root

    def test_reward_and_ppo_derivatives(self):
        _, derivative, _ = preference_loss(0.2, -0.3)
        numeric = (preference_loss(0.200001, -0.3)[0] - preference_loss(0.199999, -0.3)[0]) / 2e-6
        self.assertAlmostEqual(derivative, numeric, places=8)
        args = (-0.7, 0.1, 0.5, 0.6, 0.2, 0.5)
        _, dl, dv = sequence_ppo_loss(-0.6, 0.2, *args)
        expected = (
            sequence_ppo_loss(-0.599999, 0.2, *args)[0]
            - sequence_ppo_loss(-0.600001, 0.2, *args)[0]
        ) / 2e-6
        self.assertAlmostEqual(dl, expected, places=8)
        expected = (
            sequence_ppo_loss(-0.6, 0.200001, *args)[0]
            - sequence_ppo_loss(-0.6, 0.199999, *args)[0]
        ) / 2e-6
        self.assertAlmostEqual(dv, expected, places=8)

    async def test_return_deltas_train_reward_then_best_of_reflection(self):
        generated = 0
        environments = []

        async def rollout(query, reflection):
            environments.append(reflection)
            score = {"": 0.2, "good": 0.9, "bad": 0.1}[reflection]
            return Episode("attempt trace", score, reflection == "good")

        async def generate(parameters, context, seed):
            nonlocal generated
            generated += 1
            return "good" if generated % 2 else "bad"

        async def policy(parameters, context, text):
            index = 0 if text == "good" else 1
            gradient = np.zeros(3)
            gradient[:2] = -softmax(parameters[:2])
            gradient[index] += 1
            return (
                float(log_softmax(parameters[:2])[index]),
                gradient,
                parameters[2],
                np.array([0.0, 0.0, 1.0]),
            )

        async def reward(parameters, context, text):
            feature = np.array([1.0 if text == "good" else -1.0])
            return float(parameters @ feature), feature

        agent = Retroformer("task", rollout, generate, policy, reward)
        result = await agent.run(
            np.zeros(3),
            np.zeros(1),
            ["q"],
            trials=1,
            reward_rate=0.1,
            sft_rate=0.1,
            policy_rate=0.1,
            ppo_iterations=1,
        )
        self.assertEqual(result["environment_calls"], 3)
        np.testing.assert_allclose([r.rating for r in result["ratings"]], [0.7, -0.1])
        self.assertGreater(result["reward_parameters"][0], 0)
        self.assertEqual(len(result["policy_losses"]), 4)
        improved = await agent.improve("new q", trials=1, best_of=2)
        self.assertTrue(improved["episode"].success)
        self.assertEqual(improved["reflections"], ["good"])
