import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from active_example_selection import ActiveExampleSelection
from active_example_selection.agent import td_loss
from tests.providers import ScriptedProvider


class ActiveSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT",
            Path(__file__).parents[1] / "active_example_selection/prompts",
        )
        root.start()
        self.addCleanup(root.stop)

    def test_l1_and_conservative_gradients_match_finite_difference(self):
        values = np.array([0.2, -0.3, 0.8])
        _, gradient = td_loss(values, 1, 1.0, 0.7)
        for i in range(len(values)):
            direction = np.eye(3)[i] * 1e-6
            numerical = (
                td_loss(values + direction, 1, 1.0, 0.7)[0]
                - td_loss(values - direction, 1, 1.0, 0.7)[0]
            ) / 2e-6
            self.assertAlmostEqual(gradient[i], numerical, places=7)

    async def test_replay_updates_parameters_and_rewards_telescope(self):
        async def evaluate(demos):
            return len(demos) * 0.2

        async def represent(demos, available):
            return np.array([len(demos)]), np.array([[1.0] for _ in available])

        async def q_model(theta, states, actions):
            features = np.concatenate([actions[:, 0], [-1.0]])
            return theta[0] * features, features[:, None]

        provider = ScriptedProvider(["answer"])
        agent = ActiveExampleSelection("task", provider, evaluate, represent, q_model)
        result = await agent.run(
            list("abcd"),
            np.array([0.4]),
            episodes=12,
            max_examples=3,
            epsilon_start=0.001,
            epsilon_end=0.001,
            batch_size=4,
            cost_per_example=0.05,
            target_update_every=2,
            seed=1,
        )
        self.assertGreater(result["updates"], 0)
        self.assertNotEqual(result["parameters"][0], 0.4)
        self.assertEqual(result["indices"], [0, 1, 2])
        self.assertAlmostEqual(sum(result["rewards"]), 0.45)
        await agent.answer("q", provider=provider)
        self.assertIn("a", provider.calls[0])

    async def test_learned_stop_does_not_evaluate_a_demonstration(self):
        async def evaluate(demos):
            self.assertEqual(demos, ())
            return 0.3

        async def represent(demos, available):
            return np.zeros(1), np.zeros((len(available), 1))

        async def q_model(theta, states, actions):
            return np.r_[np.zeros(len(actions)), 1.0], np.zeros((len(actions) + 1, len(theta)))

        result = await ActiveExampleSelection("t", None, evaluate, represent, q_model).run(
            ["a"], np.zeros(1), episodes=0
        )
        self.assertEqual(result["indices"], [])
        self.assertEqual(result["evaluations"], 1)

    async def test_nonfinite_q_values_fail_visibly(self):
        async def evaluate(demos):
            return 0.0

        async def represent(demos, available):
            return np.zeros(1), np.zeros((1, 1))

        async def q_model(theta, states, actions):
            return np.array([float("nan"), 0.0]), np.zeros((2, 1))

        with self.assertRaisesRegex(ValueError, "finite"):
            await ActiveExampleSelection("t", None, evaluate, represent, q_model).run(
                ["a"], np.zeros(1), episodes=0
            )
