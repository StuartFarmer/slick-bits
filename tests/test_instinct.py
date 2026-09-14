"""INSTINCT's gradient confidence updates, reset-refit training and true prefix boundary."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from instinct import INSTINCT
from tests.providers import ScriptedProvider


def linear_surrogate(theta, contexts):
    jacobian = np.column_stack((contexts[:, 0], np.ones(len(contexts))))
    return jacobian @ theta, jacobian


class INSTINCTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "instinct/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_neural_ucb_updates_diagonal_then_refits_all_observations(self):
        provider = ScriptedProvider(["seed one", "seed two", "best result"])
        prefixes = []

        def condition(provider, prefix):
            prefixes.append(prefix.copy())
            return provider

        async def hidden(prefixes):
            return prefixes.copy()

        async def evaluate(text):
            return len(text)

        agent = INSTINCT("Task", provider, evaluate, condition, hidden, linear_surrogate)
        result = await agent.run(
            ["demo"],
            np.array([[2.0]]),
            np.zeros(2),
            domain_size=4,
            initial_samples=2,
            iterations=1,
            training_steps=5,
        )
        expected_arm = int(np.argmax(agent.contexts[:, 0]))
        self.assertEqual(result["acquisitions"][0]["chosen"], expected_arm)
        expected_gradient = np.array([agent.contexts[expected_arm, 0], 1])
        np.testing.assert_allclose(result["u"], 1 + expected_gradient**2)
        self.assertFalse(np.allclose(result["theta"], 0))
        self.assertEqual(result["training_runs"], 1)
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(result["optimizer_calls"], 3)
        self.assertEqual(len(prefixes), 3)
        np.testing.assert_allclose(
            agent.mean, np.mean(agent.contexts[[r["arm"] for r in result["records"]]], axis=0)
        )
        self.assertIn("demo", provider.calls[0])

    async def test_duplicate_instructions_skip_evaluation_and_retrain_is_repeatable(self):
        async def evaluate(text):
            return 1

        async def hidden(prefixes):
            return prefixes

        agent = INSTINCT(
            "Task",
            ScriptedProvider(["same"] * 3),
            evaluate,
            lambda provider, prefix: provider,
            hidden,
            linear_surrogate,
        )
        result = await agent.run(
            [],
            np.ones((1, 1)),
            np.zeros(2),
            domain_size=4,
            initial_samples=2,
            iterations=1,
            training_steps=2,
        )
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["cache_hits"], 2)
        theta = result["theta"].copy()
        agent._refit(2, 1e-3, 1)
        np.testing.assert_allclose(agent.theta, theta)

    async def test_nonfinite_hidden_features_and_generated_text_fail(self):
        async def hidden(prefixes):
            return np.full_like(prefixes, np.nan)

        agent = INSTINCT(
            "Task",
            ScriptedProvider([]),
            None,
            lambda provider, prefix: provider,
            hidden,
            linear_surrogate,
        )
        with self.assertRaisesRegex(ValueError, "hidden-state"):
            await agent.run([], np.ones((1, 1)), np.zeros(2), domain_size=2, initial_samples=1)
        self.assertEqual(agent.optimizer_calls, 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            await agent.induce([], provider=ScriptedProvider([" "]))
