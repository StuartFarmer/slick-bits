"""Instruction-coupled covariance, EI, soft-prefix conditioning and measurement boundaries."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from instructzero import Evaluation, InstructZero
from instructzero.agent import matern52
from tests.providers import ScriptedProvider


class InstructZeroTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "instructzero/prompts"
        )
        root.start()
        self.addCleanup(root.stop)

    async def test_condition_fit_ei_and_caching(self):
        provider = ScriptedProvider(["seed low", "seed high", "seed high"])
        prefixes = []

        def condition(provider, prefix):
            prefixes.append(prefix.copy())
            return provider

        async def evaluate(text):
            high = float(text == "seed high")
            return Evaluation(high, np.array([high, high / 2]))

        agent = InstructZero("Explain astronomy.", provider, evaluate, condition)
        result = await agent.run(
            ["demo"], np.array([[2.0], [3.0]]), initial_samples=2, iterations=1, batch_size=1
        )
        self.assertEqual(result["best"]["prompt"], "seed high")
        self.assertEqual(result["optimizer_calls"], 3)
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(result["cache_hits"], 1)
        self.assertEqual(len(result["fits"]), 1)
        self.assertTrue(np.isfinite(result["acquisitions"]).all())
        self.assertTrue((agent.expected_improvement(np.array([[0.1], [0.9]])) >= 0).all())
        for prefix in prefixes:
            self.assertAlmostEqual(prefix[1] / prefix[0], 1.5)
        self.assertIn("Explain astronomy.", provider.calls[0])
        self.assertIn("demo", provider.calls[0])

    async def test_coupled_kernel_uses_behavior_not_just_latent_distance(self):
        agent = InstructZero("Task", ScriptedProvider([]), None, None)
        agent.x = np.array([[0.0], [1.0]])
        parameters = np.log([1.0, 1.0, 1.0, 0.01])
        agent.behavior = np.array([[0.0], [0.0]])
        _, similar = agent._coupling(parameters)
        agent.behavior = np.array([[0.0], [10.0]])
        coupling, distinct = agent._coupling(parameters)
        self.assertGreater(similar[0, 1], distinct[0, 1])
        latent = matern52(agent.x, agent.x, 1)
        np.testing.assert_allclose(distinct, latent @ coupling @ latent.T + 0.01 * np.eye(2))
        self.assertGreater(np.linalg.eigvalsh(distinct).min(), 0)

    async def test_cache_snapshots_reused_evaluator_behavior_buffers(self):
        buffer = np.zeros(1)

        async def evaluate(text):
            buffer[0] = float(text == "b")
            return Evaluation(buffer[0], buffer)

        agent = InstructZero(
            "Task", ScriptedProvider(["a", "b", "a"]), evaluate, lambda provider, prefix: provider
        )
        result = await agent.run([], np.ones((1, 1)), initial_samples=3, iterations=0)
        self.assertEqual([o["behavior"].tolist() for o in result["observations"]], [[0], [1], [0]])
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(result["cache_hits"], 1)

    async def test_invalid_generation_and_behavior_propagate(self):
        async def invalid(text):
            return Evaluation(1, np.array([np.nan]))

        def condition(provider, prefix):
            return provider

        for text, message in [(" ", "empty"), ("valid", "finite")]:
            agent = InstructZero("Task", ScriptedProvider([text]), invalid, condition)
            with self.assertRaisesRegex(ValueError, message):
                await agent.run(["d"], np.ones((1, 1)), initial_samples=2, iterations=0)
            self.assertEqual(agent.optimizer_calls, 1)
