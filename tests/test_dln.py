"""DLN-2 posterior weights, residual forward contexts and layer-wise ELBO selection."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dln import DLN, Example
from tests.providers import ScriptedProvider


class DLNTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "dln/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_posterior_weighting_and_two_layer_prompt_updates(self):
        provider = ScriptedProvider(
            [
                "wrong thought",
                "wrong answer",
                "bad latent",
                "good latent",
                "new output",
                "new hidden",
            ]
        )
        contexts = []

        async def loss(prediction, target):
            return float(prediction != target)

        async def likelihood(context, target):
            contexts.append(context)
            if "First-layer instruction:" in context:
                return (
                    {"bad latent": -3, "good latent": -0.1}
                    if "new hidden" in context
                    else {"bad latent": -2, "good latent": -1}
                )[target]
            good = "good latent" in context
            if "new output" in context:
                return -0.2 if good else -2
            return -1 if good else -4

        result = await DLN("Task", provider, loss, likelihood).run(
            "old hidden",
            "old output",
            [Example("question", "target")],
            iterations=1,
            hidden_samples=2,
            prompt_samples=2,
        )
        self.assertEqual(result["hidden_instruction"], "new hidden")
        self.assertEqual(result["output_instruction"], "new output")
        weights = result["history"][0]["weights"][0]
        np.testing.assert_allclose(weights, np.exp([-6, -2]) / np.exp([-6, -2]).sum())
        self.assertEqual(result["forward_calls"], 2)
        self.assertEqual(result["optimizer_calls"], 4)
        self.assertEqual(result["likelihood_calls"], 12)
        self.assertIn("wrong thought", provider.calls[1])
        self.assertIn("Original input:\nquestion", provider.calls[1])
        self.assertIn("Desired intermediate text: good latent", provider.calls[-1])
        self.assertTrue(all("question" in context for context in contexts))

    async def test_correct_forward_pass_stops_without_optimization(self):
        async def loss(prediction, target):
            return 0

        agent = DLN("Task", ScriptedProvider(["thought", "target"]), loss, None)
        result = await agent.run("hidden", "output", [Example("x", "target")])
        self.assertEqual(result["optimizer_calls"], 0)
        self.assertEqual(result["likelihood_calls"], 0)
        self.assertTrue(result["history"][0]["converged"])

    async def test_empty_posterior_and_nonfinite_likelihood_fail(self):
        async def loss(prediction, target):
            return 1

        async def likelihood(context, target):
            return np.nan

        for posterior, expected in [(" ", "empty"), ("latent", "likelihood")]:
            agent = DLN("Task", ScriptedProvider(["thought", "wrong", posterior]), loss, likelihood)
            with self.assertRaisesRegex(ValueError, expected):
                await agent.run("hidden", "output", [Example("x", "target")], hidden_samples=1)
            self.assertEqual(agent.optimizer_calls, 1)
