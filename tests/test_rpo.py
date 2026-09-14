"""Robust prompt optimization aggregates normalized model gradients."""

import unittest

import numpy as np

from rpo import RPO


class RPOTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalization_preserves_extreme_finite_gradient_directions(self):
        async def gradient(tokens, context):
            return np.array([[0.0, -1e300], [0.0, -1e-300]])

        agent = RPO("Task", ["context"], gradient, None)
        agent.model_gradient_calls = 0
        with np.errstate(over="ignore", under="ignore"):
            aggregate = await agent._gradient((0, 0))
        np.testing.assert_allclose(aggregate, [[0, -1], [0, -1]])

    async def test_aggregate_objective_and_gradient(self):
        contexts = []

        async def gradient(tokens, context):
            contexts.append(context)
            return np.array([[0.0, -10.0 if context == "one" else -1.0]])

        async def evaluate(tokens, context):
            return float(2 - tokens[0]) + (1 if context == "two" else 0)

        result = await RPO("Robust behavior", ["one", "two"], gradient, evaluate).run(
            [0], iterations=1, top_k=1, batch_size=1
        )
        self.assertEqual(result["best"]["tokens"], (1,))
        self.assertEqual(result["best"]["loss"], 1.5)
        self.assertEqual(result["model_evaluations"], 4)
        self.assertEqual(contexts, ["one", "two"])
