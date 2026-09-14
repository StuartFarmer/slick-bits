"""DPO-Diff differentiates categorical mixtures and ranks discrete samples."""

import unittest

import numpy as np

from dpo_diff import DPODiff


class DPODiffTests(unittest.IsolatedAsyncioTestCase):
    async def test_categorical_gradient_and_bounded_sampling(self):
        async def gradient(mixed, timestep):
            return [np.array([-1.0])]

        async def evaluate(parts):
            return 0.0 if parts == ("better",) else 1.0

        agent = DPODiff("Any differentiable objective", gradient, evaluate)
        result = await agent.run(
            [["initial", "better"]],
            [np.array([[0.0], [1.0]])],
            iterations=10,
            samples=2,
            max_sample_attempts=100,
            seed=4,
        )
        self.assertGreater(result["logits"][0][1], result["logits"][0][0])
        self.assertEqual(result["best"]["parts"], ("better",))
        self.assertEqual(result["gradient_calls"], 10)
        self.assertEqual(result["evaluations"], 2)

    async def test_finite_domain_terminates(self):
        async def gradient(mixed, timestep):
            return [np.array([0.0])]

        async def evaluate(parts):
            return 1.0

        result = await DPODiff("Any", gradient, evaluate).run(
            [["only"]], [np.array([[0.0]])], iterations=0, samples=5, max_sample_attempts=3
        )
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["sample_attempts"], 3)
