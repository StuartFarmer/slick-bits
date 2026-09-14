"""Likelihood-free sampling returns a validated ensemble, not a partial population."""

import unittest

import numpy as np

from sbi_prompt import SBIPrompt


class SBIPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_accuracy_schedule_includes_perfect_stage(self):
        calls = 0

        async def evaluate(theta):
            nonlocal calls
            calls += 1
            return 0.0 if calls == 1 else 1.0

        async def validate(particles, weights):
            return 1.0

        result = await SBIPrompt("Task", evaluate, validate).run(
            dimension=1, training_size=9, initial_samples=1, particles=1
        )
        self.assertEqual(result["validations"], 10)
        self.assertEqual(result["history"][-1]["threshold"], 1.0)

    async def test_tolerance_and_completed_ensemble(self):
        async def evaluate(theta):
            return 0.0

        validated = []

        async def validate(particles, weights):
            validated.append((particles.copy(), weights.copy()))
            return 0.4

        result = await SBIPrompt("Any soft prompt task", evaluate, validate).run(
            dimension=2,
            particles=3,
            training_size=2,
            initial_samples=2,
            max_attempts_per_stage=5,
            seed=2,
        )
        self.assertEqual(result["evaluations"], 10)
        self.assertEqual(len(validated), 1)
        self.assertEqual(result["particles"].shape, (3, 2))
        np.testing.assert_allclose(result["weights"], [1 / 3] * 3)
        self.assertEqual(result["history"][0]["threshold"], 0)
        self.assertEqual(result["stopped"], "stage_budget")
