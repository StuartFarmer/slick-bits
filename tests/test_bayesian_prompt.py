"""Bayesian n-gram search respects domains and held-out selection."""

import unittest

from bayesian_prompt import BayesianPrompt


class BayesianPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_observations_and_separate_validation(self):
        training, validation = [], []

        async def evaluate(text):
            training.append(text)
            return text.count("a")

        async def validate(text):
            validation.append(text)
            return text.count("b")

        result = await BayesianPrompt("Any task", ["a", "b"], evaluate, validate).run(
            length=2, initial_samples=3, iterations=2, seed=4, restarts=2, raw_samples=8
        )
        self.assertEqual(len(training), 5)
        self.assertEqual(training, validation)
        self.assertTrue(all(set(text.split()) <= {"a", "b"} for text in training))
        self.assertEqual(
            result["best"]["validation_score"], max(text.count("b") for text in training)
        )
