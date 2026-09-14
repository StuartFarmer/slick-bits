"""Budgeted best-arm identification must spend on surviving candidates."""

import unittest

from epo import EPO


class EPOTests(unittest.IsolatedAsyncioTestCase):
    async def test_halving_and_continuous_rejects(self):
        for algorithm in ("sequential_halving", "continuous_rejects"):
            seen = []

            async def evaluate(prompt):
                seen.append(prompt)
                return {"bad": 0.0, "good": 1.0, "middle": 0.3}[prompt]

            agent = EPO("Any instruction selection task", evaluate)
            result = await agent.run(["bad", "good", "middle"], budget=30, algorithm=algorithm)
            self.assertEqual(result["best"]["prompt"], "good")
            self.assertEqual(result["evaluations"], 30)
            self.assertEqual(len(seen), 30)
            self.assertGreater(seen.count("good"), seen.count("bad"))
            self.assertEqual(sum(result["pulls"]), 30)

    async def test_short_budget_and_invalid_reward(self):
        async def score(prompt):
            return float(prompt)

        result = await EPO("Numbers", score).run(["0", "1", "2"], budget=2)
        self.assertEqual(result["best"]["prompt"], "1")
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(result["pulls"], [1, 1, 0])

        async def invalid(prompt):
            return float("nan")

        with self.assertRaises(ValueError):
            await EPO("Any", invalid).run(["x"], budget=1)
