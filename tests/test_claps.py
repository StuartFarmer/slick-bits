import unittest

import numpy as np

from claps import ClaPS


class ClaPSTests(unittest.IsolatedAsyncioTestCase):
    async def test_cluster_prune_and_greedy_restricted_search(self):
        scored = []

        async def probabilities(tokens):
            return (
                np.array([[0.5, 0.5]])
                if not tokens
                else np.array([[0.55, 0.45]])
                if tokens[0] < 2
                else np.array([[0.95, 0.05]])
            )

        async def evaluate(tokens):
            scored.append(tokens)
            return float(sum(tokens))

        result = await ClaPS(
            "Task", np.array([[0.0], [0.01], [10.0], [10.01]]), probabilities, evaluate
        ).run(clusters=2, percentile=50, prompt_length=3, seed=2)
        self.assertEqual(len(result["representatives"]), 2)
        self.assertEqual(len(result["retained"]), 1)
        retained = int(result["retained"][0])
        self.assertGreaterEqual(retained, 2)
        self.assertEqual(result["tokens"], (retained,) * 3)
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["evaluations"], 3)
        self.assertTrue(all(set(tokens) == {retained} for tokens in scored))

    async def test_equal_influence_collapse_does_not_search_removed_tokens(self):
        async def probabilities(tokens):
            return np.array([[0.5, 0.5]])

        async def evaluate(tokens):
            self.fail("strict percentile pruning retained no tokens")

        result = await ClaPS("Task", np.eye(3), probabilities, evaluate).run(clusters=3)
        self.assertEqual(result["tokens"], ())
        self.assertEqual(result["evaluations"], 0)
