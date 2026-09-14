"""Check bilateral inference, weighted boosting, and informative-learner handling."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from prefer import PREFER, Example, Learner
from tests.providers import ScriptedProvider


class PREFERTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "prefer/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_bilateral_changes_forward_winner_and_boosts_errors(self):
        async def evaluate(instruction, input, direction):
            if direction == "forward":
                return (0.9, 0.8)
            # Bilateral prediction is b for first two items, a for the third.
            return (0.8, 0.0) if input != "third" else (0.0, 0.8)

        provider = ScriptedProvider(['{"text":"reason"}', '{"text":"revised"}'])
        agent = PREFER("Task", provider, evaluate, ["a", "b"])
        result = await agent.run(
            "seed", [Example("first", "b"), Example("second", "b"), Example("third", "b")], rounds=2
        )
        self.assertEqual(len(result["ensemble"]), 1)
        self.assertAlmostEqual(result["ensemble"][0].weight, math.log(2))
        for actual, expected in zip(result["instance_weights"], [0.25, 0.25, 0.5], strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertIn("third", provider.calls[0])
        self.assertIn("reason", provider.calls[1])
        self.assertEqual(await agent.predict(result["ensemble"], "first"), "b")

    async def test_weighted_ensemble_and_perfect_terminal(self):
        async def evaluate(instruction, input, direction):
            if direction == "backward":
                return (0, 0)
            return (1, 0) if instruction == "a" else (0, 1)

        agent = PREFER("Task", ScriptedProvider([]), evaluate, ["a", "b"])
        self.assertEqual(await agent.predict([Learner("a", 1), Learner("b", 2)], "x"), "b")
        result = await agent.run("a", [Example("x", "a")])
        self.assertEqual(result["ensemble"], [Learner("a", 1)])

    async def test_invalid_confidence_and_generated_errors(self):
        async def evaluate(instruction, input, direction):
            return (float("nan"), 1)

        agent = PREFER("Task", ScriptedProvider([]), evaluate, ["a", "b"])
        with self.assertRaises(ValueError):
            await agent.run("seed", [Example("x", "a")])
        with self.assertRaises(RuntimeError):
            await agent.reflect("x", [], provider=ScriptedProvider([RuntimeError("provider")]))
