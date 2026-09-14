"""APE instruction induction, deduplication and evaluation allocation."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from ape import APE
from tests.providers import ScriptedProvider


class APETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "ape/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_ucb_explores_then_exploits_and_deduplicates(self):
        provider = ScriptedProvider(["short", "long instruction", "short"])
        evaluated = []

        async def score(text):
            evaluated.append(text)
            return float(text == "long instruction")

        result = await APE("Teach gardening.", provider, score).run(
            ["input -> output"],
            subsamples=1,
            demos_per_sample=1,
            prompts_per_sample=3,
            rounds=5,
            prompts_per_round=1,
            samples_per_eval=5,
            exploration=0.1,
            seed=1,
        )
        self.assertEqual(len(result["population"]), 2)
        self.assertEqual(set(evaluated[:2]), {"short", "long instruction"})
        self.assertEqual(evaluated[2:], ["long instruction"] * 3)
        self.assertEqual(result["best"]["samples"], 20)
        self.assertEqual(result["evaluations"], 5)
        self.assertEqual(result["optimizer_calls"], 3)
        self.assertTrue(all("Teach gardening." in c for c in provider.calls))
        self.assertTrue(all("input -> output" in c for c in provider.calls))

    async def test_sample_means_and_unevaluated_candidates(self):
        values = iter([-1.0, -3.0])

        async def score(text):
            return next(values)

        result = await APE("Task", ScriptedProvider(["a"]), score).run(
            ["demo"],
            subsamples=1,
            demos_per_sample=1,
            prompts_per_sample=1,
            rounds=2,
            prompts_per_round=1,
        )
        self.assertEqual(result["best"]["score"], -2)
        result = await APE("Task", ScriptedProvider(["a"]), score).run(
            ["demo"],
            subsamples=1,
            demos_per_sample=1,
            prompts_per_sample=1,
            rounds=0,
        )
        self.assertIsNone(result["best"])
        self.assertIsNone(result["population"][0]["score"])

    async def test_generation_and_measured_failures_propagate(self):
        async def invalid(text):
            return math.nan

        for response in (" ", RuntimeError("transport")):
            agent = APE("Task", ScriptedProvider([response]), invalid)
            with self.assertRaises((ValueError, RuntimeError)):
                await agent.run(["d"], subsamples=1, demos_per_sample=1, prompts_per_sample=1)
            self.assertEqual(agent.optimizer_calls, 1)
        agent = APE("Task", ScriptedProvider(["valid"]), invalid)
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run(["d"], subsamples=1, demos_per_sample=1, prompts_per_sample=1)
        self.assertEqual(agent.evaluations, 1)
