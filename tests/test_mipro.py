"""MIPRO joint categorical selection, teacher acceptance and full-score selection."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mipro import MIPRO, ModulePrompt, Trace
from tests.providers import ScriptedProvider


class MIPROTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "mipro/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_joint_instruction_demo_search_bootstraps_only_successful_traces(self):
        async def execute(program, example):
            return Trace(
                float(example == "good training"),
                ("accepted trace" if example == "good training" else "rejected trace",),
            )

        async def evaluate(program, examples):
            return float(
                program[0].instruction == "better"
                and program[0].demonstrations == ("accepted trace",)
            )

        provider = ScriptedProvider(["dataset summary", "better"])
        result = await MIPRO("Task", provider, evaluate, execute).run(
            [ModulePrompt("module", "seed")],
            ["bad training", "good training"],
            ["v1", "v2"],
            candidate_sets=2,
            instructions_per_module=2,
            bootstrap_attempts=2,
            iterations=15,
            startup_trials=5,
            minibatch_size=1,
            full_eval_interval=3,
            seed=3,
        )
        self.assertEqual(result["best"]["score"], 1)
        self.assertTrue(result["best"]["full"])
        self.assertEqual(result["bootstrap_calls"], 2)
        self.assertEqual(result["optimizer_calls"], 2)
        self.assertIn("accepted trace", provider.calls[1])
        self.assertNotIn("rejected trace", provider.calls[1])
        self.assertIn("dataset summary", provider.calls[1])
        self.assertTrue(any(not t["full"] for t in result["trials"]))

    async def test_multivariate_tpe_preserves_joint_dependencies(self):
        agent = MIPRO("Task", ScriptedProvider([]), None, None)
        agent.nprng = np.random.default_rng(0)
        agent.cardinalities = np.array([2, 2])
        agent.trials = [
            {"configuration": (0, 0), "score": 1},
            {"configuration": (1, 1), "score": 1},
            {"configuration": (0, 1), "score": 0},
            {"configuration": (1, 0), "score": 0},
        ]
        self.assertIn(agent._acquire(0, 100, 0.5, 0.1), {(0, 0), (1, 1)})

    async def test_zero_bootstrapped_demo_budget_allows_successful_teacher_traces(self):
        async def execute(program, example):
            return Trace(1, ("successful trace",))

        async def evaluate(program, examples):
            return 1

        result = await MIPRO("Task", ScriptedProvider(["summary"]), evaluate, execute).run(
            [ModulePrompt("module", "seed")],
            ["training"],
            ["validation"],
            max_bootstrapped_demos=0,
            candidate_sets=2,
            instructions_per_module=1,
            iterations=0,
        )
        self.assertEqual(result["demo_candidates"], [[(), ()]])

    async def test_invalid_teacher_measurement_and_blank_generation(self):
        async def execute(program, example):
            return Trace(float("nan"), ("trace",))

        agent = MIPRO("Task", ScriptedProvider([]), None, execute)
        with self.assertRaisesRegex(ValueError, "teacher"):
            await agent.run([ModulePrompt("module", "seed")], ["training"], ["validation"])
        self.assertEqual(agent.bootstrap_calls, 1)
        with self.assertRaisesRegex(ValueError, "empty"):
            await agent.summarize([], provider=ScriptedProvider([" "]))
