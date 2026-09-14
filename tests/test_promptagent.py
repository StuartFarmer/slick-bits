"""PromptAgent UCT, greedy simulation, suffix backup and success/error operations."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from promptagent import Feedback, PromptAgent
from tests.providers import ScriptedProvider


class PromptAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "promptagent/prompts"
        )
        root.start()
        self.addCleanup(root.stop)

    async def test_uct_rollout_and_suffix_sum_backup(self):
        provider = ScriptedProvider(
            [
                "root feedback",
                "<START>a<END><START>b<END>",
                "b feedback",
                "<START>c<END><START>d<END>",
                "a feedback",
                "<START>e<END><START>f<END>",
            ]
        )
        scores = dict(zip(["root", "a", "b", "c", "d", "e", "f"], range(7)))
        observed = []

        async def evaluate(text):
            return scores[text]

        async def observe(text, examples):
            observed.append(text)
            return Feedback(errors=("Wrong answer",))

        result = await PromptAgent("Explain a recipe.", provider, evaluate, observe).run(
            "root",
            ["training example"],
            iterations=3,
            expand_width=1,
            proposals_per_batch=2,
            depth_limit=2,
            min_depth=9,
            exploration=100,
        )
        self.assertEqual(result["paths"], [[0, 2, 4], [0, 2, 4], [0, 1, 6]])
        self.assertEqual(observed, ["root", "b", "a"])
        self.assertEqual(result["nodes"][0].returns, [6, 6, 7])
        self.assertEqual(result["nodes"][2].returns, [6, 6])
        self.assertEqual(result["nodes"][4].returns, [4, 4])
        self.assertEqual(result["best"].prompt, "f")
        self.assertEqual(result["best_path"], [0, 1, 6])
        self.assertEqual(result["evaluations"], 7)
        self.assertEqual(result["observations"], 3)
        self.assertEqual(result["optimizer_calls"], 6)
        self.assertIn("1. root", provider.calls[3])
        self.assertIn("2. b", provider.calls[3])
        self.assertNotIn("2. a", provider.calls[3])

    async def test_success_branch_threshold_stop_and_weak_child_pruning(self):
        provider = ScriptedProvider(
            ["Preserve the successful rule", "<START>weak<END><START>strong<END>"]
        )

        async def evaluate(text):
            return {"root": 2, "weak": 0, "strong": 3}[text]

        async def observe(text, examples):
            return Feedback(correct=("A correct answer",))

        result = await PromptAgent("Task", provider, evaluate, observe).run(
            "root",
            ["d"],
            iterations=1,
            expand_width=1,
            proposals_per_batch=2,
            depth_limit=5,
            min_depth=0,
        )
        self.assertTrue(result["nodes"][1].terminal)
        self.assertEqual(result["paths"], [[0, 2]])
        self.assertEqual(result["nodes"][0].returns, [5])
        self.assertEqual(result["best"].prompt, "strong")
        self.assertEqual(result["optimizer_calls"], 2)
        self.assertIn("A correct answer", provider.calls[0])
        self.assertIn("Strengths and opportunities", provider.calls[1])

    async def test_malformed_successor_prevents_evaluation(self):
        evaluated = []

        async def evaluate(text):
            evaluated.append(text)
            return 0

        async def observe(text, examples):
            return Feedback(errors=("error",))

        agent = PromptAgent(
            "Task", ScriptedProvider(["feedback", "<START> <END>"]), evaluate, observe
        )
        with self.assertRaisesRegex(ValueError, "response="):
            await agent.run("seed", ["d"], iterations=1, expand_width=1)
        self.assertEqual(evaluated, ["seed"])
        self.assertEqual(agent.optimizer_calls, 2)

    async def test_invalid_measurements_and_observer_errors_propagate(self):
        async def invalid(text):
            return math.nan

        async def observe(text, examples):
            raise RuntimeError("observer failed")

        with self.assertRaisesRegex(ValueError, "finite"):
            await PromptAgent("Task", ScriptedProvider([]), invalid, observe).run("seed", ["d"])

        async def evaluate(text):
            return 1

        agent = PromptAgent("Task", ScriptedProvider([]), evaluate, observe)
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            await agent.run("seed", ["d"])
        self.assertEqual(agent.observations, 1)
        self.assertEqual(agent.optimizer_calls, 0)
