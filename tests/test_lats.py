"""Check LATS search and Slick boundaries without paid model calls."""

import json
import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from lats import LATS, Feedback, Node
from tests.providers import ScriptedProvider


def action(content, thought="Plan"):
    return json.dumps({"thought": thought, "content": content})


class LATSTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "lats/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_rollout_backtracks_and_uses_failure_memory(self):
        paths = []

        async def evaluate(trajectory, content):
            path = tuple(step.action.content for step in trajectory) + (content,)
            paths.append(path)
            return Feedback(
                observation="observed " + content,
                reward=float(content == "win"),
                terminal=len(path) == 2,
                success=content == "win",
            )

        provider = ScriptedProvider(
            [
                action("A"),
                action("B"),
                '{"score": 0.9}',
                '{"score": 0.1}',
                action("lose"),
                action("lose"),
                "Avoid A's dead end.",
                action("win"),
                action("win"),
            ]
        )
        agent = LATS("Find a feasible route", provider, evaluate)
        result = await agent.run(iterations=4, candidates=2, depth=3)
        self.assertEqual(result.stop_reason, "success")
        self.assertEqual(paths, [("A",), ("B",), ("A", "lose"), ("B", "win")])
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.expansions, 3)
        self.assertEqual(agent.root.visits, 2)
        self.assertEqual(agent.root.value, 0.5)
        self.assertEqual(agent.root.children[0].value, 0)
        self.assertEqual(agent.root.children[1].value, 1)
        self.assertIn("Avoid A's dead end.", provider.calls[7])
        self.assertIn("observed A", provider.calls[2])
        self.assertEqual(len(agent.memory), 1)

    async def test_self_consistency_counts_actions_before_deduplication(self):
        async def evaluate(trajectory, content):
            return Feedback("available")

        provider = ScriptedProvider(
            [
                action("A", "first"),
                action("A", "second"),
                action("B"),
                '{"score": 0.2}',
                '{"score": 0.4}',
                "Need another step.",
            ]
        )
        agent = LATS("Task", provider, evaluate)
        result = await agent.run(iterations=1, candidates=3, depth=1, lm_weight=0.5)
        a, b = agent.root.children
        self.assertAlmostEqual(a.heuristic, 0.5 * 0.2 + 0.5 * 2 / 3)
        self.assertAlmostEqual(b.heuristic, 0.5 * 0.4 + 0.5 * 1 / 3)
        self.assertEqual((a.visits, b.visits), (1, 0))
        self.assertEqual(result.stop_reason, "budget")
        self.assertFalse(result.best.feedback.success)
        self.assertEqual(len(agent.assessments), 2)
        self.assertEqual(provider.calls[0], provider.calls[1])
        self.assertEqual(provider.calls[1], provider.calls[2])

    async def test_successful_sibling_stops_before_later_environment_calls(self):
        async def evaluate(trajectory, content):
            return Feedback(content, reward=0.1, terminal=True, success=content == "win")

        agent = LATS(
            "Task", ScriptedProvider([action("fail"), action("win"), action("skip")]), evaluate
        )
        result = await agent.run(candidates=3)
        self.assertEqual(result.best.trajectory[-1].action.content, "win")
        self.assertEqual(len(agent.assessments), 2)
        self.assertEqual(agent.root.visits, 1)
        self.assertAlmostEqual(agent.root.value, 0.1)

    async def test_no_rollout_refines_complete_candidates_with_measured_rewards(self):
        async def evaluate(trajectory, content):
            return Feedback("feedback " + content, reward={"A": 0.2, "B": 0.4, "C": 0.7}[content])

        provider = ScriptedProvider(
            [
                action("A"),
                action("B"),
                '{"score": 0.8}',
                '{"score": 0.1}',
                "Improve A",
                "Improve B",
                action("C"),
                action("C"),
                '{"score": 0.5}',
                "Improve C",
            ]
        )
        agent = LATS("Draft any artifact", provider, evaluate)
        result = await agent.run(iterations=2, candidates=2, simulate=False)
        self.assertEqual(result.best.trajectory[-1].action.content, "C")
        self.assertEqual(result.expansions, 2)
        self.assertEqual(agent.root.visits, 3)
        self.assertAlmostEqual(agent.root.value, 1.3 / 3)
        self.assertIn("Improve A", provider.calls[6])
        self.assertIn("Improve B", provider.calls[8])
        self.assertEqual(len(agent.memory), 3)

    async def test_exhaustion_and_zero_budgets_terminate(self):
        async def evaluate(trajectory, content):
            return Feedback("failed", reward=-1, terminal=True)

        agent = LATS(
            "Task", ScriptedProvider([action("A"), action("B"), "A failed", "B failed"]), evaluate
        )
        result = await agent.run(iterations=20, candidates=2)
        self.assertEqual(result.stop_reason, "exhausted")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.best.feedback.reward, -1)
        self.assertEqual(agent.root.visits, 2)
        self.assertEqual(len(agent.root.children), 2)
        for options in ({"iterations": 0}, {"depth": 0}):
            result = await agent.run(**options)
            self.assertIsNone(result.best)
            self.assertEqual(agent.calls, [])
            self.assertEqual(agent.memory, [])

    async def test_uct_explores_unvisited_and_does_not_divide_mean_twice(self):
        parent = Node(visits=10)
        child = Node(parent=parent, value=0.6, visits=2)
        self.assertAlmostEqual(child.uct(1), 0.6 + math.sqrt(math.log(10) / 2))
        self.assertEqual(Node(parent=parent).uct(1), math.inf)

    async def test_depth_cutoff_uses_configured_return_without_claiming_success(self):
        async def evaluate(trajectory, content):
            return Feedback("partial progress", reward=0.9)

        provider = ScriptedProvider([action("step"), "The trial needs more time."])
        agent = LATS("Task", provider, evaluate)
        result = await agent.run(candidates=1, depth=1, lm_weight=0, cutoff_reward=-0.25)
        self.assertEqual(result.stop_reason, "exhausted")
        self.assertEqual(agent.root.value, -0.25)
        self.assertEqual(result.best.feedback.reward, 0.9)
        self.assertFalse(result.best.feedback.success)
        self.assertEqual(agent.memory[0].reason, "depth limit")
        self.assertEqual([call["operation"] for call in agent.calls], ["sample", "reflect"])

    async def test_provider_and_reflection_failures_preserve_attempts(self):
        async def evaluate(trajectory, content):
            return Feedback("failed", terminal=True)

        agent = LATS("Task", ScriptedProvider([OSError("transport")]), evaluate)
        with self.assertRaisesRegex(OSError, "transport"):
            await agent.run(candidates=1)
        self.assertIn("error", agent.calls[0])
        self.assertEqual(agent.expansions, 1)
        agent = LATS("Task", ScriptedProvider([action("A"), " "]), evaluate)
        with self.assertRaisesRegex(ValueError, "reflection is blank"):
            await agent.run(candidates=1)
        self.assertEqual(agent.calls[-1]["response"], " ")
        self.assertIn("error", agent.calls[-1])
        self.assertEqual(agent.root.visits, 1)

    async def test_generated_and_environment_failures_are_recorded_and_propagated(self):
        async def evaluate(trajectory, content):
            return Feedback("obs")

        for bad in ("not json", action(" "), '{"thought":"x","content":"A","extra":1}'):
            agent = LATS("Task", ScriptedProvider([bad]), evaluate)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await agent.run(candidates=1)
            self.assertEqual(agent.calls[0]["response"], bad)
            self.assertIn("error", agent.calls[0])
            self.assertEqual(agent.expansions, 1)
            self.assertEqual(agent.assessments, [])
        for bad in ('{"score": NaN}', '{"score": 1.1}'):
            agent = LATS("Task", ScriptedProvider([action("A"), bad]), evaluate)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await agent.run(candidates=1)
            self.assertEqual(agent.calls[-1]["response"], bad)

        async def broken(trajectory, content):
            raise OSError("environment unavailable")

        async def nonfinite(trajectory, content):
            return Feedback("obs", reward=float("nan"))

        for callback, error in ((broken, OSError), (nonfinite, ValueError)):
            agent = LATS("Task", ScriptedProvider([action("A")]), callback)
            with self.assertRaises(error):
                await agent.run(candidates=1)
            self.assertIn("error", agent.assessments[0])

    async def test_templates_render_with_explicit_owner_from_another_directory(self):
        agent = LATS("TASK CONTEXT", ScriptedProvider([]), None)
        previous = Path.cwd()
        try:
            os.chdir(self.templates)
            for operation, args in (
                (LATS.sample, (Node(),)),
                (LATS.value, (Node(),)),
                (LATS.reflect, (Node(), 0.0, "depth limit")),
            ):
                rendered = await operation.render(agent, *args)
                self.assertIn("TASK CONTEXT", rendered)
            for template in self.templates.glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
