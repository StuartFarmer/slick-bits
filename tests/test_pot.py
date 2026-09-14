"""Offline checks of PoT transitions, search, budgets, and Slick boundaries."""

import asyncio
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from pot import PlanOfThoughts, State, Thought
from tests.providers import ScriptedProvider


class PoTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "pot/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_rollout_returns_best_complete_artifact_for_arbitrary_tasks(self):
        for task in ("Plan an exhibition", "Design a sorting algorithm"):
            provider = ScriptedProvider(
                [
                    '{"content":"First step","terminal":false}',
                    '{"judgment":"likely"}',
                    '{"content":"Complete artifact","terminal":true}',
                    '{"judgment":"sure"}',
                    '{"probability":0.9}',
                ]
            )
            agent = PlanOfThoughts(task, provider)
            result = await agent.run(depth=3, max_calls=5, success_threshold=0.8)
            self.assertEqual(result.best.trajectory, ("First step", "Complete artifact"))
            self.assertEqual(result.best.answer, "Complete artifact")
            self.assertTrue(result.solved)
            self.assertEqual(result.calls, 5)
            self.assertTrue(all(task in context for context in provider.calls))

    async def test_action_semantics_preserve_committed_prefix(self):
        provider = ScriptedProvider(
            [
                '{"content":"replacement","terminal":false}',
                '{"judgment":"sure"}',
                '{"content":"next","terminal":false}',
                '{"judgment":"likely"}',
                '{"judgment":"impossible"}',
            ]
        )
        agent = PlanOfThoughts("Task", provider)
        await agent.run(max_calls=0)
        agent.max_calls = 10
        state = State(("earlier",), Thought(content="pending"))
        replaced, _ = await agent._transition(state, "think")
        advanced, _ = await agent._transition(state, "continue")
        reverted, judgment = await agent._transition(state, "rollback")
        self.assertEqual(replaced.prefix, ("earlier",))
        self.assertEqual(replaced.current.content, "replacement")
        self.assertEqual(advanced.prefix, ("earlier", "pending"))
        self.assertEqual(advanced.current.content, "next")
        self.assertEqual(reverted.prefix, ())
        self.assertEqual(reverted.current.content, "earlier")
        self.assertEqual(judgment, "impossible")
        self.assertEqual(state.current.content, "pending")

    async def test_search_explores_alternative_and_backs_up_expected_reward(self):
        provider = ScriptedProvider(
            [
                '{"content":"bad","terminal":false}',
                '{"judgment":"likely"}',
                '{"content":"bad answer","terminal":true}',
                '{"judgment":"impossible"}',
                '{"probability":0.1}',
                '{"content":"good answer","terminal":true}',
                '{"judgment":"sure"}',
                '{"probability":0.8}',
            ]
        )
        agent = PlanOfThoughts("Task", provider)
        result = await agent.run(
            depth=2, simulations=2, max_calls=8, success_threshold=0.8, reward_min=-1, reward_max=1
        )
        self.assertEqual(result.best.answer, "good answer")
        self.assertAlmostEqual(result.best.reward, 0.6)
        self.assertEqual(agent.root.actions["continue"].visits, 1)
        self.assertEqual(agent.root.actions["think"].visits, 1)
        self.assertAlmostEqual(agent.root.actions["continue"].value, -0.8)
        self.assertAlmostEqual(agent.root.actions["think"].value, 0.6)

    async def test_zero_budget_and_interrupted_search_preserve_incumbent(self):
        agent = PlanOfThoughts("Task", ScriptedProvider([]))
        result = await agent.run(max_calls=0)
        self.assertIsNone(result.best)
        self.assertEqual(result.reason, "calls")
        self.assertEqual(result.calls, 0)
        provider = ScriptedProvider(
            [
                '{"content":"partial","terminal":false}',
                '{"judgment":"likely"}',
                '{"content":"answer","terminal":true}',
                '{"judgment":"likely"}',
                '{"probability":0.4}',
            ]
        )
        agent = PlanOfThoughts("Task", provider)
        result = await agent.run(depth=2, max_calls=5)
        self.assertEqual(result.best.answer, "answer")
        self.assertFalse(result.solved)
        self.assertEqual(result.reason, "calls")

    async def test_uct_revisits_history_and_reuses_subtree_after_execution(self):
        provider = ScriptedProvider(
            [
                '{"content":"A","terminal":false}',
                '{"judgment":"likely"}',
                '{"content":"B","terminal":false}',
                '{"judgment":"likely"}',
                '{"content":"C","terminal":true}',
                '{"probability":0.8}',
                '{"content":"X","terminal":true}',
                '{"judgment":"likely"}',
                '{"probability":0.1}',
                '{"content":"B","terminal":false}',
                '{"judgment":"likely"}',
                '{"content":"C","terminal":true}',
                '{"judgment":"sure"}',
                '{"probability":0.7}',
                '{"content":"B","terminal":false}',
                '{"judgment":"likely"}',
            ]
        )
        agent = PlanOfThoughts("Task", provider)
        result = await agent.run(depth=3, simulations=3, max_actions=1, success_threshold=1.0)
        self.assertEqual(result.state.trajectory, ("A", "B"))
        self.assertEqual(result.best.probability, 0.8)
        self.assertEqual(result.simulations, 3)
        self.assertEqual(result.reason, "actions")
        self.assertEqual(agent.history[0]["action"], "continue")
        self.assertEqual(agent.root.visits, 1)
        self.assertAlmostEqual(agent.root.actions["continue"].value, 0.7)

    async def test_complete_solution_is_not_hidden_by_high_scoring_partial(self):
        provider = ScriptedProvider(
            [
                '{"content":"unfinished","terminal":false}',
                '{"judgment":"likely"}',
                '{"probability":0.99}',
                '{"content":"finished","terminal":true}',
                '{"judgment":"sure"}',
                '{"probability":0.95}',
            ]
        )
        agent = PlanOfThoughts("Task", provider)
        result = await agent.run(depth=1, max_calls=6)
        self.assertTrue(result.solved)
        self.assertEqual(result.best.answer, "finished")

    async def test_custom_evaluation_and_separate_generation_providers(self):
        generator = ScriptedProvider(
            [
                '{"content":"start","terminal":false}',
                '{"content":"middle","terminal":false}',
            ]
        )
        judge = ScriptedProvider(['{"judgment":"likely"}', '{"judgment":"likely"}'])
        greedy = ScriptedProvider(['{"content":"artifact","terminal":true}'])
        assessed = []

        async def evaluate(trajectory):
            assessed.append(trajectory)
            return 1.0

        agent = PlanOfThoughts(
            "Task", generator, evaluate, judge_provider=judge, rollout_provider=greedy
        )
        result = await agent.run(depth=3)
        self.assertTrue(result.solved)
        self.assertEqual(assessed, [("start", "middle", "artifact")])
        self.assertEqual(result.calls, 6)

    async def test_generated_and_evaluator_errors_propagate_with_records(self):
        for bad in (
            "not json",
            '{"content":" ","terminal":false}',
            '{"content":"x","terminal":"false"}',
        ):
            agent = PlanOfThoughts("Task", ScriptedProvider([bad]))
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await agent.run()
            self.assertEqual(agent.calls[0]["response"], bad)
            self.assertIn("error", agent.calls[0])
        for probability in (float("nan"), 1.1, -0.1):

            async def evaluate(trajectory):
                return probability

            agent = PlanOfThoughts(
                "Task",
                ScriptedProvider(
                    [
                        '{"content":"artifact","terminal":true}',
                        '{"judgment":"sure"}',
                    ]
                ),
                evaluate,
            )
            with self.assertRaisesRegex(ValueError, "probability"):
                await agent.run()
            self.assertIn("error", agent.calls[-1])
        agent = PlanOfThoughts("Task", ScriptedProvider([TimeoutError("provider")]))
        with self.assertRaisesRegex(TimeoutError, "provider"):
            await agent.run()

    async def test_deadline_cancels_inflight_evaluation(self):
        cancelled = []

        async def evaluate(trajectory):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)

        agent = PlanOfThoughts(
            "Task",
            ScriptedProvider(
                [
                    '{"content":"artifact","terminal":true}',
                    '{"judgment":"sure"}',
                ]
            ),
            evaluate,
        )
        result = await agent.run(time_limit=0.1)
        self.assertEqual(result.reason, "time")
        self.assertEqual(cancelled, [True])

    async def test_templates_render_from_another_directory(self):
        agent = PlanOfThoughts("Task", ScriptedProvider([]))
        previous = Path.cwd()
        try:
            os.chdir(self.templates)
            for operation, args in (
                (PlanOfThoughts.think, ((), "")),
                (PlanOfThoughts.rollout_step, (("a",),)),
                (PlanOfThoughts.observe, (("a",),)),
                (PlanOfThoughts.judge, (("a",),)),
            ):
                rendered = await operation.render(agent, *args)
                self.assertIn("Task", rendered)
                self.assertIn('"properties"', rendered)
            for template in self.templates.glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)
