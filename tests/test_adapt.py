"""Check ADaPT control flow through real Slick prompts with scripted generation."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from adapt import ADAPT, Step, parse_plan
from tests.providers import ScriptedProvider


async def evaluate(history, action):
    return "Observed: " + action


class AdaptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "adapt/prompts"
        self.root = patch("slick.prompts.TEMPLATE_ROOT", root)
        self.root.start()
        self.addCleanup(self.root.stop)

    async def test_recursive_and_or_and_successful_checkpoint_handoff(self):
        executor = ScriptedProvider(
            [
                "action: discarded root action",
                "think: Task failed!",
                "think: Task failed!",  # Find needs another level.
                "action: discarded alternative",
                "think: Task failed!",
                "action: obtain",
                "think: Task completed!",
                "action: finish",
                "think: Task completed!",
            ]
        )
        planner = ScriptedProvider(
            [
                "Step 1: Find input\nStep 2: Finish output\nExecution Order: (Step 1 AND Step 2)",
                "Step 1: Try A\nStep 2: Try B\nStep 3: Never\n"
                "Execution Order: (Step 1 OR Step 2 OR Step 3)",
            ]
        )
        evaluated = []

        async def record(history, action):
            evaluated.append((history, action))
            return await evaluate(history, action)

        agent = ADAPT("Produce output", executor, record, planner_provider=planner)
        result = await agent.run()
        self.assertTrue(result.completed)
        self.assertEqual([s.action for s in result.steps], ["obtain", "finish"])
        self.assertEqual([e.depth for e in result.executions], [1, 2, 3, 3, 2])
        self.assertEqual(evaluated[2][0], ())  # Failed OR branch discarded.
        self.assertEqual(evaluated[3][0], (Step("obtain", "Observed: obtain"),))
        self.assertIn("Observed: obtain", executor.calls[-2])
        self.assertNotIn("discarded alternative", executor.calls[-2])
        self.assertEqual(result.calls, len(executor.calls) + len(planner.calls))
        self.assertEqual(len(planner.calls), 2)

    async def test_direct_success_never_plans_and_thoughts_never_execute(self):
        provider = ScriptedProvider(
            ["think: Check input", "action: say task completed", "think: Task completed!"]
        )
        agent = ADAPT("Any task", provider, evaluate)
        result = await agent.run(max_depth=1)
        self.assertTrue(result.completed)
        self.assertEqual(
            result.steps, (Step("say task completed", "Observed: say task completed"),)
        )
        self.assertEqual(len(agent.evaluations), 1)
        self.assertEqual(result.calls, 3)

    async def test_mixed_logic_short_circuits_without_consuming_extra_depth(self):
        provider = ScriptedProvider(
            [
                "think: Task failed!",
                "Step 1: A\nStep 2: B\nStep 3: C\nStep 4: D\n"
                "Execution Order: ((Step 1 OR Step 2) AND Step 3 AND Step 4)",
                "think: Task failed!",
                "think: Task completed!",
                "think: Task failed!",
            ]
        )
        result = await ADAPT("Goal", provider, evaluate).run(max_depth=2)
        self.assertFalse(result.completed)
        self.assertEqual([e.task for e in result.executions], ["Goal", "A", "B", "C"])
        self.assertEqual([e.depth for e in result.executions], [1, 2, 2, 2])

    async def test_failed_conjunction_does_not_leak_into_next_alternative(self):
        provider = ScriptedProvider(
            [
                "think: Task failed!",
                "Step 1: A\nStep 2: B\nStep 3: C\nExecution Order: ((Step 1 AND Step 2) OR Step 3)",
                "action: temporary",
                "think: Task completed!",
                "think: Task failed!",
                "action: final",
                "think: Task completed!",
            ]
        )
        agent = ADAPT("Goal", provider, evaluate)
        result = await agent.run(max_depth=2)
        self.assertTrue(result.completed)
        self.assertEqual(result.steps, (Step("final", "Observed: final"),))
        self.assertEqual(agent.evaluations[-1]["history"], ())
        self.assertNotIn("Observed: temporary", provider.calls[-2])

    async def test_depth_turn_and_total_call_limits(self):
        provider = ScriptedProvider(["think: Task failed!"])
        result = await ADAPT("Goal", provider, evaluate).run(max_depth=1)
        self.assertFalse(result.completed)
        self.assertEqual(result.calls, 1)

        provider = ScriptedProvider(
            ["action: attempt", "Step 1: child\nExecution Order: Step 1", "think: Task completed!"]
        )
        result = await ADAPT("Goal", provider, evaluate).run(max_executor_steps=1)
        self.assertTrue(result.completed)
        self.assertEqual(result.executions[0].reason, "step_limit")
        self.assertEqual(result.steps, ())  # Unconfirmed attempt was discarded.

        provider = ScriptedProvider(["think: Task failed!"])
        agent = ADAPT("Goal", provider, evaluate)
        with self.assertRaisesRegex(RuntimeError, "call budget"):
            await agent.run(max_calls=1)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(len(agent.executions), 1)

    async def test_invalid_generation_and_callback_errors_remain_visible(self):
        for invalid in ("", "action: ", "think: "):
            agent = ADAPT("Goal", ScriptedProvider([invalid]), evaluate)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                await agent.run()
            self.assertEqual(agent.calls[0]["response"], invalid)
            self.assertIn("error", agent.calls[0])

        agent = ADAPT("Goal", ScriptedProvider(["think: Task failed!", "invalid plan"]), evaluate)
        with self.assertRaises(ValueError):
            await agent.run()
        self.assertEqual(agent.calls[-1]["response"], "invalid plan")
        self.assertIn("error", agent.calls[-1])

        async def broken(history, action):
            raise OSError("environment unavailable")

        agent = ADAPT("Goal", ScriptedProvider(["action: attempt"]), broken)
        with self.assertRaisesRegex(OSError, "environment unavailable"):
            await agent.run()
        self.assertIn("OSError", agent.evaluations[0]["error"])
        self.assertEqual(len(agent.calls), 1)

        agent = ADAPT("Goal", ScriptedProvider([ConnectionError("offline")]), evaluate)
        with self.assertRaises(ConnectionError):
            await agent.run()
        self.assertIn("ConnectionError", agent.calls[0]["error"])

    async def test_templates_bind_owner_and_work_outside_repository(self):
        agent = ADAPT("Unique task", ScriptedProvider([]), evaluate, actions="Allowed actions")
        initial = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                rendered = await ADAPT.next_action.render(agent, "Subtask", (), [])
                self.assertIn("Unique task", rendered)
                self.assertIn("Allowed actions", rendered)
                self.assertIn("Subtask", rendered)
                rendered = await ADAPT.plan.render(agent, "Subtask", (), "step_limit", "State")
                self.assertIn("Execution Order:", rendered)
                self.assertIn("State", rendered)
            finally:
                os.chdir(initial)

        for path in (Path(__file__).resolve().parents[1] / "adapt/prompts").glob("*.j2"):
            parsed = Environment().parse(path.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])

    async def test_partial_success_is_returned_and_runs_reset_records(self):
        provider = ScriptedProvider(
            [
                "think: Task failed!",
                "Step 1: Prepare\nStep 2: Finish\nExecution Order: (Step 1 AND Step 2)",
                "action: prepared",
                "think: Task completed!",
                "think: Task failed!",
                "think: Task completed!",
            ]
        )
        agent = ADAPT("Goal", provider, evaluate)
        first = await agent.run(max_depth=2, max_calls=5)
        self.assertFalse(first.completed)
        self.assertEqual(first.steps, (Step("prepared", "Observed: prepared"),))
        second = await agent.run(max_depth=1, max_calls=1)
        self.assertTrue(second.completed)
        self.assertEqual(second.calls, 1)
        self.assertEqual(second.steps, ())
        self.assertEqual(len(second.executions), 1)
        self.assertEqual(agent.evaluations, [])
        self.assertEqual(first.calls, 5)
        self.assertNotIn("Observed: prepared", provider.calls[-1])

    def test_parser_rejects_invalid_logic_and_accepts_paper_format(self):
        valid = parse_plan(
            "# Think: Abstract plan\nStep 1: A\nStep 2: B\nStep 3: C\n"
            "Execution Order: ((Step 1 OR Step 2) AND Step 3)"
        )
        self.assertEqual(valid.tasks, {1: "A", 2: "B", 3: "C"})
        for invalid in (
            "",
            "Step 1: A",
            "Step 1: \nExecution Order: Step 1",
            "Step 1: A\nStep 1: B\nExecution Order: Step 1",
            "Step 1: A\nExecution Order: Step 2",
            "Step 1: A\nStep 2: B\nExecution Order: Step 1",
            "Step 1: A\nExecution Order: (Step 1 OR)",
            "Step 1: A\nExecution Order: (Step 1 OR Step 1)",
            "Step 1: A\nExecution Order: Step 1 junk",
            "Step 1: A\nExecution Order: __import__('os').system('echo unsafe')",
            "Step 1: A\nExecution Order: (Step 1",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_plan(invalid)


if __name__ == "__main__":
    unittest.main()
