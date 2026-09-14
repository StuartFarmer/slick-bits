"""Exercise Reflexion's trial loop through real Slick prompts, without model calls."""

import os
import unittest
from pathlib import Path

from jinja2 import Environment, nodes
from slick import prompts

from reflexion import Evaluation, Reflexion, Trajectory
from tests.providers import ScriptedProvider


class ReflexionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "reflexion" / "prompts"
        previous = prompts.TEMPLATE_ROOT
        prompts.TEMPLATE_ROOT = self.templates
        self.addCleanup(setattr, prompts, "TEMPLATE_ROOT", previous)

    async def test_failed_trials_learn_with_fifo_memory_and_stop_on_success(self):
        provider = ScriptedProvider(
            ["draft-0", "lesson-0", "draft-1", "lesson-1", "draft-2", "lesson-2", "  done\n"]
        )
        evaluated = []

        async def evaluate(trajectory):
            evaluated.append(trajectory)
            return Evaluation(trajectory.output.strip() == "done", "external evidence", 0.25)

        agent = Reflexion("Task", provider, evaluate)
        result = await agent.run("Input", max_trials=8, memory_size=2)
        self.assertEqual(result.trajectory.output, "  done\n")
        self.assertEqual(result.stop_reason, "success")
        self.assertEqual(result.memory, ("lesson-1", "lesson-2"))
        self.assertEqual(len(result.trials), 4)
        self.assertEqual(result.calls, 7)
        self.assertEqual(evaluated, [trial.trajectory for trial in result.trials])
        self.assertEqual(
            [call["operation"] for call in agent.calls],
            ["generate", "reflect", "retry", "reflect", "retry", "reflect", "retry"],
        )
        self.assertIn("draft-0", provider.calls[1])
        self.assertIn("external evidence", provider.calls[1])
        self.assertIn("0.25", provider.calls[1])
        self.assertIn("lesson-0", provider.calls[3])
        self.assertNotIn("draft-0", provider.calls[2])
        self.assertNotIn("lesson-0", provider.calls[-1])
        self.assertIn("lesson-1", provider.calls[-1])
        self.assertIn("lesson-2", provider.calls[-1])

    async def test_budget_counts_initial_trial_and_zero_makes_no_calls(self):
        async def evaluate(trajectory):
            return Evaluation(False, "try again")

        for budget, responses in [(0, []), (1, ["one"]), (2, ["one", "lesson", "two"])]:
            with self.subTest(budget=budget):
                agent = Reflexion("Task", ScriptedProvider(responses), evaluate)
                result = await agent.run("Input", max_trials=budget, memory_size=0)
                self.assertEqual(len(result.trials), budget)
                self.assertEqual(result.calls, len(responses))
                self.assertEqual(result.memory, ())
                self.assertEqual(result.stop_reason, "budget")
                self.assertEqual(result.trajectory, Trajectory(responses[-1]) if budget else None)

    async def test_external_actor_receives_memory_and_supplies_observed_trajectory(self):
        seen = []

        async def actor(task, input, memory):
            seen.append((task, input, memory))
            return Trajectory("done" if memory else "failed", "action: open; observation: locked")

        async def evaluate(trajectory):
            self.assertIn("observation: locked", trajectory.trace)
            return Evaluation(trajectory.output == "done", "environment reward", -1.0)

        provider = ScriptedProvider([])
        reflector = ScriptedProvider(["get the key"])
        agent = Reflexion("Task", provider, evaluate, actor=actor, reflection_provider=reflector)
        result = await agent.run("Room")
        self.assertEqual(seen, [("Task", "Room", ()), ("Task", "Room", ("get the key",))])
        self.assertEqual(result.calls, 1)
        self.assertEqual(result.stop_reason, "success")
        self.assertEqual(provider.calls, [])
        self.assertIn("observation: locked", reflector.calls[0])

    async def test_first_success_and_new_runs_have_no_stale_memory(self):
        async def evaluate(trajectory):
            return Evaluation(trajectory.output == "accepted")

        agent = Reflexion(
            "Task", ScriptedProvider(["fail", "lesson", "accepted", "accepted"]), evaluate
        )
        first = await agent.run("First")
        second = await agent.run("Second")
        self.assertEqual(first.memory, ("lesson",))
        self.assertEqual(second.memory, ())
        self.assertEqual(second.calls, 1)
        self.assertEqual(second.stop_reason, "success")
        self.assertNotIn("lesson", agent.calls[0]["prompt"])
        self.assertEqual(len(first.trials), 2)

    async def test_generation_failures_preserve_raw_responses_and_partial_trials(self):
        async def evaluate(trajectory):
            return Evaluation(False)

        for responses, trials in [
            ([" \n"], 0),
            (["draft", "\t"], 1),
            (["draft", "lesson", " "], 1),
            (["draft", TimeoutError("offline")], 1),
            ([("raw", ["unexpected tool"])], 0),
        ]:
            with self.subTest(responses=responses):
                agent = Reflexion("Task", ScriptedProvider(responses), evaluate)
                with self.assertRaises((ValueError, TimeoutError)):
                    await agent.run("Input")
                self.assertEqual(len(agent.calls), len(responses))
                self.assertEqual(len(agent.trials), trials)
                self.assertIn("error", agent.calls[-1])
                if isinstance(responses[-1], str):
                    self.assertEqual(agent.calls[-1]["response"], responses[-1])

    async def test_callback_errors_propagate_without_reattempts(self):
        async def evaluate(trajectory):
            raise RuntimeError("evaluation failed")

        agent = Reflexion("Task", ScriptedProvider(["draft"]), evaluate)
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            await agent.run("Input")
        self.assertEqual(agent.trajectory, Trajectory("draft"))
        self.assertEqual(agent.trials, [])
        self.assertEqual(len(agent.calls), 1)

        async def actor(task, input, memory):
            raise RuntimeError("environment failed")

        agent = Reflexion("Task", ScriptedProvider([]), evaluate, actor=actor)
        with self.assertRaisesRegex(RuntimeError, "environment failed"):
            await agent.run("Input")
        self.assertEqual(agent.calls, [])

    async def test_all_templates_render_outside_repo_without_conditional_jinja(self):
        async def evaluate(trajectory):
            return Evaluation(True)

        agent = Reflexion("Task", ScriptedProvider([]), evaluate)
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir("/tmp")
        rendered = [
            await Reflexion.generate.render(agent, "Input", ()),
            await Reflexion.retry.render(agent, "Input", ("lesson",)),
            await Reflexion.reflect.render(
                agent,
                "Input",
                Trajectory("draft", "observed trace"),
                Evaluation(False),
                ("lesson",),
            ),
        ]
        for value in rendered:
            self.assertIn("Task", value)
            self.assertIn("Input", value)
        for template in self.templates.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        self.assertEqual(agent.calls, [])


if __name__ == "__main__":
    unittest.main()
