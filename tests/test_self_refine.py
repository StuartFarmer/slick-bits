"""Check SELF-REFINE's ordering, stopping, history, and failure boundaries offline."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from self_refine import SelfRefine, Step
from tests.providers import ScriptedProvider


class SelfRefineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "self_refine/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_same_model_full_history_and_last_output(self):
        for task in ("Write an invitation", "Design a delivery schedule"):
            provider = ScriptedProvider(
                ["draft-zero", "fix-zero", "draft-one", "fix-one", "draft-two", "NO_FEEDBACK"]
            )
            agent = SelfRefine(task, provider)
            result = await agent.run("source material", max_refinements=4)
            self.assertEqual(result.output, "draft-two")
            self.assertEqual(result.stop_reason, "feedback")
            self.assertEqual(result.calls, 6)
            self.assertEqual(
                result.history,
                (
                    Step("draft-zero", "fix-zero"),
                    Step("draft-one", "fix-one"),
                    Step("draft-two", "NO_FEEDBACK"),
                ),
            )
            self.assertEqual(
                [c["operation"] for c in agent.calls],
                ["generate", "feedback", "refine", "feedback", "refine", "feedback"],
            )
            self.assertTrue(all(task in c and "source material" in c for c in provider.calls))
            self.assertNotIn("draft-zero", provider.calls[3])
            self.assertNotIn("fix-zero", provider.calls[3])
            history_prompt = provider.calls[4]
            positions = [
                history_prompt.index(s) for s in ("draft-zero", "fix-zero", "draft-one", "fix-one")
            ]
            self.assertEqual(positions, sorted(positions))

    async def test_stop_before_refining_and_exact_marker(self):
        provider = ScriptedProvider(["draft", " \nNO_FEEDBACK\n "])
        result = await SelfRefine("Task", provider).run("Input")
        self.assertEqual(result.output, "draft")
        self.assertEqual(result.calls, 2)
        provider = ScriptedProvider(["draft", "Do not say NO_FEEDBACK yet", "revision", "more"])
        result = await SelfRefine("Task", provider).run("Input", max_refinements=1)
        self.assertEqual(result.output, "revision")
        self.assertEqual(result.stop_reason, "budget")
        self.assertEqual(result.calls, 4)

    async def test_budget_includes_final_feedback_and_zero_refinements(self):
        for budget in (0, 1, 4):
            responses = ["initial"]
            for index in range(budget):
                responses.extend([f"fix {index}", f"revision {index}"])
            responses.append("still needs work")
            agent = SelfRefine("Task", ScriptedProvider(responses))
            result = await agent.run("Input", max_refinements=budget)
            self.assertEqual(result.calls, 2 * budget + 2)
            self.assertEqual(len(result.history), budget + 1)
            self.assertEqual(result.stop_reason, "budget")
            self.assertEqual(result.output, responses[-2])

    async def test_existing_draft_custom_stop_and_state_reset(self):
        seen = []

        def stop(feedback, iteration):
            seen.append((feedback, iteration))
            return iteration == 1

        provider = ScriptedProvider(["NO_FEEDBACK", "new", "custom done", "again", "NO_FEEDBACK"])
        agent = SelfRefine("Task", provider, stop=stop)
        result = await agent.run("Input", initial_output="  supplied\n")
        self.assertEqual(seen, [("NO_FEEDBACK", 0), ("custom done", 1)])
        self.assertEqual(result.calls, 3)
        self.assertEqual(result.history[0].output, "  supplied\n")
        second = await agent.run("Other input", max_refinements=0)
        self.assertEqual(second.history, (Step("again", "NO_FEEDBACK"),))
        self.assertEqual(second.calls, 2)
        self.assertEqual(len(result.history), 2)
        self.assertNotIn("supplied", provider.calls[-1])

    async def test_failures_preserve_raw_response_and_partial_state_without_retry(self):
        for responses, operation, expected_output in (
            ([" \n"], "generate", None),
            (["draft", "\t"], "feedback", "draft"),
            (["draft", "fix", " "], "refine", "draft"),
            (["draft", "fix", TimeoutError("offline")], "refine", "draft"),
        ):
            provider = ScriptedProvider(responses)
            agent = SelfRefine("Task", provider)
            with self.assertRaises((ValueError, TimeoutError)):
                await agent.run("Input")
            self.assertEqual(len(provider.calls), len(responses))
            self.assertEqual(agent.output, expected_output)
            self.assertEqual(agent.calls[-1]["operation"], operation)
            self.assertIn("error", agent.calls[-1])
            if operation == "refine":
                self.assertEqual(agent.history, [Step("draft", "fix")])
            if not isinstance(responses[-1], Exception):
                self.assertEqual(agent.calls[-1]["response"], responses[-1])

    async def test_callback_errors_propagate_after_feedback(self):
        def stop(feedback, iteration):
            raise RuntimeError("callback failed")

        agent = SelfRefine("Task", ScriptedProvider(["draft", "feedback"]), stop=stop)
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            await agent.run("Input")
        self.assertEqual(agent.history, [Step("draft", "feedback")])
        self.assertEqual(len(agent.calls), 2)

    async def test_tool_requests_are_rejected_with_raw_text_retained(self):
        provider = ScriptedProvider([])
        agent = SelfRefine("Task", provider)
        with patch.object(
            provider, "acall", return_value=("raw response", ["tool request"])
        ) as call:
            with self.assertRaisesRegex(ValueError, "tool requests"):
                await agent.run("Input")
        call.assert_awaited_once()
        self.assertEqual(agent.calls[0]["response"], "raw response")
        self.assertIn("error", agent.calls[0])
        self.assertIsNone(agent.output)

    async def test_templates_render_from_other_directories_with_custom_prompts(self):
        agent = SelfRefine(
            "Task",
            ScriptedProvider([]),
            generation_prompt="GEN examples",
            feedback_prompt="FB rubric and examples",
            refinement_prompt="REF examples",
        )
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.templates.parent)
        rendered = (
            await SelfRefine.generate.render(agent, "Input"),
            await SelfRefine.feedback.render(agent, "Input", "Current draft"),
            await SelfRefine.refine.render(agent, "Input", [Step("Old draft", "Fix")]),
        )
        for prompt, marker in zip(rendered, ("GEN examples", "FB rubric", "REF examples")):
            self.assertIn(marker, prompt)
            self.assertIn("Task", prompt)
            self.assertIn("Input", prompt)
        for template in self.templates.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        self.assertEqual(agent.calls, [])


if __name__ == "__main__":
    unittest.main()
