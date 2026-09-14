"""Exercise HMAW's three independent generations and verbatim skip connections."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from hmaw import HMAW
from tests.providers import ScriptedProvider


class HMAWTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "hmaw/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_three_stages_preserve_query_and_only_previous_instructions(self):
        for task in (
            "Explain tides to a beginner.",
            "Review this code:\n    x = {'name': '{{ untouched }}'}\n    print(x)\n",
        ):
            with self.subTest(task=task):
                provider = ScriptedProvider(["CEO marker", "Manager marker", "  Final artifact\n"])
                agent = HMAW(task, provider)
                result = await agent.run()
                self.assertEqual(result.answer, "  Final artifact\n")
                self.assertEqual(result.ceo_instructions, "CEO marker")
                self.assertEqual(result.manager_instructions, "Manager marker")
                self.assertEqual(result.optimized_prompt, provider.calls[2])
                self.assertEqual(result.calls, 3)
                self.assertEqual(len(provider.calls), 3)
                for context in provider.calls:
                    self.assertIn(f"<{task}>", context)
                self.assertNotIn("CEO marker", provider.calls[0])
                self.assertIn("<CEO marker>", provider.calls[1])
                self.assertNotIn("CEO marker", provider.calls[2])
                self.assertIn("<Manager marker>", provider.calls[2])
                self.assertEqual(agent.calls[-1]["response"], result.answer)

    async def test_role_routing_and_repeated_runs_do_not_accumulate_history(self):
        ceo = ScriptedProvider(["First CEO", "Second CEO"])
        manager = ScriptedProvider(["First manager", "Second manager"])
        worker = ScriptedProvider(["First answer", "Second answer"])
        agent = HMAW("Task", worker, ceo_provider=ceo, manager_provider=manager)
        first = await agent.run()
        second = await agent.run()
        self.assertEqual(first.answer, "First answer")
        self.assertEqual(second.answer, "Second answer")
        self.assertEqual(len(agent.calls), 3)
        for source in (ceo, manager, worker):
            self.assertEqual(len(source.calls), 2)
            self.assertNotIn("First", source.calls[1])

    async def test_failures_stop_downstream_calls_and_keep_raw_records(self):
        for stage in range(3):
            for failure in (" \n", OSError("transport failed")):
                responses = ["Prior instruction"] * stage + [failure]
                provider = ScriptedProvider(responses)
                agent = HMAW("Task", provider)
                with self.subTest(stage=stage, failure=failure):
                    with self.assertRaises((ValueError, OSError)):
                        await agent.run()
                    self.assertEqual(len(provider.calls), stage + 1)
                    self.assertEqual(len(agent.calls), stage + 1)
                    self.assertIn("error", agent.calls[-1])
                    if failure == " \n":
                        self.assertEqual(agent.calls[-1]["response"], failure)

    async def test_worker_text_is_not_truncated_at_paper_heading(self):
        answer = "Example heading: **Response for the User**: keep the whole artifact"
        result = await HMAW("Task", ScriptedProvider(["CEO", "Manager", answer])).run()
        self.assertEqual(result.answer, answer)

    async def test_tool_requests_are_rejected_without_execution(self):
        provider = ScriptedProvider([])
        agent = HMAW("Task", provider)
        with patch.object(provider, "acall", return_value=("Tool output", [object()])):
            with self.assertRaisesRegex(ValueError, "tool requests"):
                await agent.run()
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(agent.calls[0]["response"], "Tool output")
        self.assertIn("error", agent.calls[0])

    async def test_templates_render_from_another_directory_without_model_calls(self):
        provider = ScriptedProvider([])
        agent = HMAW("Original query", provider)
        previous = Path.cwd()
        try:
            os.chdir(self.templates)
            for operation, args, role in (
                (HMAW.instruct_manager, (), "CEO"),
                (HMAW.instruct_worker, ("CEO instructions",), "MANAGER"),
                (HMAW.respond, ("Manager instructions",), "WORKER"),
            ):
                rendered = await operation.render(agent, *args)
                self.assertIn(f"**Your ROLE**: <{role}>", rendered)
                self.assertIn("<Original query>", rendered)
            for template in self.templates.glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
