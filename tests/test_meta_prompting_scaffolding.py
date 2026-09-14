"""Exercise the conductor protocol and isolated expert calls through Slick."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from meta_prompting_scaffolding import MetaPromptingScaffolding
from tests.providers import ScriptedProvider


class ScaffoldingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "meta_prompting_scaffolding/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_conductor_retains_history_and_experts_only_see_delegated_context(self):
        provider = ScriptedProvider(
            [
                'Expert Designer:\n"""Design a blue garden."""',
                "Plant blue flowers.",
                'Expert Reviewer:\n"""Check this: Plant blue flowers."""',
                "Confirmed.",
                '>> FINAL ANSWER:\n"""  Plant blue flowers.\n"""',
            ]
        )
        agent = MetaPromptingScaffolding("PRIVATE: Design a garden", provider)
        result = await agent.run()
        self.assertEqual(result.answer, "  Plant blue flowers.\n")
        self.assertEqual((result.rounds, result.calls, result.stop_reason), (3, 5, "final_answer"))
        for index in (1, 3):
            self.assertNotIn("PRIVATE", provider.calls[index])
            self.assertNotIn("Meta-Expert", provider.calls[index])
        self.assertNotIn("Design a blue garden.", provider.calls[3])
        self.assertIn("You are Expert Reviewer.", provider.calls[3])
        for fragment in ("PRIVATE", "Design a blue garden.", "Plant blue flowers.", "Confirmed."):
            self.assertIn(fragment, provider.calls[4])
        self.assertIn("have another expert(s) verify it", provider.calls[2])

    async def test_first_expert_precedes_final_and_only_first_final_is_returned(self):
        provider = ScriptedProvider(
            [
                'Expert A: """First"""\nExpert B: """Second"""\n>> FINAL ANSWER: """Early"""',
                "Checked",
                '>> FINAL ANSWER: """One"""\n>> FINAL ANSWER: """Two"""',
            ]
        )
        result = await MetaPromptingScaffolding("Any task", provider).run()
        self.assertEqual((result.answer, result.calls), ("One", 3))
        self.assertIn("First", provider.calls[1])
        self.assertNotIn("Second", provider.calls[1])

    async def test_malformed_rounds_consume_budget_and_remain_visible(self):
        for response in (
            "unformatted",
            '>> FINAL ANSWER: """unterminated',
            'Expert A: """   """',
            '>> FINAL ANSWER: """ """',
        ):
            with self.subTest(response=response):
                provider = ScriptedProvider([response, '>> FINAL ANSWER: """Recovered"""'])
                agent = MetaPromptingScaffolding("Task", provider)
                result = await agent.run(max_rounds=2)
                self.assertEqual((result.answer, result.rounds), ("Recovered", 2))
                self.assertEqual(agent.calls[0]["response"], response)
                self.assertIn(response, provider.calls[1])
                self.assertIn("If you have determined the final answer", provider.calls[1])
                self.assertIn("This is the last round", provider.calls[1])
        agent = MetaPromptingScaffolding("Task", ScriptedProvider(["still thinking"]))
        result = await agent.run(max_rounds=1)
        self.assertIsNone(result.answer)
        self.assertEqual((result.rounds, result.calls, result.stop_reason), (1, 1, "round_limit"))
        result = await agent.run(max_rounds=0)
        self.assertEqual((result.rounds, result.calls, result.answer), (0, 0, None))

    async def test_python_code_is_delegated_and_result_returns_to_conductor(self):
        executed = []

        async def execute(code):
            executed.append(code)
            return "42\n"

        provider = ScriptedProvider(
            [
                'Expert Python: """Calculate six times seven."""',
                "```python\nprint(6 * 7)\n```\nPlease run this code!",
                '>> FINAL ANSWER: """42"""',
            ]
        )
        agent = MetaPromptingScaffolding("Compute", provider, execute_python=execute)
        result = await agent.run()
        self.assertEqual(result.answer, "42")
        self.assertEqual(executed, ["print(6 * 7)"])
        self.assertIn("Please run this code!", provider.calls[1])
        self.assertIn("output of the code when executed:\n\n42", provider.calls[2])
        self.assertEqual(agent.executions[0]["output"], "42\n")

    async def test_python_requires_opt_in_marker_and_complete_fence(self):
        async def never_execute(code):
            self.fail("Unexpected code execution")

        for enabled, response in (
            (False, "```python\nprint(42)\n```\nPlease run this code!"),
            (True, "```python\nprint(42)\n```"),
            (True, "Please run this code!"),
            (True, "```python\nprint(42)\nPlease run this code!"),
        ):
            with self.subTest(enabled=enabled, response=response):
                provider = ScriptedProvider(
                    [
                        'Expert Python: """Compute."""',
                        response,
                        '>> FINAL ANSWER: """Done"""',
                    ]
                )
                agent = MetaPromptingScaffolding(
                    "Task", provider, execute_python=never_execute if enabled else None
                )
                await agent.run()
                if not enabled:
                    self.assertNotIn("special access to Expert Python", provider.calls[0])
                if enabled and response.endswith("Please run this code!"):
                    self.assertIn("No complete Python code block", provider.calls[2])

    async def test_python_selects_complete_blocks_without_repairing_fence_pairing(self):
        executed = []

        async def execute(code):
            executed.append(code)
            return "42"

        provider = ScriptedProvider(
            [
                'Expert Python: """Compute."""',
                "```text\nExplanation\n```\nNow run this:\n"
                "```python\nprint(42)\n```\nPlease run this code!",
                'Expert Python: """Recheck."""',
                "```python\nprint(1)\n```\nThen:\n```\nprint(42)\n```\nPlease run this code!",
                '>> FINAL ANSWER: """42"""',
            ]
        )
        await MetaPromptingScaffolding("Task", provider, execute_python=execute).run()
        self.assertEqual(executed, ["print(42)", "print(42)"])

    async def test_empty_expert_allows_final_but_native_tool_requests_abort(self):
        provider = ScriptedProvider(['Expert A: """   """\n>> FINAL ANSWER: """Done"""'])
        result = await MetaPromptingScaffolding("Task", provider).run()
        self.assertEqual((result.answer, result.calls), ("Done", 1))
        agent = MetaPromptingScaffolding("Task", ScriptedProvider([("raw", ["tool request"])]))
        with self.assertRaisesRegex(ValueError, "native tool requests"):
            await agent.run()
        self.assertEqual(agent.calls[0]["response"], "raw")
        self.assertIn("ValueError", agent.calls[0]["error"])

    async def test_errors_propagate_with_call_and_execution_records(self):
        agent = MetaPromptingScaffolding("Task", ScriptedProvider([OSError("transport")]))
        with self.assertRaisesRegex(OSError, "transport"):
            await agent.run()
        self.assertIn("OSError", agent.calls[0]["error"])

        async def execute(code):
            raise TimeoutError("sandbox unavailable")

        agent = MetaPromptingScaffolding(
            "Task",
            ScriptedProvider(
                [
                    'Expert Python: """Compute."""',
                    "```python\nprint(42)\n```\nPlease run this code!",
                ]
            ),
            execute_python=execute,
        )
        with self.assertRaisesRegex(TimeoutError, "sandbox unavailable"):
            await agent.run()
        self.assertEqual(len(agent.calls), 2)
        self.assertIn("Please run this code!", agent.calls[-1]["response"])
        self.assertIn("TimeoutError", agent.executions[0]["error"])

    async def test_run_reset_and_all_templates_from_another_directory(self):
        provider = ScriptedProvider(
            ['>> FINAL ANSWER: """First"""', '>> FINAL ANSWER: """Second"""']
        )
        agent = MetaPromptingScaffolding("Task marker", provider)
        previous = Path.cwd()
        try:
            os.chdir(self.templates.parent)
            first = await agent.run()
            second = await agent.run()
            self.assertEqual((first.answer, second.answer), ("First", "Second"))
            self.assertEqual((first.calls, second.calls, len(agent.calls)), (1, 1, 1))
            self.assertNotIn("First", provider.calls[1])
            self.assertIn("First", first.history[-1].content)
            for operation in (MetaPromptingScaffolding.consult, MetaPromptingScaffolding.program):
                rendered = await operation.render(agent, "Expert Tester", "Instruction marker")
                self.assertIn("Instruction marker", rendered)
                self.assertNotIn("Task marker", rendered)
            for template in self.templates.glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
