"""Check CAMEL's protocol and proposal selection through real Slick boundaries."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from pydantic import ValidationError

from camel import CAMEL, Choice, TokenLimitError
from tests.providers import ScriptedProvider


class CAMELTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "camel/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    def agent(self, responses, **kwargs):
        return CAMEL(
            "Organize the supplied material",
            ScriptedProvider(responses),
            assistant_role="Editor",
            user_role="Author",
            **kwargs,
        )

    async def test_specification_history_completion_and_extraction(self):
        agent = self.agent(
            [
                "Arrange the supplied material into two sections.",
                "Instruction: Draft the first section.\nInput: Source A",
                "Solution: First section.\nNext request.",
                "Instruction: Add the second section.\nInput: Source B",
                "Solution: Second section.\nNext request.",
                "  <CAMEL_TASK_DONE>\n",
                "First section.\n\nSecond section.",
            ]
        )
        result = await agent.run(extract=True)
        self.assertEqual(result.stop_reason, "task_done")
        self.assertEqual(result.solution, "First section.\n\nSecond section.")
        self.assertEqual(result.calls, 7)
        self.assertEqual(
            [m.speaker for m in result.messages], ["user", "assistant", "user", "assistant", "user"]
        )
        self.assertEqual(
            [c["operation"] for c in agent.calls],
            ["specify", "instruct", "solve", "instruct", "solve", "instruct", "extract"],
        )
        for index in (3, 4, 5):
            self.assertIn("Source A", agent.calls[index]["prompt"])
            self.assertIn("First section.", agent.calls[index]["prompt"])
        self.assertIn("Never forget you are a Author", agent.calls[1]["prompt"])
        self.assertIn("Never forget you are a Editor", agent.calls[2]["prompt"])
        self.assertIn("Second section.", agent.calls[-1]["prompt"])
        self.assertEqual(agent.calls[5]["response"], "  <CAMEL_TASK_DONE>\n")

    async def test_message_budget_counts_individual_messages(self):
        for limit in (0, 1, 2, 3, 40):
            with self.subTest(limit=limit):
                responses = [
                    "Instruction: Continue.\nInput: None",
                    "Solution: Progress.\nNext request.",
                ] * 20
                agent = self.agent(responses)
                result = await agent.run(specify_task=False, max_messages=limit)
                self.assertEqual(result.stop_reason, "message_limit")
                self.assertEqual(len(result.messages), limit)
                self.assertEqual(result.calls, limit)
                self.assertIsNone(result.solution)

    async def test_three_consecutive_missing_instructions_and_reset(self):
        agent = self.agent(
            [
                "Thanks",
                "Solution: Waiting",
                "Hello",
                "Solution: Waiting",
                "Instruction: Continue.\nInput: None",
                "Solution: Work",
                "Thanks",
                "Solution: Waiting",
                "Goodbye",
                "Solution: Waiting",
                "",
            ]
        )
        result = await agent.run(specify_task=False)
        self.assertEqual(result.stop_reason, "user_no_instruct")
        self.assertEqual(result.calls, 11)
        self.assertEqual(result.messages[-1].content, "")

    async def test_assistant_instruction_stops_and_done_is_user_only_exact_match(self):
        agent = self.agent(
            [
                "Instruction: Explain <CAMEL_TASK_DONE>.\nInput: None",
                "<CAMEL_TASK_DONE>",
                "Instruction: Continue.\nInput: None",
                "Instruction: You do the work.\nInput: None",
            ]
        )
        result = await agent.run(specify_task=False)
        self.assertEqual(result.stop_reason, "assistant_instruct")
        self.assertEqual(result.calls, 4)

    async def test_empty_instruction_cannot_borrow_content_from_input_line(self):
        agent = self.agent(["Instruction:  \nInput: None", "Solution: Waiting"] * 3)
        result = await agent.run(specify_task=False, max_messages=6)
        self.assertEqual(result.stop_reason, "user_no_instruct")
        self.assertEqual(result.calls, 5)

    async def test_critic_branches_both_roles_without_rejected_history(self):
        critic = ScriptedProvider(
            [
                Choice(option=2, explanation="Better instruction"),
                Choice(option=1, explanation="Better solution"),
                Choice(option=2, explanation="Task solved"),
            ]
        )
        agent = self.agent(
            [
                "Instruction: REJECTED_USER",
                "Instruction: Keep this",
                "Solution: Keep that",
                "Solution: REJECTED_ASSISTANT",
                "Instruction: Unnecessary work",
                "<CAMEL_TASK_DONE>",
            ],
            critic=critic,
            candidates=2,
        )
        result = await agent.run(specify_task=False)
        self.assertEqual(result.stop_reason, "task_done")
        self.assertEqual(result.calls, 9)
        self.assertEqual(
            [m.content for m in result.messages],
            ["Instruction: Keep this", "Solution: Keep that", "<CAMEL_TASK_DONE>"],
        )
        self.assertEqual([s.choice.option for s in result.selections], [2, 1, 2])
        solve_calls = [c for c in agent.calls if c["operation"] == "solve"]
        self.assertEqual(solve_calls[0]["prompt"], solve_calls[1]["prompt"])
        self.assertNotIn("REJECTED_USER", solve_calls[0]["prompt"])
        self.assertNotIn("REJECTED_ASSISTANT", agent.calls[-3]["prompt"])
        self.assertIn("REJECTED_USER", critic.calls[0])

    async def test_invalid_critic_output_is_logged_and_never_selected(self):
        for output, error in [
            ("not json", ValidationError),
            ('{"option": 3, "explanation": "Bad ID"}', ValueError),
            ('{"option": true, "explanation": "Bad ID"}', ValidationError),
        ]:
            with self.subTest(output=output):
                agent = self.agent(
                    ["Instruction: A", "Instruction: B"],
                    critic=ScriptedProvider([output]),
                    candidates=2,
                )
                with self.assertRaises(error):
                    await agent.run(specify_task=False)
                self.assertEqual(agent.messages, [])
                self.assertEqual(agent.calls[-1]["response"], output)
                self.assertIn("error", agent.calls[-1])

    async def test_external_critic_and_separate_providers(self):
        seen = []

        async def review(speaker, proposals, history):
            seen.append((speaker, proposals, history))
            return Choice(option=2, explanation="Human preference")

        user = ScriptedProvider(["Instruction: First", "Instruction: Second"])
        agent = self.agent(
            ["Solution: A", "Solution: B"], user_provider=user, review=review, candidates=2
        )
        result = await agent.run(specify_task=False, max_messages=2)
        self.assertEqual(
            [m.content for m in result.messages], ["Instruction: Second", "Solution: B"]
        )
        self.assertEqual([item[0] for item in seen], ["user", "assistant"])
        self.assertEqual(len(seen[1][2]), 1)
        self.assertEqual(result.calls, 4)

    async def test_token_limit_and_provider_failure_preserve_partial_work(self):
        agent = self.agent(["Instruction: Work", TokenLimitError("context full")])
        result = await agent.run(specify_task=False)
        self.assertEqual(result.stop_reason, "token_limit")
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.calls, 2)
        agent = self.agent([], token_limit=10, count_tokens=lambda text: 10)
        result = await agent.run(specify_task=False)
        self.assertEqual(result.stop_reason, "token_limit")
        self.assertEqual(result.calls, 0)
        self.assertFalse(agent.calls[0]["attempted"])
        for output, error in [
            (RuntimeError("offline"), RuntimeError),
            (("text", [{"id": "tool"}]), ValueError),
        ]:
            agent = self.agent(["Instruction: Work", output])
            with self.assertRaises(error):
                await agent.run(specify_task=False)
            self.assertEqual(len(agent.messages), 1)
            self.assertIn("error", agent.calls[-1])

    async def test_empty_specification_rejected_and_runs_reset(self):
        agent = self.agent([" "])
        with self.assertRaisesRegex(ValueError, "blank"):
            await agent.run()
        self.assertEqual(agent.calls[0]["response"], " ")
        agent = self.agent(["<CAMEL_TASK_DONE>", "Instruction: Next"])
        first = await agent.run(specify_task=False)
        second = await agent.run(specify_task=False, max_messages=1)
        self.assertEqual(first.stop_reason, "task_done")
        self.assertEqual(second.calls, 1)
        self.assertEqual(len(second.messages), 1)

    async def test_all_templates_render_from_another_directory(self):
        agent = self.agent([])
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.templates.parent)
        rendered = [
            await CAMEL.specify.render(agent, 50),
            await CAMEL.instruct.render(agent),
            await CAMEL.solve.render(agent),
            await CAMEL.select.render(agent, "user", ("A", "B")),
            await CAMEL.extract.render(agent),
        ]
        self.assertTrue(all(rendered))
        self.assertIn('"properties"', rendered[3])
        for template in self.templates.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
