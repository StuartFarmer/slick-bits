"""Check independent sampling, answer marginalization, and failure accounting offline."""

import os
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from self_consistency import SelfConsistency, extract_answer
from tests.providers import ScriptedProvider


class SelfConsistencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "self_consistency/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_vote_counts_answers_not_paths_and_samples_identical_prompts(self):
        responses = [
            "Wrong path. The answer is 26.",
            "Subtract then multiply. The answer is 18.",
            "Multiply then subtract. The answer is 18.",
            "I cannot give an answer.",
        ]
        provider = ScriptedProvider(responses)
        agent = SelfConsistency(
            "Solve the supplied problem.", provider, examples="Q: demo\nA: demo"
        )
        result = await agent.run("Question {{ literal }}", samples=4)
        self.assertEqual(result.answer, "18")
        self.assertEqual(result.counts, {"26": 1, "18": 2})
        self.assertEqual(result.consistency, 0.5)
        self.assertEqual(len(result.samples), 4)
        self.assertIsNone(result.samples[-1].answer)
        self.assertIsNotNone(result.samples[-1].error)
        self.assertEqual([sample.response for sample in result.samples], responses)
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(len(set(provider.calls)), 1)
        self.assertIn("Question {{ literal }}", provider.calls[0])
        self.assertIn("Q: demo\nA: demo", provider.calls[0])
        self.assertEqual([call["response"] for call in agent.calls], responses)

    async def test_default_budget_plurality_ties_and_run_reset(self):
        provider = ScriptedProvider(
            ["The answer is first."] * 40
            + ["The answer is B.", "The answer is A.", "The answer is C."]
        )
        agent = SelfConsistency("Choose a label.", provider)
        first = await agent.run("input")
        self.assertEqual(len(first.samples), 40)
        self.assertEqual(first.consistency, 1.0)
        second = await agent.run("another input", samples=3)
        self.assertEqual(second.answer, "B")
        self.assertEqual(second.consistency, 1 / 3)
        self.assertEqual(len(agent.calls), 3)
        self.assertEqual(len(first.samples), 40)

    async def test_custom_parser_defines_equivalence_and_supports_unrelated_tasks(self):
        for task, outputs, parser, expected in (
            (
                "Return an amount.",
                ["$18.00", "18", "26"],
                lambda text: str(Decimal(text.removeprefix("$")).normalize()),
                "18",
            ),
            ("Classify a document.", ["YES", "yes", "No"], str.casefold, "yes"),
        ):
            agent = SelfConsistency(task, ScriptedProvider(outputs), extract_answer=parser)
            result = await agent.run("task data", samples=3)
            self.assertEqual(result.answer, expected)
            self.assertEqual(result.counts[expected], 2)

    async def test_all_invalid_samples_exhaust_budget_without_repair(self):
        agent = SelfConsistency("Task", ScriptedProvider(["", "The answer is .", "missing"]))
        with self.assertRaisesRegex(ValueError, "no valid answers"):
            await agent.run("input", samples=3)
        self.assertEqual(len(agent.calls), 3)
        self.assertEqual(len(agent.samples), 3)
        self.assertTrue(all(sample.error for sample in agent.samples))

    async def test_provider_tool_and_programming_errors_propagate_with_partial_records(self):
        for failure in (TimeoutError("offline"), ValueError("transport"), ("raw", ["tool"])):
            provider = ScriptedProvider(["The answer is A.", failure, "unused"])
            agent = SelfConsistency("Task", provider)
            with self.assertRaises((ValueError, TimeoutError)):
                await agent.run("input", samples=3)
            self.assertEqual(len(provider.calls), 2)
            self.assertEqual(len(agent.samples), 1)
            self.assertIn("error", agent.calls[-1])
            if isinstance(failure, tuple):
                self.assertEqual(agent.calls[-1]["response"], "raw")

        def broken_parser(text):
            raise RuntimeError("parser bug")

        agent = SelfConsistency("Task", ScriptedProvider(["raw"]), extract_answer=broken_parser)
        with self.assertRaisesRegex(RuntimeError, "parser bug"):
            await agent.run("input", samples=2)
        self.assertEqual(agent.calls[-1]["response"], "raw")
        self.assertIn("error", agent.calls[-1])

    def test_default_extraction_stops_before_next_question_and_keeps_text_answers(self):
        self.assertEqual(extract_answer("Work. The answer is 3.14.\nQ: The answer is 9."), "3.14")
        self.assertEqual(extract_answer("The answer is A. Revised. The answer is B."), "B")
        self.assertEqual(extract_answer("The answer is Arthur's Magazine."), "Arthur's Magazine")
        for output in ("18", "  ", "The answer is .", "Q: invented\nThe answer is fake."):
            with self.assertRaises(ValueError):
                extract_answer(output)

    async def test_template_rendering_is_local_literal_and_without_control_flow(self):
        agent = SelfConsistency("Task {{ literal }}", ScriptedProvider([]))
        previous = Path.cwd()
        try:
            for directory in (self.templates.parent, self.templates.parent.parent):
                os.chdir(directory)
                rendered = await SelfConsistency.generate.render(agent, "Data")
                self.assertIn("Task {{ literal }}", rendered)
                self.assertIn("Data", rendered)
            parsed = Environment().parse((self.templates / "generate.j2").read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)
        self.assertEqual(agent.calls, [])


if __name__ == "__main__":
    unittest.main()
