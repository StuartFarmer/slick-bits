"""Exercise APET's rewrite, input retention, isolated answers, and failures offline."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from apet import APET
from tests.providers import ScriptedProvider


class APETTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "apet/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_rewrite_then_fresh_answer_for_unrelated_tasks(self):
        for task in ("Explain tides to a child.", "Review SQL: SELECT '{{ literal }}';"):
            provider = ScriptedProvider(["Rewritten instructions", "  Answer\n"])
            agent = APET(task, provider)
            result = await agent.run()
            self.assertEqual(result.original_prompt, task)
            self.assertEqual(result.optimized_prompt, "Rewritten instructions")
            self.assertEqual(result.answer, "  Answer\n")
            self.assertIsNone(result.original_answer)
            self.assertIsNone(result.optimized_score)
            self.assertEqual(result.calls, 2)
            self.assertIn(task, provider.calls[0])
            self.assertEqual(provider.calls[1], "Rewritten instructions")

    async def test_comparison_order_and_evaluation_do_not_select_the_better_answer(self):
        seen = []

        async def evaluate(answer):
            seen.append(answer)
            return {"Baseline answer": 1.0, "Worse answer": 0.0}[answer]

        provider = ScriptedProvider(["Rewrite", "Baseline answer", "Worse answer"])
        agent = APET("Original task", provider, evaluate)
        result = await agent.run(compare=True)
        self.assertEqual(provider.calls[1:], ["Original task", "Rewrite"])
        self.assertEqual(seen, ["Baseline answer", "Worse answer"])
        self.assertEqual(result.answer, "Worse answer")
        self.assertEqual(result.original_answer, "Baseline answer")
        self.assertEqual((result.original_score, result.optimized_score), (1.0, 0.0))
        self.assertEqual(result.calls, 3)

    async def test_rewrite_only_retains_missing_input_exactly_and_resets_records(self):
        data = '  {"value": "{{ untouched }}"}\n'
        provider = ScriptedProvider(["Rewrite", "Already contains " + data])
        agent = APET("Interpret this data:\n" + data, provider, input_text=data)
        first = await agent.optimize()
        self.assertEqual(first, 'Rewrite\n\nInput for question:\n"""\n' + data + '\n"""')
        self.assertEqual(agent.calls[0]["response"], "Rewrite")
        second = await agent.optimize()
        self.assertEqual(second, "Already contains " + data)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(len(provider.calls), 2)

    async def test_blank_transport_and_tool_failures_stop_and_keep_raw_records(self):
        for prior in ([], ["Rewrite"], ["Rewrite", "Baseline"]):
            for failure in (" \n", TimeoutError("offline"), ("raw", ["tool request"])):
                with self.subTest(prior=prior, failure=failure):
                    provider = ScriptedProvider([*prior, failure])
                    agent = APET("Task", provider, input_text="Data")
                    with self.assertRaises((ValueError, TimeoutError)):
                        await agent.run(compare=True)
                    self.assertEqual(len(provider.calls), len(prior) + 1)
                    self.assertIn("error", agent.calls[-1])
                    if failure == " \n":
                        self.assertEqual(agent.calls[-1]["response"], " \n")
                    if isinstance(failure, tuple):
                        self.assertEqual(agent.calls[-1]["response"], "raw")

    async def test_evaluator_failures_propagate_after_generation(self):
        for failure in (float("nan"), float("inf"), RuntimeError("evaluation failed")):

            async def evaluate(answer):
                if isinstance(failure, Exception):
                    raise failure
                return failure

            provider = ScriptedProvider(["Rewrite", "Answer"])
            agent = APET("Task", provider, evaluate)
            with self.assertRaises((ValueError, RuntimeError)):
                await agent.run()
            self.assertEqual(len(agent.calls), 2)
            self.assertEqual(agent.calls[-1]["response"], "Answer")

    async def test_templates_render_from_other_directories_and_leave_input_literal(self):
        agent = APET("Task {{ literal }}", ScriptedProvider([]))
        previous = Path.cwd()
        try:
            for directory in (self.templates.parent, self.templates.parent.parent):
                os.chdir(directory)
                rendered = await APET.reformulate.render(agent)
                self.assertIn("Task {{ literal }}", rendered)
                self.assertIn("three different experts", rendered)
                self.assertIn("Let's think step-by-step", rendered)
                self.assertEqual(await APET.respond.render(agent, "Answer this"), "Answer this")
            for template in self.templates.glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)
        self.assertEqual(agent.calls, [])


if __name__ == "__main__":
    unittest.main()
