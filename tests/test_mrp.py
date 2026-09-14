"""Exercise MRP selection and execution through real Slick boundaries, offline."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from mrp import MRP, Method
from tests.providers import ScriptedProvider


class MRPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "mrp/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_scores_every_method_then_executes_only_stable_argmax(self):
        for task in ("Plan deliveries", "Write a poem"):
            seen = []

            async def solve(input):
                seen.append(input)
                return "  chosen artifact\n"

            methods = [
                Method("A", "Description A"),
                Method("B", "Description B", solve),
                Method("C", "Description C"),
            ]
            provider = ScriptedProvider(['{"score": 2}', '{"score": 7}', '{"score": 7}'])
            agent = MRP(task, provider, methods=methods)
            result = await agent.run("input data")
            self.assertEqual(result.method, "B")
            self.assertEqual(result.output, "  chosen artifact\n")
            self.assertEqual([a.score for a in result.assessments], [2, 7, 7])
            self.assertEqual(result.calls, 3)
            self.assertEqual(seen, ["input data"])
            for call, method in zip(provider.calls, methods):
                self.assertIn(task, call)
                self.assertIn("input data", call)
                self.assertIn(method.description, call)

    async def test_all_seven_default_paths(self):
        paths = {
            "Chain-of-Thoughts": (["answer"], ["chain_of_thought"], "answer"),
            "Tree-of-Thoughts": (
                ["first", "second", "third", '{"index": 1}'],
                ["propose"] * 3 + ["vote"],
                "second",
            ),
            "Analogical Prompting": (["answer"], ["analogical"], "answer"),
            "Self-Refine": (["draft", "revision"], ["draft", "revise"], "revision"),
            "Solo Performance Prompting": (["answer"], ["collaborate"], "answer"),
            "Step-Back Prompting": (
                ["principles", "answer"],
                ["abstract", "solve_with_principles"],
                "answer",
            ),
            "SimToM": (["known facts", "answer"], ["perspective", "solve_with_facts"], "answer"),
        }
        for name, (responses, operations, output) in paths.items():
            with self.subTest(method=name):
                agent = MRP("Task constraints", ScriptedProvider([]))
                agent.provider = ScriptedProvider(
                    ['{"score": 7}' if m.name == name else '{"score": 1}' for m in agent.methods]
                    + responses
                )
                result = await agent.run("original input")
                self.assertEqual(result.method, name)
                self.assertEqual(result.output, output)
                self.assertEqual(result.calls, 7 + len(responses))
                self.assertEqual([c["operation"] for c in agent.calls[7:]], operations)
                self.assertTrue(all("original input" in c for c in agent.provider.calls))
                if len(responses) == 2:
                    self.assertIn(responses[0], agent.provider.calls[-1])
                if name == "Tree-of-Thoughts":
                    self.assertEqual(len(set(agent.provider.calls[7:10])), 1)
                    self.assertTrue(all(c in agent.provider.calls[-1] for c in responses[:3]))

    async def test_custom_prompt_method_and_reset(self):
        provider = ScriptedProvider(['{"score": 5}', " first\n", '{"score": 3}', "second"])
        agent = MRP("Task", provider, methods=[Method("Custom", "Use supplied evidence only")])
        first = await agent.run("one")
        second = await agent.run("two")
        self.assertEqual(first.output, " first\n")
        self.assertEqual(second.output, "second")
        self.assertEqual(first.assessments[0].score, 5)
        self.assertEqual(second.calls, 2)
        self.assertIn("Use supplied evidence only", provider.calls[-1])
        self.assertNotIn("first", provider.calls[-1])

    async def test_invalid_scores_abort_and_preserve_raw_without_retry(self):
        for raw in (
            '{"score": 0}',
            '{"score": 8}',
            '{"score": true}',
            '{"score": "7"}',
            '{"score": 3.5}',
            "{}",
            "not json",
        ):
            agent = MRP("Task", ScriptedProvider([raw]), methods=[Method("A", "Description")])
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                await agent.run("Input")
            self.assertEqual(agent.calls[0]["response"], raw)
            self.assertIn("error", agent.calls[0])
            self.assertIsNone(agent.selected)

    async def test_execution_failures_keep_selection_and_call_evidence(self):
        for response in (" \n", TimeoutError("offline"), ("raw", ["tool request"])):
            agent = MRP(
                "Task",
                ScriptedProvider(['{"score": 4}', response]),
                methods=[Method("A", "Description")],
            )
            with self.assertRaises((ValueError, TimeoutError)):
                await agent.run("Input")
            self.assertEqual(agent.selected.name, "A")
            self.assertEqual(len(agent.calls), 2)
            self.assertIn("error", agent.calls[-1])
            self.assertIsNone(agent.output)

    async def test_partial_scoring_failure_never_executes_an_incumbent(self):
        agent = MRP(
            "Task",
            ScriptedProvider(['{"score": 7}', TimeoutError("offline")]),
            methods=[Method("A", "First"), Method("B", "Second")],
        )
        with self.assertRaises(TimeoutError):
            await agent.run("Input")
        self.assertEqual([(a.method, a.score) for a in agent.assessments], [("A", 7)])
        self.assertIsNone(agent.selected)
        self.assertEqual(len(agent.calls), 2)
        self.assertIn("error", agent.calls[-1])

    async def test_invalid_vote_is_rejected_before_indexing(self):
        for raw in ('{"index": -1}', '{"index": 3}', '{"index": true}'):
            agent = MRP("Task", ScriptedProvider(['{"score": 7}', "a", "b", "c", raw]))
            agent.methods = tuple(m for m in agent.methods if m.name == "Tree-of-Thoughts")
            with self.assertRaises(ValueError):
                await agent.run("Input")
            self.assertEqual(agent.calls[-1]["response"], raw)
            self.assertIn("error", agent.calls[-1])

    async def test_callback_failure_propagates(self):
        async def fail(input):
            raise RuntimeError("handler failed")

        agent = MRP(
            "Task",
            ScriptedProvider(['{"score": 7}']),
            methods=[Method("External", "Description", fail)],
        )
        with self.assertRaisesRegex(RuntimeError, "handler failed"):
            await agent.run("Input")
        self.assertEqual(agent.execution_error, "RuntimeError: handler failed")
        self.assertEqual(len(agent.calls), 1)

    async def test_templates_render_outside_repository_and_have_no_branches(self):
        agent = MRP("Task", ScriptedProvider([]))
        original = Path.cwd()
        self.addCleanup(os.chdir, original)
        os.chdir(self.templates.parent)
        rendered = await MRP.assess_method.render(agent, "Input", agent.methods[0])
        self.assertIn('"maximum": 7', rendered)
        self.assertEqual(rendered.count("# Output Format"), 1)
        for template in self.templates.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        await self.test_all_seven_default_paths()


if __name__ == "__main__":
    unittest.main()
