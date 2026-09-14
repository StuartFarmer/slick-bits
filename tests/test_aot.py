"""Check AoT generation boundaries and failure evidence without model calls."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from aot import AoT, Example
from tests.providers import ScriptedProvider


class AoTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "aot/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_search_is_one_call_and_evaluation_sees_only_final_artifact(self):
        for strategy in ("dfs", "bfs", "plans"):
            with self.subTest(strategy=strategy):
                seen = []

                async def evaluate(output):
                    seen.append(output)
                    return 0.75

                raw = "Branch A failed; branch B works.\nanswer:\n  chosen artifact\n"
                provider = ScriptedProvider([raw])
                agent = AoT(
                    "Design a delivery schedule",
                    provider,
                    evaluate,
                    examples=[Example("Old input", "Try A; prune A; recover via B", "Old answer")],
                )
                result = await agent.run("New input", strategy=strategy)
                self.assertEqual(result.output, "  chosen artifact\n")
                self.assertEqual(result.response, raw)
                self.assertEqual(result.calls, 1)
                self.assertEqual(result.evaluation, 0.75)
                self.assertEqual(seen, [result.output])
                self.assertEqual(agent.calls[0]["operation"], f"solve_{strategy}")
                for text in (agent.task, "Old input", "prune A", "Old answer", "New input"):
                    self.assertIn(text, provider.calls[0])

    async def test_warmup_is_one_extra_call_and_supplied_start_skips_it(self):
        provider = ScriptedProvider(["compatible candidates", "answer:\nfirst", "answer:\nsecond"])
        agent = AoT("Write an essay", provider, warmup_instructions="Find compatible themes")
        first = await agent.run("Input A", warmup=True)
        self.assertEqual(first.calls, 2)
        self.assertEqual(first.warm_start, "compatible candidates")
        self.assertIn("Find compatible themes", provider.calls[0])
        self.assertIn("compatible candidates", provider.calls[1])
        second = await agent.run("Input B", warmup=True, warm_start="chosen theme")
        self.assertEqual(second.calls, 1)
        self.assertEqual(second.warm_start, "chosen theme")
        self.assertIn("chosen theme", provider.calls[-1])
        self.assertNotIn("compatible candidates", provider.calls[-1])
        self.assertIsNone(second.evaluation)

    async def test_invalid_output_aborts_without_retry_and_preserves_raw(self):
        for raw in (" ", "unfinished search", "answer:\n \n"):
            agent = AoT("Task", ScriptedProvider([raw]))
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                await agent.run("Input")
            self.assertEqual(len(agent.calls), 1)
            self.assertEqual(agent.calls[0]["response"], raw)
            self.assertIn("error", agent.calls[0])
            self.assertIsNone(agent.output)

    async def test_answer_delimiter_preserves_multiline_artifacts(self):
        raw = "Candidate answer: discard this\r\nANSWER: \r\n    first\r\n\r\n  second\r\n"
        agent = AoT("Preserve indentation", ScriptedProvider([raw]))
        result = await agent.run("Input")
        self.assertEqual(result.output, "    first\r\n\r\n  second\r\n")
        self.assertEqual(agent.calls[0]["operation"], "solve_plans")

    async def test_provider_and_warmup_failures_preserve_partial_progress(self):
        for response in (TimeoutError("offline"), ("raw", ["tool request"])):
            agent = AoT("Task", ScriptedProvider(["start", response]))
            with self.assertRaises((TimeoutError, ValueError)):
                await agent.run("Input", warmup=True)
            self.assertEqual(agent.warm_start, "start")
            self.assertEqual(len(agent.calls), 2)
            self.assertIn("error", agent.calls[-1])
        agent = AoT("Task", ScriptedProvider([" "]))
        with self.assertRaises(ValueError):
            await agent.run("Input", warmup=True)
        self.assertEqual(len(agent.calls), 1)

    async def test_evaluator_failure_preserves_answer_without_regeneration(self):
        for value in (float("nan"), float("inf"), RuntimeError("evaluator failed")):

            async def evaluate(output):
                if isinstance(value, Exception):
                    raise value
                return value

            agent = AoT("Task", ScriptedProvider(["answer:\nartifact"]), evaluate)
            with self.subTest(value=value), self.assertRaises((ValueError, RuntimeError)):
                await agent.run("Input")
            self.assertEqual(agent.output, "artifact")
            self.assertEqual(len(agent.calls), 1)
            self.assertIsNotNone(agent.evaluation_error)

    async def test_templates_render_from_other_directory_without_branches(self):
        agent = AoT("Task", ScriptedProvider([]))
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.templates.parent)
        for operation in (AoT.solve_dfs, AoT.solve_bfs, AoT.solve_plans):
            rendered = await operation.render(agent, "Input", "Start")
            self.assertIn("Task", rendered)
            self.assertIn("Input", rendered)
            self.assertIn("Start", rendered)
        self.assertIn("Input", await AoT.prepare.render(agent, "Input"))
        for template in self.templates.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        self.assertEqual(agent.calls, [])


if __name__ == "__main__":
    unittest.main()
