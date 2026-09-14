"""Exercise STaR's model reset, filtering, and hint-free training offline."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from star import Demonstration, Example, Solution, STaR
from tests.providers import ScriptedProvider


async def exact(input, answer, expected):
    return answer == expected


class STaRTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "star/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_two_rounds_reset_base_and_replace_training_data(self):
        examples = [Example("input-one", "gold-one"), Example("input-two", "gold-two")]
        demo = Demonstration("seed-input", "seed-rationale", "seed-answer")
        base = ScriptedProvider(
            [
                Solution(rationale="direct-one", answer="gold-one"),
                Solution(rationale="wrong-two", answer="wrong"),
                Solution(rationale="recovered-two", answer="gold-two"),
            ]
        )
        updated = ScriptedProvider(
            [
                Solution(rationale="new-one", answer="gold-one"),
                Solution(rationale="new-two", answer="gold-two"),
            ]
        )
        final = ScriptedProvider([])
        batches = []

        async def train(original, training, iteration):
            self.assertIs(original, base)
            batches.append(training)
            return [updated, final][iteration - 1]

        agent = STaR("Classify arbitrary inputs", base, exact, train, demonstrations=[demo])
        result = await agent.run(examples, iterations=2)
        self.assertIs(result.provider, final)
        self.assertEqual(result.stop_reason, "budget")
        self.assertEqual([(r.generated, r.rationalized) for r in result.history], [(1, 1), (2, 0)])
        self.assertEqual(len(agent.calls), 5)
        self.assertEqual([len(p.calls) for p in (base, updated, final)], [3, 2, 0])
        self.assertEqual([len(batch) for batch in batches], [2, 2])
        self.assertEqual([row.source for row in batches[0]], ["generate", "rationalize"])
        self.assertNotIn("gold-one", base.calls[0])
        self.assertNotIn("gold-two", base.calls[1])
        self.assertIn("gold-two", base.calls[2])
        for batch in batches:
            for row in batch:
                self.assertNotIn("Correct answer hint:", row.prompt)
                self.assertNotIn(row.example.answer, row.prompt)
                self.assertIn("seed-rationale", row.prompt)
                parsed = Solution.model_validate_json(row.completion)
                self.assertEqual(parsed.answer, row.example.answer)
        self.assertEqual(batches[0][1].prompt, base.calls[1])
        self.assertNotIn("recovered-two", batches[1][1].completion)

    async def test_filter_failed_rationalizations_and_disable_ablation(self):
        for rationalization in (True, False):
            with self.subTest(rationalization=rationalization):
                provider = ScriptedProvider(
                    [
                        Solution(rationale="unsolved", answer="wrong"),
                        Solution(rationale="still unsolved", answer="wrong-again"),
                    ]
                )

                async def train(*args):
                    self.fail("Cannot train on an empty accepted set")

                agent = STaR("Any task", provider, exact, train)
                result = await agent.run(
                    [Example("input", "gold")], rationalization=rationalization
                )
                self.assertEqual(result.stop_reason, "no_training_examples")
                self.assertIs(result.provider, provider)
                self.assertEqual(agent.training, ())
                self.assertEqual(len(agent.calls), 2 if rationalization else 1)
                self.assertTrue(all(call["correct"] is False for call in agent.calls))

    async def test_custom_checker_and_nonaccumulating_filtered_batch(self):
        provider = ScriptedProvider(
            [
                Solution(rationale="valid", answer="CANONICAL"),
                Solution(rationale="bad", answer="wrong"),
                Solution(rationale="bad hint", answer="wrong"),
            ]
        )
        seen = []

        async def evaluate(input, answer, expected):
            seen.append((input, answer, expected))
            return answer.lower() == expected

        async def train(base, rows, iteration):
            self.assertEqual(len(rows), 1)
            self.assertEqual(Solution.model_validate_json(rows[0].completion).answer, "canonical")
            return base

        agent = STaR("Any task", provider, evaluate, train)
        result = await agent.run([Example("x", "canonical"), Example("y", "gold")], iterations=1)
        self.assertEqual(len(seen), 3)
        self.assertEqual(result.history[0].generated, 1)
        self.assertEqual(result.history[0].rationalized, 0)

    async def test_failures_propagate_with_raw_generation_records(self):
        for response in (
            "not json",
            '{"rationale":" ","answer":"a"}',
            '{"rationale":"r","answer":" "}',
            '{"rationale":"r","answer":"a","extra":true}',
            ('{"rationale":"r","answer":"a"}', [{"name": "unexpected"}]),
            RuntimeError("transport failed"),
        ):
            with self.subTest(response=response):
                provider = ScriptedProvider([response])

                async def train(*args):
                    self.fail("Invalid generation reached training")

                agent = STaR("Any task", provider, exact, train)
                with self.assertRaises((ValueError, RuntimeError)):
                    await agent.run([Example("x", "a")])
                self.assertEqual(len(agent.calls), 1)
                self.assertIn("error", agent.calls[0])
                if not isinstance(response, Exception):
                    self.assertEqual(
                        agent.calls[0]["response"],
                        response[0] if isinstance(response, tuple) else response,
                    )

    async def test_callback_failures_are_not_retried(self):
        for phase in ("evaluate", "train"):

            async def evaluate(*args):
                if phase == "evaluate":
                    raise RuntimeError("checker failed")
                return True

            async def train(*args):
                raise RuntimeError("trainer failed")

            agent = STaR(
                "Task", ScriptedProvider([Solution(rationale="r", answer="a")]), evaluate, train
            )
            with self.assertRaises(RuntimeError):
                await agent.run([Example("x", "a")])
            self.assertEqual(len(agent.calls), 1)
            self.assertEqual(agent.history, [])
            if phase == "train":
                self.assertEqual(len(agent.training), 1)

    async def test_zero_budget_empty_dataset_and_all_templates_from_other_directory(self):
        async def train(*args):
            self.fail("No training expected")

        agent = STaR("Task", ScriptedProvider([]), exact, train)
        result = await agent.run([Example("x", "a")], iterations=0)
        self.assertEqual(result.history, ())
        self.assertEqual(agent.calls, [])
        result = await agent.run([])
        self.assertEqual(result.stop_reason, "no_training_examples")
        old_cwd = Path.cwd()
        try:
            os.chdir(self.templates.parent)
            normal = await STaR.generate.render(agent, "input")
            hinted = await STaR.rationalize.render(agent, "input", "secret")
        finally:
            os.chdir(old_cwd)
        self.assertNotIn("secret", normal)
        self.assertIn("secret", hinted)
        for text in (normal, hinted):
            self.assertIn("# Output Format", text)
            self.assertIn('"required": ["rationale", "answer"]', text)
        for template in self.templates.glob("*.j2"):
            tree = Environment().parse(template.read_text())
            self.assertEqual(list(tree.find_all((nodes.If, nodes.CondExpr))), [])
