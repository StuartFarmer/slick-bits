"""Exercise the Meta-Prompting loop through Slick without paid generation."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from pydantic import ValidationError

from meta_prompting import Audit, Case, Example, MetaPrompting
from tests.providers import ScriptedProvider


class MetaPromptingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "meta_prompting/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_best_of_n_revision_examples_and_blind_auditing(self):
        generator = ScriptedProvider(
            ["bad", "good", "gold", "gold", "fixed", "fixed", "gold", "gold"]
        )
        auditor = ScriptedProvider(
            [
                Audit(score=0.2, critique="Missing detail"),
                Audit(score=1, critique="Pass"),
                Audit(score=0.8, critique="Incomplete"),
                Audit(score=0.8, critique="Incomplete"),
                Audit(score=1, critique="Pass"),
                Audit(score=1, critique="Pass"),
                Audit(score=1, critique="Pass"),
                Audit(score=1, critique="Pass"),
            ]
        )
        optimizer = ScriptedProvider(
            [
                '{"groups": [{"critique": "Require the missing detail", "members": [0]}]}',
                "revised-private-instruction",
            ]
        )
        agent = MetaPrompting(
            "Summarize material", generator, optimizer, rules=["Be complete"], auditor=auditor
        )
        result = await agent.run(
            "private-instruction",
            [Case("train", reference="train-answer")],
            [Case("gold-input", reference="gold-answer")],
            best_of_n=2,
            threshold=1,
            max_iterations=1,
            anchors=[Example("anchor", "human-answer")],
        )
        self.assertEqual(result.instruction.text, "revised-private-instruction")
        self.assertEqual(result.stop_reason, "threshold")
        self.assertEqual(result.train.score, 1)
        self.assertEqual(result.gold.score, 1)
        self.assertEqual(
            result.instruction.examples,
            (Example("anchor", "human-answer"), Example("train", "good")),
        )
        self.assertEqual(len(agent.calls), 18)
        self.assertEqual(len(agent.evaluations), 8)
        self.assertEqual([trial.accepted for trial in result.history], [True, True])
        for context in auditor.calls:
            self.assertNotIn("private-instruction", context)
            self.assertNotIn("human-answer", context)
        for context in optimizer.calls:
            self.assertNotIn("gold-input", context)
            self.assertNotIn("gold-answer", context)
        self.assertIn("train-answer", auditor.calls[0])
        self.assertNotIn("train-answer", generator.calls[0])
        self.assertIn("human-answer", generator.calls[0])
        self.assertNotIn("<output>good</output>", generator.calls[4])
        self.assertIn("<output>good</output>", generator.calls[6])

    async def test_golden_regression_rolls_back_even_when_mean_improves(self):
        generator = ScriptedProvider(["0.2", "1", "0.4", "0.9", "0.9", "1"])
        optimizer = ScriptedProvider(
            ['{"groups": [{"critique": "Improve", "members": [0]}]}', "bad revision"]
        )

        async def evaluate(case, output):
            return Audit(score=float(output), critique="Measured")

        agent = MetaPrompting("Any task", generator, optimizer, rules=[], evaluate=evaluate)
        result = await agent.run(
            "initial", [Case("train")], [Case("g1"), Case("g2")], best_of_n=1, max_iterations=1
        )
        self.assertEqual(result.instruction.text, "initial")
        self.assertEqual(result.train.score, 0.2)
        self.assertFalse(result.history[-1].accepted)
        self.assertEqual(result.history[-1].reason, "gold_regression")
        self.assertGreater(result.history[-1].gold.score, result.gold.score)

    async def test_threshold_zero_budget_and_no_training_gradients(self):
        for task in ("Classify a message", "Design a schedule"):
            for scores, budget, expected in (
                ([1, 1], 3, "threshold"),
                ([0.5, 0.5], 0, "budget"),
                ([1, 0.5], 3, "no_gradients"),
            ):
                with self.subTest(task=task, expected=expected):
                    measured = iter(scores)

                    async def evaluate(case, output):
                        return Audit(score=next(measured), critique="Measured")

                    generator = ScriptedProvider(["answer", "gold output"])
                    optimizer = ScriptedProvider([])
                    agent = MetaPrompting(task, generator, optimizer, rules=[], evaluate=evaluate)
                    result = await agent.run(
                        "initial",
                        [Case("train")],
                        [Case("gold")],
                        best_of_n=1,
                        max_iterations=budget,
                    )
                    self.assertEqual(result.stop_reason, expected)
                    self.assertEqual(result.calls, 2)
                    self.assertEqual(result.evaluations, 2)
                    self.assertEqual(len(result.history), 1)
                    self.assertEqual(optimizer.calls, [])

    async def test_review_rejection_then_acceptance_and_history(self):
        generator = ScriptedProvider(["0.2", "0.8", "0.7", "0.8"])
        groups = '{"groups": [{"critique": "Fix detail", "members": [0]}]}'
        optimizer = ScriptedProvider([groups, "rejected", groups, "accepted"])
        reviewed = []

        async def evaluate(case, output):
            return Audit(score=float(output), critique="Measured")

        async def review(old, new):
            reviewed.append((old.text, new.text))
            return new.text == "accepted"

        agent = MetaPrompting(
            "Task", generator, optimizer, rules=[], evaluate=evaluate, review=review
        )
        result = await agent.run(
            "initial", [Case("train")], [Case("gold-secret")], best_of_n=1, max_iterations=2
        )
        self.assertEqual(reviewed, [("initial", "rejected"), ("initial", "accepted")])
        self.assertEqual(result.instruction.text, "accepted")
        self.assertEqual(result.stop_reason, "budget")
        self.assertEqual(result.calls, 8)
        self.assertIsNone(result.history[1].train)
        self.assertIn("review_rejected", optimizer.calls[-1])
        self.assertNotIn("gold-secret", optimizer.calls[-1])
        self.assertEqual(result.history[2].train.score, 0.7)

    async def test_train_regression_and_ties(self):
        for next_score, accepted in ((0.4, False), (0.5, True)):
            generator = ScriptedProvider(["0.5", "0.8", str(next_score), "0.8"])
            optimizer = ScriptedProvider(
                ['{"groups": [{"critique": "Improve", "members": [0]}]}', "revision"]
            )

            async def evaluate(case, output):
                return Audit(score=float(output), critique="Measured")

            agent = MetaPrompting("Task", generator, optimizer, rules=[], evaluate=evaluate)
            result = await agent.run(
                "initial", [Case("train")], [Case("gold")], best_of_n=1, max_iterations=1
            )
            self.assertEqual(result.history[-1].accepted, accepted)
            self.assertEqual(result.instruction.text, "revision" if accepted else "initial")
            self.assertEqual(
                result.history[-1].reason, "accepted" if accepted else "train_regression"
            )

    async def test_cluster_partition_and_frequency(self):
        failures = [{"critique": "failure"}] * 3
        for members in (([0], [1, 2]), ([0], [1, 1]), ([0], [1]), ([0], [1, 3])):
            import json

            response = json.dumps(
                {
                    "groups": [
                        {"critique": "singleton", "members": members[0]},
                        {"critique": "repeated", "members": members[1]},
                    ]
                }
            )
            agent = MetaPrompting(
                "Task", ScriptedProvider([]), ScriptedProvider([response]), rules=[]
            )
            if members == ([0], [1, 2]):
                groups = await agent._invoke(agent.aggregate, "optimizer", failures)
                self.assertEqual([g.members for g in groups], [(1, 2), (0,)])
            else:
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    await agent._invoke(agent.aggregate, "optimizer", failures)
                self.assertEqual(agent.calls[-1]["response"], response)
                self.assertIn("error", agent.calls[-1])

    async def test_malformed_audits_and_measured_scores_abort_without_retry(self):
        for raw in (
            "not JSON",
            '{"score": 2, "critique": "bad"}',
            '{"score": true, "critique": "bad"}',
            '{"score": 0.5, "critique": " "}',
        ):
            auditor = ScriptedProvider([raw])
            agent = MetaPrompting(
                "Task",
                ScriptedProvider(["artifact"]),
                ScriptedProvider([]),
                rules=[],
                auditor=auditor,
            )
            with self.assertRaises(ValidationError):
                await agent.run("initial", [Case("train")], [Case("gold")], best_of_n=1)
            self.assertEqual(agent.calls[-1]["response"], raw)
            self.assertIn("error", agent.calls[-1])
            self.assertIn("error", agent.evaluations[-1])
            self.assertEqual(len(auditor.calls), 1)
            self.assertEqual(agent.instruction.text, "initial")

        for score in (float("nan"), float("inf"), -0.1, 1.1):

            async def evaluate(case, output):
                return Audit.model_construct(score=score, critique="bad measurement")

            agent = MetaPrompting(
                "Task",
                ScriptedProvider(["artifact"]),
                ScriptedProvider([]),
                rules=[],
                evaluate=evaluate,
            )
            with self.assertRaises(ValidationError):
                await agent.run("initial", [Case("train")], [Case("gold")], best_of_n=1)
            self.assertIn("error", agent.evaluations[-1])

    async def test_generation_and_callback_failures_leave_partial_progress(self):
        for response, expected_error in (
            (" \n", ValueError),
            (TimeoutError("offline"), TimeoutError),
            (("raw", [{"name": "tool"}]), ValueError),
        ):
            agent = MetaPrompting(
                "Task", ScriptedProvider([response]), ScriptedProvider([]), rules=[]
            )
            with self.assertRaises(expected_error):
                await agent.run("initial", [Case("train")], [Case("gold")], best_of_n=1)
            self.assertEqual(len(agent.calls), 1)
            self.assertIn("error", agent.calls[0])
            self.assertEqual(agent.evaluations, [])

        async def evaluate(case, output):
            raise RuntimeError("evaluator failed")

        agent = MetaPrompting(
            "Task",
            ScriptedProvider(["artifact"]),
            ScriptedProvider([]),
            rules=[],
            evaluate=evaluate,
        )
        with self.assertRaisesRegex(RuntimeError, "evaluator failed"):
            await agent.run("initial", [Case("train")], [Case("gold")], best_of_n=1)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(agent.evaluations[0]["output"], "artifact")
        self.assertIn("error", agent.evaluations[0])

    async def test_reset_and_inference_preserve_artifact_bytes_and_reference_separation(self):
        async def evaluate(case, output):
            return Audit(score=1, critique="Pass")

        agent = MetaPrompting(
            "Task",
            ScriptedProvider(["one", "gold", "two", "gold", "  result\n"]),
            ScriptedProvider([]),
            rules=[],
            evaluate=evaluate,
        )
        first = await agent.run("first", [Case("train")], [Case("gold")], best_of_n=1)
        second = await agent.run("second", [Case("train")], [Case("gold")], best_of_n=1)
        self.assertEqual(first.instruction.text, "first")
        self.assertEqual(second.instruction.text, "second")
        self.assertEqual(len(agent.history), 1)
        output = await agent.predict(second.instruction, Case("new", reference="secret"))
        self.assertEqual(output, "  result\n")
        self.assertNotIn("secret", agent.calls[-1]["prompt"])
        self.assertEqual(second.calls, 2)

    async def test_templates_render_from_another_working_directory(self):
        from meta_prompting import Instruction
        from meta_prompting.agent import Gradient

        agent = MetaPrompting("Task", ScriptedProvider([]), ScriptedProvider([]), rules=["Rule"])
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.templates.parent)
        failures = [{"critique": "Fix", "output": "Bad"}]
        rendered = (
            await MetaPrompting.generate.render(agent, "Instruction", Case("Input"), []),
            await MetaPrompting.audit.render(agent, Case("Input", reference="Reference"), "Output"),
            await MetaPrompting.aggregate.render(agent, failures),
            await MetaPrompting.revise.render(
                agent,
                Instruction("Instruction"),
                [Gradient(critique="Fix", members=(0,))],
                failures,
                [],
            ),
        )
        self.assertTrue(all(rendered))
        self.assertIn('"properties"', rendered[1])
        self.assertIn('"properties"', rendered[2])
        for template in self.templates.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        self.assertEqual(agent.calls, [])


if __name__ == "__main__":
    unittest.main()
