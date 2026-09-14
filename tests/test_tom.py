"""Offline checks for geometric alignment and the judge/editor loop."""

import asyncio
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from tests.providers import ScriptedProvider
from tom import Dimension, Judgement, Profile, TheoryOfMind, alignment


class TheoryOfMindTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "tom/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)
        self.dimensions = (Dimension("accuracy", "Agreement with supplied facts"),)

    def test_loss_uses_volume_and_vertex_distance_including_zero_targets(self):
        result = alignment({"a": 50, "b": 100}, {"a": 100, "b": 100})
        self.assertEqual(result.area, 0.5)
        self.assertEqual(result.expected_area, 1)
        self.assertEqual(result.tma, 0.5)
        self.assertEqual(result.tmd, 0.25)
        self.assertEqual(result.loss, 0.4375)
        swapped = alignment({"a": 100, "b": 50}, {"a": 50, "b": 100})
        self.assertEqual(swapped.tma, 0)
        self.assertEqual(swapped.loss, 0.5)
        self.assertEqual(alignment({"a": 0}, {"a": 0}).loss, 0)
        self.assertAlmostEqual(alignment({"a": 10}, {"a": 0}).loss, 0.1275)

    def test_editor_profile_averages_only_human_observations_and_scales_covariance(self):
        profile = Profile({"a": 80, "b": 90})
        self.assertEqual(profile.weights(), {"a": 1, "b": 1})
        profile.observe({"a": 0, "b": 0})
        profile.observe({"a": 100, "b": 100})
        self.assertEqual(profile.targets, {"a": 50, "b": 50})
        self.assertEqual(profile.weights(), {"a": 0.5, "b": 0.5})
        result = alignment({"a": 100, "b": 100}, profile.targets, profile.weights())
        self.assertEqual(result.expected_area, 0.0625)
        self.assertEqual(result.tmd, 0.25)

    async def test_generate_judge_metaprompt_regenerate_and_converge(self):
        for task in ("Describe a museum exhibit", "Write a Python function"):
            generator = ScriptedProvider(["first draft", "  improved draft\n"])
            judge = ScriptedProvider(
                [
                    Judgement(scores={"accuracy": 50}, feedback="Missing source facts"),
                    Judgement(scores={"accuracy": 100}, feedback="Matches the source"),
                ]
            )
            editor = ScriptedProvider(["Use every relevant source fact."])
            profile = Profile({"accuracy": 100})
            agent = TheoryOfMind(
                task,
                generator,
                self.dimensions,
                profile,
                context="source marker",
                judge_provider=judge,
                editor_provider=editor,
            )
            result = await agent.run()
            self.assertEqual(result.stop_reason, "converged")
            self.assertEqual(result.best.content, "  improved draft\n")
            self.assertEqual(result.best.instruction, "Use every relevant source fact.")
            self.assertEqual(len(result.history), 2)
            self.assertEqual(
                [r["operation"] for r in agent.calls],
                ["generate", "judge", "rewrite_prompt", "regenerate", "judge"],
            )
            self.assertIn("50 percentage points below", editor.calls[0])
            self.assertIn("increase", editor.calls[0])
            self.assertIn("first draft", generator.calls[1])
            self.assertIn("Use every relevant source fact.", generator.calls[1])
            for request in generator.calls + judge.calls + editor.calls:
                self.assertIn(task, request)
                self.assertIn("source marker", request)
            self.assertEqual(profile.samples, [])

    async def test_budget_returns_best_but_revisions_follow_latest(self):
        provider = ScriptedProvider(["80", "instruction 2", "20", "instruction 3", "30"])

        async def evaluate(content):
            return Judgement(scores={"accuracy": float(content)}, feedback="Measured")

        agent = TheoryOfMind(
            "Task",
            provider,
            self.dimensions,
            Profile({"accuracy": 100}),
            evaluate=evaluate,
            max_iterations=3,
        )
        result = await agent.run()
        self.assertEqual(result.stop_reason, "iterations")
        self.assertEqual(result.best.content, "80")
        self.assertEqual([step.content for step in result.history], ["80", "20", "30"])
        self.assertIn('"content": "20"', provider.calls[3])
        self.assertEqual(len(provider.calls), 5)
        self.assertEqual(len(agent.calls), 8)

    async def test_learning_uses_judge_and_updates_profile_only_after_valid_scores(self):
        provider = ScriptedProvider(
            [
                Judgement(scores={"accuracy": 60}, feedback="Human edit one"),
                Judgement(scores={"accuracy": 80}, feedback="Human edit two"),
                Judgement(scores={"unknown": 90}, feedback="Wrong dimension"),
            ]
        )
        profile = Profile({"accuracy": 100})
        agent = TheoryOfMind("Task", provider, self.dimensions, profile)
        await agent.learn("human edit one")
        await agent.learn("human edit two")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            await agent.learn("invalid")
        self.assertEqual(profile.targets, {"accuracy": 70})
        self.assertEqual(len(profile.samples), 2)
        self.assertIn("unknown", agent.calls[-1]["response"])
        self.assertIn("error", agent.calls[-1])

    async def test_bad_generated_outputs_abort_without_retry_and_retain_raw_response(self):
        for response in (
            "not json",
            '{"scores":{"accuracy":101},"feedback":"bad"}',
            '{"scores":{"accuracy":NaN},"feedback":"bad"}',
            '{"scores":{"accuracy":true},"feedback":"bad"}',
            '{"scores":{},"feedback":"missing"}',
            '{"scores":{"other":50},"feedback":"bad"}',
        ):
            provider = ScriptedProvider(["draft", response])
            agent = TheoryOfMind("Task", provider, self.dimensions, Profile({"accuracy": 100}))
            with self.subTest(response=response), self.assertRaises(ValueError):
                await agent.run()
            self.assertEqual(len(provider.calls), 2)
            self.assertEqual(agent.calls[-1]["response"], response)
            self.assertIn("error", agent.calls[-1])

    async def test_four_dimensions_preserve_targets_and_reduce_undesired_trait(self):
        targets = {"accuracy": 100, "novelty": 50, "repetition": 0, "relevance": 100}
        dimensions = tuple(Dimension(name, f"Degree of {name}") for name in targets)
        provider = ScriptedProvider(
            [
                "draft",
                Judgement(scores={**targets, "repetition": 10}, feedback="One repeated point"),
                "Remove the repeated point.",
                "revised",
                Judgement(scores=targets, feedback="Targets met"),
            ]
        )
        agent = TheoryOfMind("Task", provider, dimensions, Profile(targets), threshold=0.001)
        result = await agent.run()
        self.assertEqual(result.stop_reason, "converged")
        self.assertIn("decrease repetition", provider.calls[2])
        self.assertIn("novelty matches expectations (50); preserve it", provider.calls[2])
        self.assertEqual(result.best.alignment.loss, 0)

    async def test_threshold_is_strict_and_one_iteration_has_no_editor_call(self):
        provider = ScriptedProvider(
            ["draft", Judgement(scores={"a": 10, "b": 0}, feedback="Difference is 10 points")]
        )
        agent = TheoryOfMind(
            "Task",
            provider,
            (Dimension("a", "A"), Dimension("b", "B")),
            Profile({"a": 0, "b": 0}),
            max_iterations=1,
        )
        result = await agent.run()
        self.assertEqual(result.best.alignment.loss, 0.05)
        self.assertEqual(result.stop_reason, "iterations")
        self.assertEqual(len(provider.calls), 2)

    async def test_evaluator_rejection_and_tool_requests_are_logged_and_propagated(self):
        async def evaluate(content):
            return Judgement.model_construct(scores={"accuracy": float("inf")}, feedback="bad")

        provider = ScriptedProvider(["draft"])
        agent = TheoryOfMind(
            "Task", provider, self.dimensions, Profile({"accuracy": 100}), evaluate=evaluate
        )
        with self.assertRaises(ValueError):
            await agent.run()
        self.assertEqual(agent.history, [])
        self.assertIn("error", agent.calls[-1])
        with patch.object(provider, "acall", return_value=("response", [object()])):
            with self.assertRaisesRegex(ValueError, "tool requests"):
                await agent.run()
        self.assertEqual(agent.calls[-1]["response"], "response")
        self.assertIn("error", agent.calls[-1])
        for responses in ([" \n"], ["draft", OSError("offline")]):
            agent = TheoryOfMind(
                "Task", ScriptedProvider(responses), self.dimensions, Profile({"accuracy": 100})
            )
            with self.assertRaises((ValueError, OSError)):
                await agent.run()
            self.assertIn("error", agent.calls[-1])

    async def test_timeout_retains_only_completed_evaluations_and_cancels_pending_work(self):
        provider = ScriptedProvider(["first", "revise", "second"])
        cancelled = []

        async def evaluate(content):
            if content == "second":
                try:
                    await asyncio.sleep(10)
                finally:
                    cancelled.append(content)
            return Judgement(scores={"accuracy": 20}, feedback="Low")

        agent = TheoryOfMind(
            "Task",
            provider,
            self.dimensions,
            Profile({"accuracy": 100}),
            evaluate=evaluate,
            timeout=0.05,
        )
        result = await agent.run()
        self.assertEqual(result.stop_reason, "timeout")
        self.assertEqual(result.best.content, "first")
        self.assertEqual(len(result.history), 1)
        self.assertEqual(cancelled, ["second"])
        self.assertIn("error", agent.calls[-1])
        agent.timeout = 0
        result = await agent.run()
        self.assertIsNone(result.best)
        self.assertEqual(agent.calls, [])

    async def test_templates_render_from_other_directory_and_have_no_control_logic(self):
        agent = TheoryOfMind(
            "task marker", ScriptedProvider([]), self.dimensions, Profile({"accuracy": 100})
        )
        previous = Path.cwd()
        try:
            os.chdir(self.templates)
            for operation, args in (
                (TheoryOfMind.generate, ("instruction",)),
                (TheoryOfMind.regenerate, ("instruction", "draft")),
                (TheoryOfMind.judge, ("draft",)),
                (TheoryOfMind.rewrite_prompt, ({"content": "draft"}, ["feedback"])),
            ):
                self.assertIn("task marker", await operation.render(agent, *args))
            for template in self.templates.glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
