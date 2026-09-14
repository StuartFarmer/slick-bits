"""Check EvoPrompting's budgets, parent retirement, and tuning boundary offline."""

import asyncio
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts

from tests.providers import ScriptedProvider


class EvoPromptingChecks(unittest.TestCase):
    def test_bound_template_from_another_launch_directory(self):
        from evoprompting import Evaluation, EvoPrompting, Individual

        async def check():
            agent = EvoPrompting("Arrange a seminar.", None, None, tune=None)
            parent = Individual("first candidate", Evaluation(0.2, 100), -20)
            rendered = await EvoPrompting.crossmut.render(agent, (parent,), {"cost": 90})
            self.assertIn(agent.task, rendered)
            self.assertLess(rendered.index("first candidate"), rendered.index('"cost": 90'))
            self.assertEqual(rendered.count('"""Metrics:'), 2)

        root = Path(__file__).resolve().parents[1] / "evoprompting/prompts"
        cwd = Path.cwd()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prompts, "TEMPLATE_ROOT", root),
        ):
            try:
                os.chdir(directory)
                asyncio.run(check())
            finally:
                os.chdir(cwd)

    def test_multiple_prompts_and_empty_training_skip(self):
        from evoprompting import Evaluation, EvoPrompting

        async def check():
            provider = ScriptedProvider(["a", "b", "c", "d", "e", "f", "g", "h"])

            async def evaluate(content):
                return Evaluation(0.1, 100)

            async def tune(*args):
                self.fail("all accepted children were selected; training must be skipped")

            result = await EvoPrompting(
                "Plan a seminar.",
                lambda temperature: provider,
                evaluate,
                tune=tune,
                rounds=2,
                prompts_per_round=2,
                samples_per_prompt=2,
                survivors=4,
            ).run(["seed"])
            self.assertEqual(len(result.attempts), 8)
            self.assertEqual(result.evaluations, 9)
            self.assertTrue(all(not r.training for r in result.history))
            for i in range(0, 8, 2):
                self.assertEqual(provider.calls[i], provider.calls[i + 1])

        with patch.object(
            prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[1] / "evoprompting/prompts"
        ):
            asyncio.run(check())

    def test_algorithm(self):
        self.assertIsNotNone(importlib.util.find_spec("evoprompting"), "implement EvoPrompting")
        from evoprompting import Evaluation, EvoPrompting

        async def check():
            original = ScriptedProvider(["a", "b", "c"])
            tuned = ScriptedProvider(["d", "e", "f", "g", "h", "i"])
            temperatures, tuning = [], []

            def factory(temperature):
                temperatures.append(temperature)
                return original

            def next_factory(temperature):
                temperatures.append(temperature)
                return tuned

            async def tune(current, children, settings):
                tuning.append((current, tuple(c.content for c in children), settings))
                return next_factory

            async def evaluate(content):
                return Evaluation(error=0.1, cost={"seed": 1, "a": 2, "b": 3}.get(content, 10))

            agent = EvoPrompting(
                "Improve a workshop plan.",
                factory,
                evaluate,
                tune=tune,
                rounds=3,
                prompts_per_round=1,
                samples_per_prompt=3,
                survivors=1,
            )
            result = await agent.run(["seed"])
            self.assertEqual(len(result.attempts), 9)
            self.assertEqual(result.evaluations, 10)
            self.assertEqual(result.best.content, "a")
            self.assertEqual(result.top[0].content, "c")
            self.assertEqual([r.parents[0].content for r in result.history], ["seed", "a", "b"])
            self.assertEqual([x[1] for x in tuning], [("b", "c"), ("d", "e", "f")])
            self.assertIs(tuning[0][0], factory)
            self.assertIs(tuning[1][0], next_factory)
            self.assertEqual(tuning[0][2].epochs, 5)
            self.assertEqual(tuning[0][2].prompt_length, 16)
            self.assertEqual(tuning[0][2].batch_size, 16)
            self.assertEqual(tuning[0][2].learning_rate, 0.1)
            self.assertTrue(set(temperatures) <= {0.2, 0.6, 0.8, 1.0})
            self.assertEqual(len(temperatures), 9)
            self.assertTrue(all(agent.task in call for call in original.calls + tuned.calls))
            self.assertEqual(len(result.attempts[0].parents), 2)  # sampling with replacement
            self.assertIs(result.provider, next_factory)

        with patch.object(
            prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[1] / "evoprompting/prompts"
        ):
            asyncio.run(check())

    def test_rejections_budget_and_raw_text(self):
        from evoprompting import CandidateRejected, Evaluation, EvoPrompting

        async def check():
            code = "    return value\n\n"
            provider = ScriptedProvider(["seed", "bad", "bad", "threshold", "nan", " ", code])
            evaluated = []

            async def evaluate(content):
                evaluated.append(content)
                if content == "bad":
                    raise CandidateRejected("does not satisfy the task interface")
                if content == "threshold":
                    return Evaluation(error=0.5, cost=1)
                if content == "nan":
                    return Evaluation(error=float("nan"), cost=1)
                return Evaluation(error=0.1, cost=1)

            result = await EvoPrompting(
                "Rewrite a function body.",
                lambda temperature: provider,
                evaluate,
                tune=None,
                rounds=1,
                prompts_per_round=1,
                samples_per_prompt=7,
            ).run(["seed", "seed"])
            self.assertEqual(result.evaluations, 5)
            self.assertEqual(evaluated, ["seed", "bad", "threshold", "nan", code])
            self.assertEqual(
                [a.status for a in result.attempts],
                [
                    "duplicate",
                    "rejected",
                    "duplicate",
                    "filtered",
                    "rejected",
                    "rejected",
                    "accepted",
                ],
            )
            self.assertEqual(result.attempts[5].raw, " ")
            self.assertEqual(result.best.content, code)
            self.assertEqual(result.stop_reason, "rounds")

        with patch.object(
            prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[1] / "evoprompting/prompts"
        ):
            asyncio.run(check())

    def test_custom_metrics_and_empty_pool(self):
        from evoprompting import Evaluation, EvoPrompting, Individual, paper_targets

        async def check():
            async def evaluate(content):
                return Evaluation(error=0.1, cost=100, metrics={"latency": 3}, fitness=42)

            provider = ScriptedProvider(["seed", "seed"])
            agent = EvoPrompting(
                "Optimize a query.",
                lambda temperature: provider,
                evaluate,
                tune=None,
                targets=lambda parents: {"latency": 2},
                rounds=5,
                prompts_per_round=1,
                samples_per_prompt=2,
            )
            result = await agent.run(["seed"])
            self.assertEqual(result.stop_reason, "no_parents")
            self.assertEqual(len(provider.calls), 2)
            self.assertIsNone(result.best)
            self.assertIn('"latency": 2', provider.calls[0])
            self.assertIn('"latency": 3', provider.calls[0])
            parent = Individual("x", Evaluation(error=0.1, cost=4800), -480)
            self.assertEqual(paper_targets((parent,)), {"num_params": 4300, "val_accuracy": 0.918})
            agent.provider = lambda temperature: ScriptedProvider(["child"])
            agent.rounds, agent.samples_per_prompt = 1, 1
            fresh = await agent.run(["seed"])
            self.assertEqual(fresh.best.fitness, 42)
            self.assertEqual(fresh.evaluations, 2)

        root = Path(__file__).resolve().parents[1] / "evoprompting/prompts"
        for template in root.glob("*.j2"):
            self.assertFalse(
                list(Environment().parse(template.read_text()).find_all((nodes.If, nodes.CondExpr)))
            )
        with patch.object(prompts, "TEMPLATE_ROOT", root):
            asyncio.run(check())

    def test_provider_evaluator_and_tuner_errors_propagate(self):
        from evoprompting import Evaluation, EvoPrompting

        async def check():
            async def evaluate(content):
                if content == "boom":
                    raise RuntimeError("evaluation infrastructure failed")
                return Evaluation(0.1, 1)

            async def tune(*args):
                raise RuntimeError("training failed")

            for responses, tuner, message in (
                ([RuntimeError("transport failed")], None, "transport failed"),
                (["boom", "unused"], None, "evaluation infrastructure failed"),
                (["a", "b"], tune, "training failed"),
                ([asyncio.CancelledError()], None, None),
            ):
                provider = ScriptedProvider(responses)
                agent = EvoPrompting(
                    "Optimize text.",
                    lambda temperature: provider,
                    evaluate,
                    tune=tuner,
                    rounds=2,
                    prompts_per_round=1,
                    samples_per_prompt=2,
                    survivors=1,
                )
                if message:
                    with self.assertRaisesRegex(RuntimeError, message):
                        await agent.run(["seed"])
                else:
                    # ScriptedProvider does not raise BaseException instances.
                    async def cancel(content):
                        raise asyncio.CancelledError

                    agent.evaluate = cancel
                    with self.assertRaises(asyncio.CancelledError):
                        await agent.run(["seed"])

        with patch.object(
            prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[1] / "evoprompting/prompts"
        ):
            asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
