"""Offline checks for process supervision, selection, and official data adapters."""

import math
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from lets_verify_step_by_step import (
    StepProbabilities,
    VerifyStepByStep,
    process_examples,
    synthetic_labels,
    synthetic_outcome,
)
from lets_verify_step_by_step.data import phase2_examples, scored_sample_trial
from tests.providers import ScriptedProvider


def probabilities(correct):
    return StepProbabilities(positive=correct, neutral=0, negative=1 - correct)


class VerificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "lets_verify_step_by_step/prompts"
        patcher = patch("slick.prompts.TEMPLATE_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_product_selection_includes_final_step_and_hides_answers(self):
        provider = ScriptedProvider(["Inspect\nReject", "Check\nAccept"])

        async def reward(problem, steps):
            self.assertEqual(problem, "Classify a document")
            return [
                probabilities(p) for p in ([0.99, 0.1] if steps[0] == "Inspect" else [0.8, 0.8])
            ]

        async def evaluate(problem, solution):
            self.assertEqual(len(provider.calls), 2)
            return solution.steps[-1] == "Accept"

        agent = VerifyStepByStep("Apply the supplied rubric", provider, reward, evaluate)
        result = await agent.run("Classify a document", n=2)
        self.assertEqual(result.best.solution.text, "Check\nAccept")
        self.assertAlmostEqual(result.best.score, 0.64)
        self.assertTrue(result.correct)
        self.assertEqual(result.calls, 2)
        self.assertEqual(provider.calls[0], provider.calls[1])
        self.assertNotIn("Reject", provider.calls[1])

    async def test_neutral_policy_and_underflow_do_not_change_ranking(self):
        async def reward(problem, steps):
            p = 0.01 if steps[0] == "worse" else 0.02
            return [probabilities(p)] * len(steps)

        provider = ScriptedProvider(["\n".join([word] * 500) for word in ("worse", "better")])
        result = await VerifyStepByStep("Task", provider, reward).run("Input", n=2)
        self.assertEqual(result.best.solution.steps[0], "better")
        self.assertEqual([sample.score for sample in result.samples], [0.0, 0.0])

        async def neutral_reward(problem, steps):
            return [StepProbabilities(0.1, 0.8, 0.1)]

        for neutral, expected in ((True, 0.9), (False, 0.1)):
            agent = VerifyStepByStep(
                "Task", ScriptedProvider(["done"]), neutral_reward, neutral_is_correct=neutral
            )
            self.assertAlmostEqual((await agent.run("Input", n=1)).best.score, expected)

    async def test_active_selection_trains_only_selected_prefixes(self):
        provider = ScriptedProvider([f"candidate {i}\nresult {i}" for i in range(6)])

        async def reward(problem, steps):
            return [probabilities(1 - int(steps[0][-1]) / 10)] * 2

        async def evaluate(problem, solution):
            return solution.steps[0] == "candidate 0"

        labeled = []

        async def label(problem, steps):
            labeled.append(steps[0])
            return (1, -1)

        async def fit(examples, epochs):
            self.assertEqual(epochs, 2)
            self.assertEqual(len(examples), 10)
            self.assertEqual([x.label for x in examples], [1, -1] * 5)
            return reward

        agent = VerifyStepByStep("Task", provider, reward, evaluate)
        training = await agent.learn(["Input"], label, fit, pool_size=6, k=5)
        self.assertEqual(
            labeled, ["candidate 1", "candidate 2", "candidate 3", "candidate 4", "candidate 0"]
        )
        self.assertEqual(len(training), 10)
        self.assertEqual(len(provider.calls), 6)
        self.assertEqual(tuple(agent.training), training)

    async def test_minimum_zero_ties_and_active_quota_shortage(self):
        async def reward(problem, steps):
            return (
                [probabilities(0.8), probabilities(0.8)]
                if steps[0] == "a"
                else [probabilities(0.7), probabilities(1)]
            )

        for reduction, expected in (("product", "b"), ("minimum", "a")):
            agent = VerifyStepByStep(
                "Task", ScriptedProvider(["a\nanswer", "b\nanswer"]), reward, reduction=reduction
            )
            self.assertEqual((await agent.run("Input", n=2)).best.solution.steps[0], expected)

        async def zero_reward(problem, steps):
            return [probabilities(0)]

        async def all_correct(problem, solution):
            return True

        agent = VerifyStepByStep(
            "Task", ScriptedProvider(["first", "second"]), zero_reward, all_correct
        )
        result = await agent.run("Input", n=2)
        self.assertEqual(result.best.solution.text, "first")
        self.assertEqual(result.best.score, 0)
        selected = await agent.select_for_labeling("Input", result.samples, 2)
        self.assertEqual([x.solution.text for x in selected], ["first", "second"])

    async def test_training_rounds_replace_reward_and_keep_generator_fixed(self):
        provider = ScriptedProvider(["a", "b", "a", "b"])

        async def original(problem, steps):
            return [probabilities(0.9 if steps[0] == "a" else 0.1)]

        async def trained(problem, steps):
            return [probabilities(0.1 if steps[0] == "a" else 0.9)]

        async def evaluate(problem, solution):
            return False

        async def label(problem, steps):
            return [-1]

        batches = []

        async def fit(examples, epochs):
            batches.append([x.steps[0] for x in examples])
            return trained

        agent = VerifyStepByStep("Task", provider, original, evaluate)
        await agent.learn(["Input"], label, fit, pool_size=2, k=1, rounds=2)
        self.assertEqual(batches, [["a"], ["a", "b"]])
        self.assertIs(agent.provider, provider)
        self.assertIs(agent.reward, trained)

    async def test_blank_output_and_reward_failure_are_recorded_without_retry(self):
        async def reward(problem, steps):
            return [probabilities(1)] * len(steps)

        for response in (" ", ("raw", ["tool"]), TimeoutError("offline")):
            agent = VerifyStepByStep("Task", ScriptedProvider([response]), reward)
            with self.assertRaises((ValueError, TimeoutError)):
                await agent.run("Input", n=2)
            self.assertEqual(len(agent.calls), 1)
            self.assertIn("error", agent.calls[0])

        for scores in ([], [StepProbabilities(1, 1, 1)], [probabilities(float("nan"))]):

            async def invalid_reward(problem, steps):
                return scores

            agent = VerifyStepByStep("Task", ScriptedProvider(["answer"]), invalid_reward)
            with self.assertRaises(ValueError):
                await agent.run("Input", n=1)
            self.assertEqual(agent.calls[0]["response"], "answer")
            self.assertIn("error", agent.calls[0])

    async def test_templates_bind_from_either_launch_directory(self):
        async def reward(problem, steps):
            return [probabilities(1)] * len(steps)

        agent = VerifyStepByStep("Task {{ literal }}", ScriptedProvider([]), reward)
        previous = Path.cwd()
        try:
            for directory in (self.root.parent, self.root.parent.parent):
                os.chdir(directory)
                rendered = await VerifyStepByStep.generate.render(agent, "source text")
                self.assertIn("Task {{ literal }}", rendered)
                self.assertIn("source text", rendered)
        finally:
            os.chdir(previous)
        parsed = Environment().parse((self.root / "generate.j2").read_text())
        self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])

    def test_first_error_and_strict_synthetic_threshold(self):
        scores = [
            probabilities(0.9),
            StepProbabilities(0.8, 0, 0.2),
            probabilities(0.79),
            probabilities(1),
        ]
        labels = synthetic_labels(scores)
        self.assertEqual(labels, (1, 1, -1))
        self.assertFalse(synthetic_outcome(scores))
        examples = process_examples("problem", ("a", "b", "c", "d"), (1, 0, -1, 1))
        self.assertEqual([x.label for x in examples], [1, 0, -1])
        self.assertEqual(examples[-1].steps, ("a", "b", "c"))
        with self.assertRaises(ValueError):
            process_examples("problem", ("a", "b"), (1,))

    def test_official_phase2_mapping_ignores_alternatives_and_qc(self):
        row = {
            "generation": 1,
            "is_quality_control_question": False,
            "is_initial_screening_question": False,
            "question": {"problem": "arbitrary input"},
            "label": {
                "finish_reason": "found_error",
                "steps": [
                    {
                        "completions": [{"text": "start", "rating": 0, "flagged": None}],
                        "chosen_completion": 0,
                        "human_completion": None,
                    },
                    {
                        "completions": [
                            {"text": "error", "rating": -1, "flagged": False},
                            {"text": "repair", "rating": 1, "flagged": False},
                        ],
                        "chosen_completion": None,
                        "human_completion": None,
                    },
                ],
            },
        }
        self.assertEqual(
            [(x.steps, x.label) for x in phase2_examples(row)],
            [(("start",), 0), (("start", "error"), -1)],
        )
        row["is_quality_control_question"] = True
        self.assertEqual(phase2_examples(row), ())

    def test_official_evaluation_counts_missing_samples_as_failures(self):
        rows = [
            {
                "problem": "p",
                "answer": "yes",
                "prm_score": 0.9,
                "orm_score": 0.1,
                "is_correct": True,
            },
            {
                "problem": "p",
                "given_answer": "no",
                "prm_score": 0.1,
                "orm_score": 0.9,
                "is_correct": False,
            },
        ]
        self.assertEqual(
            scored_sample_trial(rows, n=3, samples_per_problem=3, rng=random.Random(0)), 1
        )
        self.assertEqual(
            scored_sample_trial(
                rows, n=3, samples_per_problem=3, method="orm", rng=random.Random(0)
            ),
            0,
        )
        mean = (
            sum(
                scored_sample_trial(rows[:1], n=1, samples_per_problem=2, rng=random.Random(seed))
                for seed in range(300)
            )
            / 300
        )
        self.assertTrue(0.4 < mean < 0.6, mean)
        self.assertTrue(math.isfinite(mean))


if __name__ == "__main__":
    unittest.main()
