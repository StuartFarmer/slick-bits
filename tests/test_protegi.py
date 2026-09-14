"""ProTeGi's gradient, semantic expansion, sampled rejection and final beam."""

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from protegi import Evaluation, ProTeGi
from tests.providers import ScriptedProvider


class ProTeGiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "protegi/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_gradients_paraphrases_and_successive_rejects(self):
        # Source uses <START>...<END>, including the unusual END opening syntax.
        provider = ScriptedProvider(
            [
                "<START>Explain the edge case<END>",
                "<START>revised<END>",
                "best",
                "worse",
            ]
        )
        scores = {"seed": 1, "revised": 2, "best": 3, "worse": 0}
        batches = []

        async def evaluate(text, examples):
            batches.append(tuple(examples))
            return Evaluation(scores[text], ("Incorrect edge case",))

        result = await ProTeGi("Parse dates.", provider, evaluate).run(
            ["seed"],
            ["d1", "d2", "d3", "d4"],
            iterations=1,
            beam_size=1,
            minibatch_size=2,
            gradients_per_batch=1,
            steps_per_gradient=1,
            paraphrases=1,
            selection_budget=12,
            prompts_per_round=4,
        )
        self.assertEqual(result["best"], {"prompt": "best", "score": 3})
        self.assertEqual(result["history"][0]["rejected"], ["worse", "seed", "revised"])
        self.assertEqual(result["optimizer_calls"], 4)
        self.assertEqual(result["evaluations"], 11)  # gradient + 4+3+2 rejects + final
        self.assertEqual(result["example_evaluations"], 15)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[-1]), 4)
        self.assertIn("Incorrect edge case", provider.calls[0])
        self.assertIn("Explain the edge case", provider.calls[1])
        self.assertIn("Input: revised", provider.calls[2])
        self.assertIn("Input: seed", provider.calls[3])

    async def test_originals_survive_duplicates_and_no_gradient_mode(self):
        async def evaluate(text, examples):
            return Evaluation(len(text))

        agent = ProTeGi("Task", ScriptedProvider(["seed"]), evaluate)
        result = await agent.run(
            ["seed"], ["d"], iterations=1, beam_size=1, gradient_batches=0, paraphrases=1
        )
        self.assertEqual(result["history"], [{"beam": ["seed"], "rejected": []}])
        self.assertEqual(result["evaluations"], 2)

    async def test_failures_propagate_and_generated_contract_is_checked(self):
        async def evaluate(text, examples):
            return Evaluation(1, ("error",))

        agent = ProTeGi("Task", ScriptedProvider(["missing markers"]), evaluate)
        with self.assertRaisesRegex(ValueError, "response="):
            await agent.run(["seed"], ["d"], iterations=1, gradients_per_batch=1)
        self.assertEqual(agent.evaluations, 1)
        self.assertEqual(agent.optimizer_calls, 1)

        async def invalid(text, examples):
            return Evaluation(math.nan)

        with self.assertRaisesRegex(ValueError, "finite"):
            await ProTeGi("Task", ScriptedProvider([]), invalid).run(["seed"], ["d"], iterations=0)

        async def broken(text, examples):
            raise RuntimeError("evaluator failure")

        with self.assertRaisesRegex(RuntimeError, "evaluator failure"):
            await ProTeGi("Task", ScriptedProvider([]), broken).run(["seed"], ["d"])

    async def test_paper_selectors_budget_ranking_and_validation_boundary(self):
        # Negative rewards catch unsampled arms incorrectly receiving score zero.
        scores = {"bad": -4, "good": -2, "best": -1, "worst": -8, "middle": -3}
        training = [{"input": i, "target": str(i)} for i in range(20)]
        validation = [{"input": "held out", "target": "answer"}]
        for selection in ("ucb", "ucb-e", "sr", "sh", "uniform"):
            with self.subTest(selection=selection):

                async def evaluate(text, examples):
                    if examples == validation:
                        return Evaluation(10 if text == "good" else 0)
                    self.assertTrue(all(example in training for example in examples))
                    return Evaluation(scores[text])

                result = await ProTeGi("Any task", ScriptedProvider([]), evaluate).run(
                    list(scores),
                    training,
                    iterations=1,
                    beam_size=2,
                    gradient_batches=0,
                    paraphrases=0,
                    selection=selection,
                    selection_budget=101,
                    samples_per_eval=3,
                    exploration=0,
                    validation_examples=validation,
                )
                self.assertEqual(set(result["history"][0]["beam"]), {"best", "good"})
                self.assertEqual(result["best"], {"prompt": "good", "score": 10})
                measurements = result["measurements"]
                used = sum(m["size"] for m in measurements if m["phase"] == "selection")
                self.assertGreater(used, 0)
                self.assertLessEqual(used, 101)
                self.assertEqual(
                    result["example_evaluations"], sum(m["size"] for m in measurements)
                )
                self.assertEqual([m["size"] for m in measurements if m["phase"] == "final"], [1, 1])

    async def test_ucb_weights_observations_by_actual_batch_size(self):
        calls = {"first": 0, "second": 0}

        async def evaluate(text, examples):
            calls[text] += 1
            score = {"first": (0, 1, 0, 0), "second": (0, 0.6, 0)}[text][calls[text] - 1]
            return Evaluation(score)

        result = await ProTeGi("Task", ScriptedProvider([]), evaluate).run(
            ["first", "second"],
            ["a", "b", "c"],
            iterations=1,
            beam_size=1,
            gradient_batches=0,
            paraphrases=0,
            selection="ucb",
            selection_budget=7,
            samples_per_eval=3,
            exploration=0,
        )
        selected = [m for m in result["measurements"] if m["phase"] == "selection"]
        self.assertEqual(
            [(m["prompt"], m["size"]) for m in selected],
            [("first", 3), ("second", 3), ("first", 1)],
        )
        # first: (3 * 1 + 1 * 0) / 4 = .75 beats .6; a mean of batch means loses.
        self.assertEqual(result["history"][0]["beam"], ["first"])

    async def test_insufficient_selection_budget_never_uses_empty_batches(self):
        for selection in ("ucb", "ucb-e", "sr", "sh", "uniform"):
            with self.subTest(selection=selection):

                async def evaluate(text, examples):
                    self.assertTrue(examples)
                    return Evaluation(0)

                with self.assertRaisesRegex(RuntimeError, "budget"):
                    await ProTeGi("Task", ScriptedProvider([]), evaluate).run(
                        ["a", "b", "c"],
                        ["d"],
                        iterations=1,
                        beam_size=1,
                        gradient_batches=0,
                        paraphrases=0,
                        selection=selection,
                        selection_budget=2,
                    )

    async def test_small_datasets_odd_beams_ties_and_budget_exhaustion(self):
        async def evaluate(text, examples):
            self.assertEqual(examples, ["only example"])
            return Evaluation(-1)

        for selection in ("ucb", "ucb-e", "sr", "sh", "uniform"):
            for width in (1, 2, 3):
                for budget in (5, 11, 50):
                    with self.subTest(selection=selection, width=width, budget=budget):
                        result = await ProTeGi("Task", ScriptedProvider([]), evaluate).run(
                            ["a", "b", "c", "d", "e"],
                            ["only example"],
                            iterations=1,
                            beam_size=width,
                            gradient_batches=0,
                            paraphrases=0,
                            selection=selection,
                            selection_budget=budget,
                        )
                        self.assertEqual(result["history"][0]["beam"], list("abc")[:width])
                        used = sum(
                            m["size"] for m in result["measurements"] if m["phase"] == "selection"
                        )
                        self.assertLessEqual(used, budget)

    async def test_ucb_exploration_revisits_other_arms(self):
        async def evaluate(text, examples):
            return Evaluation(float(text == "best"))

        for selection in ("ucb", "ucb-e"):
            for exploration in (0, 2):
                result = await ProTeGi("Task", ScriptedProvider([]), evaluate).run(
                    ["best", "worse"],
                    ["a", "b"],
                    iterations=1,
                    beam_size=1,
                    gradient_batches=0,
                    paraphrases=0,
                    selection=selection,
                    selection_budget=30,
                    samples_per_eval=1,
                    exploration=exploration,
                )
                other_pulls = sum(
                    m["phase"] == "selection" and m["prompt"] == "worse"
                    for m in result["measurements"]
                )
                if exploration:
                    self.assertGreater(other_pulls, 1)
                else:
                    self.assertEqual(other_pulls, 1)

    async def test_large_finite_rewards_do_not_overflow_running_means(self):
        async def evaluate(text, examples):
            return Evaluation(1.5e308 if text == "best" else 1e308)

        for selection in ("ucb", "ucb-e", "sr", "sh", "uniform"):
            with self.subTest(selection=selection):
                result = await ProTeGi("Task", ScriptedProvider([]), evaluate).run(
                    ["bad", "best"],
                    list("abcde"),
                    iterations=1,
                    beam_size=1,
                    gradient_batches=0,
                    paraphrases=0,
                    selection=selection,
                    selection_budget=20,
                )
                self.assertEqual(result["best"]["prompt"], "best")

    async def test_templates_and_rejected_raw_outputs_from_another_directory(self):
        provider = ScriptedProvider(
            [
                "<START>reason<END>",
                "<START>new prompt<END>",
                "variation",
                "<START> <END>",
                "<START>one<END><START>two<END>",
                "   ",
                RuntimeError("transport failed"),
            ]
        )

        async def evaluate(text, examples):
            return Evaluation(0)

        agent = ProTeGi("Summarize records", provider, evaluate)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                rendered = await ProTeGi.gradients.render(agent, "seed", ["error"], 1)
                self.assertIn("Summarize records", rendered)
                self.assertEqual(
                    await agent.gradients("seed", ["error"], 1, provider=provider), ["reason"]
                )
                self.assertEqual(
                    await agent.revise("seed", ["error"], "reason", 1, provider=provider),
                    ["new prompt"],
                )
                self.assertEqual(
                    await agent.paraphrase("new prompt", provider=provider), "variation"
                )
                with self.assertRaisesRegex(ValueError, "response="):
                    await agent.gradients("seed", ["error"], 1, provider=provider)
                with self.assertRaisesRegex(ValueError, "response="):
                    await agent.revise("seed", ["error"], "reason", 1, provider=provider)
                with self.assertRaisesRegex(ValueError, "response="):
                    await agent.paraphrase("seed", provider=provider)
                with self.assertRaisesRegex(RuntimeError, "transport failed"):
                    await agent.paraphrase("seed", provider=provider)
            finally:
                os.chdir(previous)
