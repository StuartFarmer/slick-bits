"""Check PROMST categorized feedback, ancestry, heuristic admission, and budgets."""

import importlib.util
import inspect
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from promst import PROMST, Evaluation, Feedback, Heuristic
from tests.providers import ScriptedProvider


class PROMSTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "promst/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_paper_defaults_and_stagnation_stop(self):
        defaults = inspect.signature(PROMST.run).parameters
        self.assertEqual(defaults["beam_size"].default, 5)
        self.assertEqual(defaults["children"].default, 8)
        self.assertEqual(defaults["threshold_factor"].default, 0.8)
        provider = ScriptedProvider(
            [value for i in range(100) for value in ("advice", f"candidate {i}")]
        )

        async def evaluate(text):
            return Evaluation(1, [Feedback("error", "repair")])

        result = await PROMST("Task", provider, evaluate).run("seed", depth=10)
        self.assertEqual(result["evaluations"], 101)  # 1 + 20 + 5*8 + 5*8
        self.assertEqual(result["stop_reason"], "stagnation")
        self.assertEqual(result["generation_best"], [1, 1, 1, 1])
        self.assertEqual(result["best"].prompt, "seed")  # Stable ties retain incumbents.

    async def test_fits_once_per_generation_on_previous_measured_archive(self):
        provider = ScriptedProvider([v for p in ("a", "b", "c", "d") for v in ("advice", p)])
        fitted = []

        async def evaluate(text):
            return Evaluation(0 if text == "seed" else 1, [Feedback("error", "repair")])

        async def predict(text):
            return [1] * 5

        async def fit(history):
            fitted.append([item.prompt for item in history])
            return Heuristic(predict, [0] * 5)

        result = await PROMST("Task", provider, evaluate, fit).run(
            "seed", depth=3, first_children=2, children=1, beam_size=2, score_start=2
        )
        self.assertEqual(fitted, [["seed", "a", "b"]])
        self.assertEqual(result["fit_calls"], 1)
        self.assertEqual(result["evaluations"], 5)

    async def test_categorized_feedback_and_global_beam_ancestry(self):
        provider = ScriptedProvider(
            ["format advice", "loop advice", "better", "format advice", "loop advice", "best"]
        )

        async def evaluate(text):
            return Evaluation(
                {"seed": 0, "better": 1, "best": 2}[text],
                [Feedback("format", "missing field"), Feedback("loop", "repeats")],
            )

        result = await PROMST("Plan routes", provider, evaluate).run(
            "seed", depth=3, children=1, first_children=1, beam_size=1
        )
        self.assertEqual(result["best"].prompt, "best")
        self.assertEqual(result["best"].ancestors, ["seed", "better"])
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(result["optimizer_calls"], 6)
        self.assertIn('"seed", "better"', provider.calls[-1])
        for context in provider.calls:
            self.assertIn("Plan routes", context)

    async def test_screening_uses_variance_and_error_not_only_predicted_mean(self):
        provider = ScriptedProvider(["advice", "reject", "advice", "admit"])
        measured, fitted = [], []

        async def evaluate(text):
            measured.append(text)
            return Evaluation(1 if text == "seed" else 2, [Feedback("format", "repair")])

        async def predict(text):
            return [0, 0] if text == "reject" else [0, 1]

        async def fit(history):
            fitted.append([item.prompt for item in history])
            return Heuristic(predict, [0.25, 0.25])

        result = await PROMST("Task", provider, evaluate, fit).run(
            "seed", depth=2, children=1, first_children=1, score_start=1
        )
        self.assertEqual(measured, ["seed", "admit"])
        self.assertEqual(fitted, [["seed"]])
        self.assertEqual(result["history"][0]["rejection"], "heuristic")
        self.assertEqual(result["history"][1]["bound"], 1)
        self.assertEqual(result["prediction_calls"], 2)
        self.assertEqual(result["attempts"], 2)

    async def test_three_n_screening_budget_and_empty_feedback_stop(self):
        async def evaluate(text):
            return Evaluation(1, [Feedback("format", "repair")])

        async def predict(text):
            return [0, 0]

        async def fit(history):
            return Heuristic(predict, [0])

        provider = ScriptedProvider(["advice", "a", "advice", "b", "advice", "c"])
        result = await PROMST("Task", provider, evaluate, fit).run(
            "seed", depth=2, children=1, first_children=1, score_start=1
        )
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["evaluations"], 1)

        async def solved(text):
            return Evaluation(1)

        result = await PROMST("Task", ScriptedProvider([]), solved).run("seed")
        self.assertEqual(result["optimizer_calls"], 0)

    async def test_nonfinite_heuristic_and_blank_revision_propagate(self):
        async def evaluate(text):
            return Evaluation(1, [Feedback("error", "repair")])

        async def predict(text):
            return [float("nan")]

        async def fit(history):
            return Heuristic(predict, [0])

        with self.assertRaisesRegex(ValueError, "finite"):
            await PROMST("Task", ScriptedProvider(["advice", "child"]), evaluate, fit).run(
                "seed", depth=2, children=1, first_children=1, score_start=1
            )
        agent = PROMST("Task", ScriptedProvider(["advice", " "]), evaluate)
        with self.assertRaisesRegex(ValueError, "empty generated"):
            await agent.run("seed", depth=2, children=1, first_children=1)
        self.assertEqual(agent.evaluations, 1)
        self.assertEqual(agent.attempts, 1)
        self.assertEqual(agent.responses[-1], {"operation": "revise", "response": " "})
        self.assertEqual(agent.history[-1]["rejection"], "generation_pending")

    async def test_sampling_duplicates_and_rendering_from_another_directory(self):
        provider = ScriptedProvider(["advice", "seed", "advice", "child"])

        async def evaluate(text):
            return Evaluation(1, [Feedback("error", f"instance-{i:02d}") for i in range(30)])

        agent = PROMST("Any task", provider, evaluate)
        result = await agent.run("seed", depth=2, first_children=2)
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["history"][0]["rejection"], "duplicate")
        self.assertEqual(provider.calls[0].count("instance-"), 10)
        self.assertEqual(provider.calls[2].count("instance-"), 10)
        self.assertNotEqual(provider.calls[0], provider.calls[2])
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                summary = await PROMST.summarize_feedback.render(agent, "seed", "error", ["x"])
                revision = await PROMST.revise.render(agent, "seed", [], ["seed"])
            finally:
                os.chdir(previous)
        self.assertIn("Any task", summary)
        self.assertIn("Any task", revision)
        for path in (Path(__file__).resolve().parents[1] / "promst/prompts").glob("*.j2"):
            tree = Environment().parse(path.read_text())
            self.assertEqual(list(tree.find_all((nodes.If, nodes.CondExpr))), [])

    async def test_improvement_resets_patience_and_evaluation_errors_propagate(self):
        provider = ScriptedProvider([v for p in ("a", "b", "c", "d") for v in ("advice", p)])

        async def evaluate(text):
            return Evaluation(
                {"seed": 1, "a": 1, "b": 2, "c": 2, "d": 2}[text], [Feedback("error", "repair")]
            )

        result = await PROMST("Task", provider, evaluate).run(
            "seed", first_children=1, children=1, beam_size=1, patience=2
        )
        self.assertEqual(result["generation_best"], [1, 1, 2, 2, 2])
        self.assertEqual(result["stop_reason"], "stagnation")

        async def invalid(text):
            return Evaluation(float("nan"))

        with self.assertRaisesRegex(ValueError, "fitness must be finite"):
            await PROMST("Task", ScriptedProvider([]), invalid).run("seed")

        async def failed(text):
            raise RuntimeError("evaluation failed")

        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            await PROMST("Task", ScriptedProvider([]), failed).run("seed")

    async def test_duplicate_generation_does_not_prevent_later_improvement(self):
        provider = ScriptedProvider(["advice", "seed", "advice", "better"])

        async def evaluate(text):
            return Evaluation(1 if text == "seed" else 2, [Feedback("error", "repair")])

        result = await PROMST("Task", provider, evaluate).run(
            "seed", depth=3, first_children=1, children=1, beam_size=1
        )
        self.assertEqual(result["best"].prompt, "better")
        self.assertEqual(result["generation_best"], [1, 1, 2])


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
    "Install promst/requirements-score-model.txt to check real Longformer training",
)
class LongformerTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_five_model_training_and_heldout_errors(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            LongformerConfig,
            LongformerForSequenceClassification,
            PreTrainedTokenizerFast,
        )

        from promst import LongformerFit
        from promst.agent import Candidate

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "goal": 2}, "[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
            )
            tokenizer.save_pretrained(directory)
            base = LongformerForSequenceClassification(
                LongformerConfig(
                    vocab_size=3,
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                    attention_window=4,
                    max_position_embeddings=32,
                    num_labels=1,
                    problem_type="regression",
                    pad_token_id=1,
                    hidden_dropout_prob=0,
                    attention_probs_dropout_prob=0,
                )
            )
            base.save_pretrained(directory)
            history = [Candidate(f"goal {i}", i / 10 - 0.5, []) for i in range(10)]
            trainer = LongformerFit(
                checkpoint=directory, epochs=1, batch_size=2, max_length=16, learning_rate=0.01
            )
            self.assertIsNone(await trainer(history[:4]))
            heuristic = await trainer(history)
            self.assertEqual(len(heuristic.errors), 5)
            self.assertEqual(len(trainer.models), 5)
            self.assertEqual(len(set(tuple(test) for train, test in trainer.splits)), 5)
            for model, (train, test), error in zip(
                trainer.models, trainer.splits, heuristic.errors
            ):
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                self.assertEqual((len(train), len(test)), (8, 2))
                self.assertFalse(set(train) & set(test))
                self.assertEqual(set(train) | set(test), set(range(10)))
                self.assertFalse(
                    torch.equal(model.classifier.out_proj.weight, base.classifier.out_proj.weight)
                )
                batch = tokenizer(
                    [history[i].prompt for i in test],
                    padding=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
                with torch.no_grad():
                    predicted = model(**batch).logits.flatten().tolist()
                expected = sum(abs(p - history[i].score) for p, i in zip(predicted, test)) / 2
                self.assertAlmostEqual(error, expected, places=6)
            scores = await heuristic.predict("goal new")
            self.assertEqual(len(scores), 5)
            self.assertTrue(all(math.isfinite(x) for x in scores))


if __name__ == "__main__":
    unittest.main()
