"""Check GPS generation replacement and final archive selection."""

import random
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from gps import GPS
from gps.agent import fill_prompt, mask_prompt, preserves_placeholders
from tests.providers import ScriptedProvider


class GPSTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "gps/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_children_replace_generation_while_archive_retains_best(self):
        scores = {"seed": 10, "other": 3, "child": 2, "grandchild": 1}

        async def evaluate(text):
            return scores[text]

        provider = ScriptedProvider(["child", "grandchild"])
        result = await GPS("Write recipes", provider, evaluate).run(
            ["seed", "other"], generations=3, top_k=1, offspring_per_parent=1
        )
        self.assertEqual(result["best"]["prompt"], "seed")
        self.assertEqual(result["population"][0]["prompt"], "grandchild")
        self.assertIn("Sentence 1: child", provider.calls[1])
        self.assertIn("Write recipes", provider.calls[0])
        self.assertEqual(result["evaluations"], 4)

    async def test_constraints_duplicates_empty_fallback_and_counts(self):
        async def evaluate(text):
            return len(text)

        provider = ScriptedProvider(["", "lost slot", "seed {{input}}", "new {{input}}"])
        result = await GPS("Task", provider, evaluate).run(
            ["seed {{input}}"], generations=2, top_k=1, offspring_per_parent=4
        )
        self.assertEqual(
            [item["rejection"] for item in result["history"]],
            ["empty", "template constraint", "duplicate", ""],
        )
        self.assertEqual(result["evaluations"], 2)
        fallback = await GPS(
            "Task", ScriptedProvider(["bad"]), evaluate, accept=lambda parent, child: False
        ).run(["seed"], generations=2, top_k=1, offspring_per_parent=1)
        self.assertEqual(fallback["population"][0]["prompt"], "seed")
        self.assertEqual(fallback["evaluations"], 1)

    async def test_evaluation_failure_propagates(self):
        async def fail(text):
            return float("nan")

        with self.assertRaisesRegex(ValueError, "finite"):
            await GPS("Task", ScriptedProvider([]), fail).run(["seed"])

    async def test_global_duplicates_are_not_evaluated_again(self):
        scores = {"seed": 5, "other": 4, "child": 1}

        async def evaluate(text):
            return scores[text]

        result = await GPS("Task", ScriptedProvider(["child", "other"]), evaluate).run(
            ["seed", "other"], generations=3, top_k=1, offspring_per_parent=1
        )
        self.assertEqual(result["history"][-1]["rejection"], "duplicate")
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(result["population"], [{"prompt": "child", "score": 1}])
        self.assertEqual([p["prompt"] for p in result["archive"]], ["seed", "child"])

    async def test_paper_defaults_six_reproductions_and_seed_count_top_k(self):
        async def evaluate(text):
            return 1

        result = await GPS("Task", ScriptedProvider([""] * 180), evaluate).run(["a", "b"])
        self.assertEqual(result["optimizer_calls"], 180)
        self.assertEqual(result["attempts"], 180)
        self.assertEqual(len(result["generations"]), 7)
        self.assertEqual([p["prompt"] for p in result["finalists"]], ["a", "b"])

    async def test_final_rescoring_only_uses_generation_winners(self):
        seen = []
        measurements = iter([10, 9, 8, 0, 20])

        async def evaluate(text):
            seen.append(text)
            return next(measurements)

        result = await GPS("Task", ScriptedProvider(["child"]), evaluate).run(
            ["seed", "other"],
            generations=2,
            top_k=1,
            offspring_per_parent=1,
            rescore_final=True,
        )
        self.assertEqual(seen, ["seed", "other", "child", "seed", "child"])
        self.assertEqual(result["best"], {"prompt": "child", "score": 20})
        self.assertEqual(result["evaluations"], 5)

    async def test_back_translation_has_two_calls_and_rejects_bad_intermediate(self):
        async def evaluate(text):
            return len(text)

        provider = ScriptedProvider(["lire {{x}}", "read {{x}}", "lost input"])
        result = await GPS("Task", provider, evaluate).run(
            ["seed {{x}}"],
            generations=2,
            strategy="back_translation",
            languages=("French", "German"),
            offspring_per_parent=2,
        )
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["optimizer_calls"], 3)
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(result["history"][0]["translation"], "lire {{x}}")
        self.assertEqual(result["history"][1]["rejection"], "translation constraint")
        self.assertIn("French", provider.calls[0])
        self.assertIn("lire {{x}}", provider.calls[1])
        self.assertIn("German", provider.calls[2])

    async def test_cloze_reconstructs_only_masked_words_and_records_malformed_output(self):
        async def evaluate(text):
            return len(text)

        provider = ScriptedProvider(
            [
                "<extra_id_0> inspect <extra_id_1>",
                "unstructured text",
                "<extra_id_0> <extra_id_1>",
            ]
        )
        result = await GPS("Task", provider, evaluate).run(
            ["read {{\ninput\n}}"],
            generations=2,
            strategy="cloze",
            offspring_per_parent=3,
        )
        self.assertEqual(result["population"][0]["prompt"], "inspect {{\ninput\n}}")
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(
            [h["rejection"] for h in result["history"]], ["", "invalid cloze", "invalid cloze"]
        )
        self.assertEqual(result["history"][1]["raw"], "unstructured text")
        self.assertIn("<extra_id_0> {{\ninput\n}}", provider.calls[0])

    def test_multiline_slots_and_control_tags_are_preserved(self):
        parent = "{% if x %}read {{\nx\n}}{% endif %}"
        self.assertFalse(preserves_placeholders(parent, "read"))
        self.assertFalse(preserves_placeholders(parent, "read {{\nx\n}}"))
        self.assertTrue(preserves_placeholders(parent, parent.replace("read", "check")))

    async def test_provider_failure_is_not_retried_and_keeps_attempt_record(self):
        async def evaluate(text):
            return 1

        agent = GPS("Task", ScriptedProvider([RuntimeError("offline")]), evaluate)
        with self.assertRaisesRegex(RuntimeError, "offline"):
            await agent.run(["seed"], generations=2, offspring_per_parent=1)
        self.assertEqual(agent.optimizer_calls, 1)
        self.assertEqual(agent.attempts, 1)
        self.assertEqual(agent.history[0]["rejection"], "pending")

    def test_cloze_randomness_protected_text_and_ordered_fill_contract(self):
        parent = "Choose  the\nanswer {% if x %}{{ input }}{% endif %}."
        masked, pieces = mask_prompt(parent, random.Random(4), 1.0)
        self.assertEqual(
            masked, "<extra_id_0>  <extra_id_1>\n<extra_id_2> {% if x %}{{ input }}{% endif %}."
        )
        restored = fill_prompt(
            "<extra_id_0> Select <extra_id_1> a <extra_id_2> response <extra_id_3>", pieces
        )
        self.assertEqual(restored, "Select  a\nresponse {% if x %}{{ input }}{% endif %}.")
        self.assertEqual(
            mask_prompt(parent, random.Random(4), 0.5), mask_prompt(parent, random.Random(4), 0.5)
        )
        for raw in ("<extra_id_1> x <extra_id_0>", "<extra_id_0> x", ""):
            with self.assertRaisesRegex(ValueError, "invalid cloze"):
                fill_prompt(raw, pieces)

    async def test_uneditable_cloze_uses_no_model_calls(self):
        async def evaluate(text):
            return 1

        result = await GPS("Task", ScriptedProvider([]), evaluate).run(
            ["{{ input }}"], generations=2, strategy="cloze", offspring_per_parent=1
        )
        self.assertEqual(result["optimizer_calls"], 0)
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["history"][0]["rejection"], "no editable tokens")

    async def test_all_templates_render_with_explicit_owner_and_no_branches(self):
        agent = GPS("Unrelated task", ScriptedProvider([]), None)
        operations = (
            (GPS.mutate, ("seed {{x}}",)),
            (GPS.translate, ("seed {{x}}", "French")),
            (GPS.back_translate, ("lire {{x}}", "French")),
            (GPS.cloze, ("<extra_id_0> {{x}}", 1)),
        )
        for operation, args in operations:
            rendered = await operation.render(agent, *args)
            self.assertIn("Unrelated task", rendered)
            self.assertIn("{{x}}", rendered)
        for path in (Path(__file__).resolve().parents[1] / "gps/prompts").glob("*.j2"):
            parsed = Environment().parse(path.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
