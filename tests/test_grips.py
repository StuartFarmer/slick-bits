"""Check GrIPS's edit provenance, acceptance, and evaluation budget."""

import math
import random
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from grips import GrIPS
from grips.agent import edit_text
from tests.providers import ScriptedProvider


def spans(text):
    return [match.span() for match in re.finditer(r"\S+", text)]


class GrIPSTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "grips/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_neighborhood_snapshot_best_selection_and_counts(self):
        provider = ScriptedProvider(["worse", "winner", "tie", "lower"])
        values = {"seed": 1, "worse": 0, "winner": 3, "tie": 3, "lower": 2}

        async def evaluate(text):
            return values[text]

        result = await GrIPS("Teach botany", provider, evaluate, spans).run(
            "seed", iterations=2, candidates=2, operations=("substitute",)
        )
        self.assertEqual(result["best"], {"prompt": "winner", "score": 3})
        self.assertEqual(result["evaluations"], 5)
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["optimizer_calls"], 4)
        self.assertEqual([h["accepted"] for h in result["history"]], [True, False])
        self.assertIn("Phrase: seed", provider.calls[1])
        self.assertIn("Phrase: winner", provider.calls[2])
        self.assertIn("Teach botany", provider.calls[0])

    async def test_native_edits_preserve_offsets_and_restore_deleted_phrases(self):
        async def paraphrase(phrase):
            return "rewritten"

        deleted = await edit_text("a b", [], ("delete",), 1, spans, paraphrase, random.Random(1))
        self.assertEqual(deleted.text, "b")
        self.assertEqual(deleted.deleted, ["a"])
        restored = await edit_text(
            deleted.text, deleted.deleted, ("add",), 1, spans, paraphrase, random.Random(1)
        )
        self.assertEqual(restored.text, "b a")
        self.assertEqual(restored.added, ["a"])
        swapped = await edit_text("a b", [], ("swap",), 1, spans, paraphrase, random.Random(1))
        self.assertEqual(swapped.text, "b a")

    async def test_empty_edit_rejection_patience_and_annealed_global_best(self):
        async def score(text):
            return 1 if text == "seed" else 0

        result = await GrIPS("Task", ScriptedProvider([]), score, spans).run(
            "seed", operations=("delete",), candidates=2, iterations=9, patience=2
        )
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(len(result["rejections"]), 4)
        self.assertEqual(result["evaluations"], 1)
        annealed = await GrIPS("Task", ScriptedProvider(["worse"]), score, spans).run(
            "seed",
            operations=("substitute",),
            candidates=1,
            iterations=1,
            anneal=True,
            temperature=1e9,
        )
        self.assertEqual(annealed["current"]["prompt"], "worse")
        self.assertEqual(annealed["best"]["prompt"], "seed")

    async def test_blank_paraphrase_and_bad_measurement_propagate(self):
        async def score(text):
            return 1

        provider = ScriptedProvider([" "])
        agent = GrIPS("Task", provider, score, spans)
        with self.assertRaisesRegex(ValueError, "empty paraphrase"):
            await agent.run("seed", operations=("substitute",), candidates=1)
        self.assertEqual(agent.evaluations, 1)
        for value in (math.nan, math.inf):

            async def bad(text):
                return value

            with self.assertRaisesRegex(ValueError, "finite"):
                await GrIPS("Task", ScriptedProvider([]), bad, spans).run("seed")


if __name__ == "__main__":
    unittest.main()
