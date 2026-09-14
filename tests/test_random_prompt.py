import unittest
from pathlib import Path
from unittest.mock import patch

from random_prompt import RandomPrompt
from tests.providers import ScriptedProvider


class RandomPromptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "random_prompt/prompts"
        )
        root.start()
        self.addCleanup(root.stop)

    async def test_vocabulary_draws_preserve_duplicates_and_replace_all_slots(self):
        tokens, contexts = [], []

        def decode(ids):
            tokens.append(ids)
            return "|"

        async def evaluate(context):
            contexts.append(context)
            return -len(contexts)

        result = await RandomPrompt("task", ScriptedProvider([]), evaluate, decode=decode).run(
            "x{separator}y{separator}", vocabulary_size=4, draws=3, min_length=4, max_length=4
        )
        self.assertTrue(all(len(set(row)) == 4 for row in tokens))
        self.assertEqual(contexts, ["x|y|"] * 3)
        self.assertEqual(result["best"].score, -1)
        self.assertEqual(result["evaluations"], 3)

    async def test_unconditional_is_empty_context_and_preserves_whitespace(self):
        provider = ScriptedProvider([" \n", "!"])
        lengths = []

        def configure(model, length):
            lengths.append(length)
            return model

        async def evaluate(context):
            return float("!" in context)

        result = await RandomPrompt("task", provider, evaluate, configure_sampling=configure).run(
            "A{separator}B", mode="unconditional", draws=2, min_length=2, max_length=2
        )
        self.assertEqual(provider.calls, ["", ""])
        self.assertEqual(lengths, [2, 2])
        self.assertEqual(result["candidates"][0].separator, " \n")
        self.assertEqual(result["best"].separator, "!")

    async def test_nonfinite_score_propagates(self):
        async def evaluate(context):
            return float("nan")

        with self.assertRaisesRegex(ValueError, "finite"):
            await RandomPrompt("task", None, evaluate, decode=lambda _: "!").run(
                "{separator}", draws=1
            )
