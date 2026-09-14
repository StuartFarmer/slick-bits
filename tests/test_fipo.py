"""FIPO's optional response modules preserve the learned model's wire format."""

import unittest
from pathlib import Path
from unittest.mock import patch

from fipo import FIPO
from tests.providers import ScriptedProvider


class FIPOTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_module_combinations(self):
        async def evaluate(text):
            return len(text)

        for response, reference in (
            (None, None),
            ("attempt", None),
            (None, "answer"),
            ("attempt", "answer"),
        ):
            provider = ScriptedProvider(["Golden Prompt: improved"])
            with patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "fipo/prompts"):
                result = await FIPO("original", provider, evaluate).run(
                    response=response, reference=reference, max_words=42
                )
            self.assertEqual(result["prompt"], "improved")
            context = provider.calls[0]
            self.assertIn("less than 42 words", context)
            self.assertEqual("Sliver Response:\n" in context, response is not None)
            self.assertEqual("Golden Response:\n" in context, reference is not None)
            self.assertIn("Sliver Prompt:\noriginal", context)
