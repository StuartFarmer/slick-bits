import unittest
from pathlib import Path
from unittest.mock import patch

from tests.providers import ScriptedProvider
from vlm_blackbox import VLMBlackBox


class VLMBlackBoxTests(unittest.IsolatedAsyncioTestCase):
    async def test_extreme_pools_deduplication_and_required_token(self):
        seen = []

        async def evaluate(text):
            seen.append(text)
            return {"low X": 0, "mid X": 1, "high X": 2, "new X": 3}[text]

        provider = ScriptedProvider(
            [
                '{"analysis":"contrast","templates":["new X"]}',
                '{"analysis":"contrast","templates":["new X"]}',
                '{"analysis":"bad","templates":["lost"]}',
            ]
        )
        with patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "vlm_blackbox/prompts"
        ):
            agent = VLMBlackBox("task", provider, evaluate, required_tokens=("X",))
            result = await agent.run(["low X", "mid X", "high X"], rounds=2, pool_size=1)
            self.assertEqual(result["best"], "new X")
            self.assertEqual(len(seen), 4)
            self.assertNotIn("mid X", provider.calls[0])
            self.assertIn("new X", provider.calls[1])
            with self.assertRaisesRegex(ValueError, "required token"):
                await agent.contrast([], [], 1, provider=provider)
