import json
import unittest
from pathlib import Path
from unittest.mock import patch

from generative_elicitation import GenerativeElicitation
from tests.providers import ScriptedProvider


class ElicitationTests(unittest.IsolatedAsyncioTestCase):
    async def test_answers_inform_next_question_and_hypothesis(self):
        async def oracle(query):
            return "allow" if query == "case one?" else "deny"

        responses = [
            {"text": "case one?"},
            {"text": "case two?"},
            {"text": "instruction"},
            {"text": "prediction"},
            {"text": "edge?"},
            {"text": "updated"},
        ]
        provider = ScriptedProvider([json.dumps(x) for x in responses])
        original = [("prior?", "prior answer")]
        with patch(
            "slick.prompts.TEMPLATE_ROOT",
            Path(__file__).parents[1] / "generative_elicitation/prompts",
        ):
            agent = GenerativeElicitation("task", provider, oracle)
            result = await agent.run(queries=2, history=original)
            self.assertIn("allow", provider.calls[1])
            self.assertIn("deny", provider.calls[2])
            self.assertEqual(len(original), 1)
            self.assertEqual(
                await agent.predict(result["history"], "new", provider=provider), "prediction"
            )
            result = await agent.run(queries=1, mode="edge_case")
            self.assertEqual(result["history"], [("edge?", "deny")])
