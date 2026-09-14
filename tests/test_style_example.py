"""Check the reusable example with the repository's shared scripted provider."""

import asyncio
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts

from tests.providers import ScriptedProvider

ASSETS = Path(__file__).resolve().parents[1] / "skills/slick-development/assets"
SPEC = importlib.util.spec_from_file_location("checked_proposal", ASSETS / "checked_proposal.py")
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


class StyleExampleChecks(unittest.TestCase):
    def test_generic_proposal_and_revision(self):
        async def check():
            provider = ScriptedProvider(
                [
                    '{"description":"Initial draft", "content":"long"}',
                    '{"description":"Short revision", "content":"a"}',
                ]
            )

            async def evaluate(content):
                return -len(content)

            agent = EXAMPLE.ProposalDesigner("Rewrite a welcome message.", provider, evaluate)
            self.assertEqual(provider.calls, [])
            with patch.object(prompts, "TEMPLATE_ROOT", ASSETS / "prompts"):
                rendered = await EXAMPLE.ProposalDesigner.propose.render(agent, "Be concise.")
                self.assertIn(agent.task, rendered)
                result = await agent.run(["Be concise.", "Shorter still."])
            self.assertEqual(result.proposal.content, "a")
            self.assertEqual(result.score, -1)
            self.assertIn("Create a candidate", provider.calls[0])
            self.assertIn("Revise this candidate", provider.calls[1])
            self.assertIn('"content": "long"', provider.calls[1])

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
