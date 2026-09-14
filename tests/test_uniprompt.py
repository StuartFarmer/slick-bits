import unittest
from pathlib import Path
from unittest.mock import patch

from tests.providers import ScriptedProvider
from uniprompt import UniPrompt
from uniprompt.agent import Groups


class UniPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_feedback_aggregation_and_elitist_beam(self):
        provider = ScriptedProvider(
            ["facet", Groups(groups=[[0]]), "minibatch feedback", "set section", "better"]
        )

        async def errors(instruction, batch):
            return ["expected X, got Y"]

        async def evaluate(instruction):
            return float(instruction == "better")

        with patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "uniprompt/prompts"):
            result = await UniPrompt("Task", provider, errors, evaluate).run(
                "initial", ["input"], epochs=1, epsilon=0
            )
        self.assertEqual(result["best"]["prompt"], "better")
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(result["history"][0]["edits"][0][0]["delta"], 1)
        self.assertIn("Minibatch feedback", provider.calls[3])

    async def test_rejects_generated_partition_duplicates(self):
        provider = ScriptedProvider([Groups(groups=[[0, 0]])])
        with patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "uniprompt/prompts"):
            agent = UniPrompt("Task", provider, None, None)
            with self.assertRaisesRegex(ValueError, "partition"):
                await agent.cluster(feedbacks=["a", "b"], group_count=2, provider=provider)
