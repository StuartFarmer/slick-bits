import random
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from medprompt import Example, MedPrompt, Question
from tests.providers import ScriptedProvider


class MedPromptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "medprompt/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_correct_pool_knn_recency_and_inverted_choice_votes(self):
        async def embed(inputs):
            return np.array([[1.0, float(text == "far")] for text in inputs])

        async def evaluate(query, answer):
            return float(answer == 1)

        training = [Example(Question(text, ("x", "y")), 0) for text in ("near", "far", "wrong")]
        # Reproduce only the public seeded shuffle contract; all votes choose original index 1.
        rng = random.Random(4)
        responses = ["Reason.\nAnswer: [A]", "Reason.\nAnswer: [A]", "Bad.\nAnswer: [B]"]
        for _ in range(3):
            order = [0, 1]
            rng.shuffle(order)
            rng.choice([0])
            rng.choice([0])
            responses.append(f"Reason.\nAnswer: [{chr(65 + order.index(1))}]")
        provider = ScriptedProvider(responses)
        result = await MedPrompt("task", provider, evaluate, embed).run(
            training, [Question("query", ("x", "y"))], neighbors=2, ensemble=3, seed=4
        )
        self.assertEqual(len(result["pool"]), 2)
        self.assertEqual(result["predictions"][0]["answer"], 1)
        self.assertEqual(result["predictions"][0]["neighbors"], [1, 0])
        self.assertLess(provider.calls[3].index("far"), provider.calls[3].index("near"))
        self.assertEqual(result["generations"], 6)
        self.assertEqual(result["evaluations"], 1)

    async def test_no_correct_rationales_stops_before_evaluation(self):
        agent = MedPrompt("task", ScriptedProvider(["[B]"]), None, None)
        with self.assertRaisesRegex(ValueError, "no correct"):
            await agent.run([Example(Question("q", ("x", "y")), 0)], [])
        self.assertEqual(agent.evaluations, 0)

    def test_ambiguous_choices_are_rejected(self):
        self.assertIsNone(MedPrompt._parse("[A] or [B]", 2))
