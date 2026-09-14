import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from selective_annotation import SelectiveAnnotation
from tests.providers import ScriptedProvider


class SelectiveAnnotationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT",
            Path(__file__).parents[1] / "selective_annotation/prompts",
        )
        root.start()
        self.addCleanup(root.stop)

    async def test_fast_vote_discount_selects_two_regions(self):
        async def embed(inputs):
            return np.array([[1.0, 0.1], [1.0, -0.1], [-1.0, 0.1], [-1.0, -0.1]])

        annotated = []

        async def annotate(input):
            annotated.append(input)
            return "label " + input

        async def evaluate(examples):
            return len(examples)

        provider = ScriptedProvider(["prediction"])
        agent = SelectiveAnnotation("task", provider, evaluate, embed, annotate)
        result = await agent.run(list("abcd"), budget=2, mode="fast_votek", vote_neighbors=1)
        self.assertEqual(result["indices"], [0, 2])
        self.assertEqual(annotated, ["a", "c"])
        self.assertEqual(result["uncertainty_calls"], 0)
        await agent.answer("q", result["examples"], provider=provider)
        self.assertIn("label c", provider.calls[0])

    async def test_full_vote_uses_frozen_initial_labels_and_uncertainty_buckets(self):
        async def embed(inputs):
            return np.array([[np.cos(i), np.sin(i)] for i in range(6)])

        annotated, contexts = [], []

        async def annotate(input):
            annotated.append(input)
            return input

        async def uncertainty(demos, input):
            self.assertEqual(len(annotated), 1)
            contexts.append(demos)
            return float(input)

        async def evaluate(examples):
            return 1.0

        result = await SelectiveAnnotation("t", None, evaluate, embed, annotate, uncertainty).run(
            list("012345"), budget=3, initial=1, vote_neighbors=1, neighbors=1
        )
        self.assertEqual(result["annotations"], 3)
        self.assertEqual(result["uncertainty_calls"], 5)
        first_bucket = [i for i, _ in result["uncertainties"][:2]]
        self.assertIn(result["indices"][1], first_bucket)
        self.assertTrue(all(len(demos) <= 1 for demos in contexts))

    async def test_annotation_failure_propagates(self):
        async def embed(inputs):
            return np.ones((2, 1))

        async def annotate(input):
            raise RuntimeError("oracle offline")

        agent = SelectiveAnnotation("t", None, None, embed, annotate)
        with self.assertRaisesRegex(RuntimeError, "oracle offline"):
            await agent.run(["a", "b"], budget=1)
        self.assertEqual(agent.evaluations, 0)
