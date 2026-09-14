import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bo_lift import BOLIFT, Observation
from bo_lift.agent import mmr
from tests.providers import ScriptedProvider


class BOLIFTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "bo_lift/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_discrete_ei_selects_uncertain_candidate(self):
        async def evaluate(text):
            self.assertEqual(text, "risky")
            return 5

        async def embed(texts):
            return np.array([[1.0, i + 1.0] for i, _ in enumerate(texts)])

        provider = ScriptedProvider(["2###", "2###", "0###", "6###"])
        result = await BOLIFT("Task", provider, evaluate, embed).run(
            ["steady", "risky"],
            [Observation("a", 1), Observation("b", 2)],
            iterations=1,
            predictions=2,
            acquisition="ei",
            inverse_count=0,
        )
        self.assertEqual(result["best"].candidate, "risky")
        self.assertEqual(result["history"][0]["estimates"][1]["acquisition"], 2)
        self.assertEqual(result["optimizer_calls"], 4)

    async def test_inverse_filter_and_mmr_diversity(self):
        async def evaluate(text):
            return 4

        async def embed(texts):
            values = {
                "ideal": [1.0, 0],
                "near": [1.0, 0],
                "far": [0.0, 1],
                "a": [1.0, 1],
                "b": [-1.0, 0],
            }
            return np.array([values[text] for text in texts])

        provider = ScriptedProvider(["ideal###", "3###"])
        result = await BOLIFT("Task", provider, evaluate, embed).run(
            ["near", "far"],
            [Observation("a", 1), Observation("b", 2)],
            iterations=1,
            predictions=1,
            inverse_count=1,
        )
        self.assertEqual(result["best"].candidate, "near")
        self.assertEqual(result["optimizer_calls"], 2)
        indices = mmr(np.array([[1.0, 0], [0.99, 0.01], [0.0, 1]]), np.array([1.0, 0]), 2, 0)
        self.assertEqual(indices, [0, 2])
