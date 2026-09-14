import unittest
from pathlib import Path
from unittest.mock import patch

from llm_bo import LLMBO, Configurations
from tests.providers import ScriptedProvider


class LLMBOTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "llm_bo/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_outcome_conditioning_ei_uncertainty_and_actual_measurement(self):
        evaluated = []

        async def evaluate(point):
            evaluated.append(point["x"])
            return {0.0: 1.0, 1.0: 2.0, 2.0: 0.5, 3.0: 0.8}[point["x"]]

        provider = ScriptedProvider(
            [
                Configurations(configurations=[{"x": 0}, {"x": 1}]),
                Configurations(configurations=[{"x": 2}, {"x": 3}, {"x": 9}]),
                "## 0.9 ##",
                "## 0.9 ##",
                "## -1 ##",
                "## 3 ##",
            ]
        )
        result = await LLMBO("Task", provider, evaluate, {"x": (0, 3)}).run(
            iterations=1, warm_start_count=2, candidates=2, predictions=2, jitter=False
        )
        self.assertEqual(evaluated, [0.0, 1.0, 3.0])
        self.assertAlmostEqual(result["history"][0]["target"], 1.2)
        self.assertEqual(result["best"].value, 0.8)
        self.assertEqual(result["rejections"][0]["reason"], "bounds")
        self.assertEqual(result["optimizer_calls"], 6)

    async def test_duplicate_sampling_stops_after_attempt_budget(self):
        async def evaluate(point):
            return 1

        duplicate = Configurations(configurations=[{"x": 0}])
        provider = ScriptedProvider([duplicate, duplicate])
        result = await LLMBO("Task", provider, evaluate, {"x": (0, 1)}).run(
            [{"x": 0}], iterations=4, sampling_attempts=2
        )
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["optimizer_calls"], 2)
        self.assertEqual(len(result["rejections"]), 2)
