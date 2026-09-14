import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eureka import Eureka, TrainingResult
from tests.providers import ScriptedProvider


class EurekaTests(unittest.IsolatedAsyncioTestCase):
    async def test_numpy_training_history_is_snapshotted(self):
        values = np.array([1.0, 2.0])

        async def train(code):
            return TrainingResult(1.0, {"component": values})

        trial = await Eureka("task", ScriptedProvider([]), train, interface="contract").assess(
            "code"
        )
        values[:] = 99.0
        self.assertEqual(trial.result.components["component"], (1.0, 2.0))
        self.assertEqual(trial.feedback["components"]["component"]["mean"], 1.5)

    async def test_batch_context_is_not_global_best_and_all_failure_keeps_context(self):
        scores = {"a": 5, "b": 3, "c": 1, "d": 2, "g": 6, "h": 4}

        async def train(code):
            return (
                TrainingResult(error="failed")
                if code in "ef"
                else TrainingResult(scores[code], {"distance": [1.0, 2.0, 3.0]})
            )

        provider = ScriptedProvider([json.dumps({"code": c}) for c in "abcdefgh"])
        with patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "eureka/prompts"):
            result = await Eureka("task", provider, train, interface="interface").run(
                rounds=4, candidates=2
            )
        self.assertEqual(result["best"].code, "g")
        self.assertEqual(result["training_calls"], 8)
        self.assertIn("selected program:\na", provider.calls[2])
        self.assertIn("selected program:\nd", provider.calls[4])
        self.assertIn("selected program:\nd", provider.calls[6])
        self.assertEqual(result["trials"][0].feedback["components"]["distance"]["mean"], 2.0)
