import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auto_cot import AutoCoT
from tests.providers import ScriptedProvider


class AutoCoTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "auto_cot/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_nearest_inadmissible_trace_falls_through_without_gold_filter(self):
        async def embed(inputs):
            return np.array([[0.0], [1.0], [9.0]])

        async def evaluate(demos):
            return len(demos)

        provider = ScriptedProvider(
            ["Missing period", "2", "It equals 99.", "99", "It is 4.", "4", "prediction"]
        )
        agent = AutoCoT("arithmetic", provider, evaluate, embed)
        result = await agent.run(["one", "two", "three"], clusters=2, arithmetic=True)
        self.assertEqual({d.question for d in result["demonstrations"]}, {"two", "three"})
        self.assertEqual(result["generations"], 6)
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(await agent.answer("query", provider=provider), "prediction")
        self.assertIn("It equals 99.", provider.calls[-1])

    async def test_all_clusters_can_be_omitted(self):
        async def embed(inputs):
            return np.ones((len(inputs), 1))

        async def evaluate(demos):
            self.assertEqual(demos, ())
            return 0

        result = await AutoCoT("task", ScriptedProvider(["bad", ""]), evaluate, embed).run(
            ["q"], clusters=1
        )
        self.assertEqual(result["indices"], [])

    async def test_provider_failure_stops_before_evaluation(self):
        async def embed(inputs):
            return np.ones((1, 1))

        agent = AutoCoT("task", ScriptedProvider([RuntimeError("offline")]), None, embed)
        with self.assertRaisesRegex(RuntimeError, "offline"):
            await agent.run(["q"], clusters=1)
        self.assertEqual(agent.evaluations, 0)
