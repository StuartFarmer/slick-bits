"""Probe agreement controls the expensive model's evaluation allocation."""

import unittest
from pathlib import Path
from unittest.mock import patch

from probe_sampling import ProbeSampling
from tests.providers import ScriptedProvider


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_agreement_filters_but_keeps_probe_winner(self):
        async def score(prompt):
            return float(prompt)

        provider = ScriptedProvider(['{"prompts":["1","2","3","4"]}'])
        with patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "probe_sampling/prompts"
        ):
            result = await ProbeSampling("Minimize", provider, score, score).run(
                "10", iterations=1, candidates=4, probe_size=2, filtered_size=4
            )
        self.assertEqual(result["best"]["prompt"], "1")
        self.assertEqual(result["history"][0]["retained"], 1)
        self.assertAlmostEqual(result["history"][0]["correlation"], 1.0)
        self.assertLessEqual(result["target_evaluations"], 4)
        self.assertEqual(result["draft_evaluations"], 4)

    async def test_flat_draft_retains_full_filter_budget(self):
        async def draft(prompt):
            return 0.0

        async def target(prompt):
            return float(prompt)

        provider = ScriptedProvider(['{"prompts":["1","2","3"]}'])
        with patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "probe_sampling/prompts"
        ):
            result = await ProbeSampling("Minimize", provider, draft, target).run(
                "10", iterations=1, candidates=3, probe_size=2, filtered_size=3
            )
        self.assertEqual(result["best"]["prompt"], "1")
        self.assertEqual(result["target_evaluations"], 4)
        self.assertIsNone(result["history"][0]["correlation"])
