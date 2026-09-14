import json
import unittest
from pathlib import Path
from unittest.mock import patch

from optimus import Execution, OptiMUS
from tests.providers import ScriptedProvider


class OptiMUSTests(unittest.IsolatedAsyncioTestCase):
    async def test_distinct_execution_logic_repairs_and_data_separation(self):
        outputs = iter(
            [
                Execution("execution_error", feedback="syntax"),
                Execution("test_failure", feedback="constraint"),
                Execution("passed", output=42),
            ]
        )
        seen = []

        async def execute(code, tests, data):
            seen.append((code, tests, data))
            return next(outputs)

        provider = ScriptedProvider(
            [json.dumps({"text": text}) for text in ["model", "code0", "tests", "code1", "code2"]]
        )
        with patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "optimus/prompts"):
            result = await OptiMUS(provider, execute, interface="contract").run(
                "problem", "PRIVATE DATA", repairs=2
            )
        self.assertEqual(result["result"]["execution"].output, 42)
        self.assertEqual(result["execution_calls"], 3)
        self.assertTrue(all(t == "tests" and d == "PRIVATE DATA" for _, t, d in seen))
        self.assertNotIn("PRIVATE DATA", "\n".join(provider.calls))
        self.assertIn("execution error", provider.calls[3])
        self.assertIn("semantic tests", provider.calls[4])

    async def test_exhausted_variant_is_rephrased_and_human_tests_stay_fixed(self):
        async def execute(code, tests, data):
            self.assertEqual(tests, "human tests")
            return Execution("passed" if code == "code1" else "test_failure")

        provider = ScriptedProvider(
            [json.dumps({"text": t}) for t in ["model0", "code0", "rephrased", "model1", "code1"]]
        )
        with patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "optimus/prompts"):
            result = await OptiMUS(provider, execute, interface="contract").run(
                "problem", {}, test_code="human tests", repairs=0, augmentations=1
            )
        self.assertEqual(len(result["variants"]), 2)
        self.assertEqual(result["result"]["problem"], "rephrased")
