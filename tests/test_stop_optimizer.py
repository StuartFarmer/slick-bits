import json
import unittest
from pathlib import Path
from unittest.mock import patch

from stop_optimizer import (
    SEED_IMPROVER,
    STOP,
    BudgetExhausted,
    Capabilities,
    ImproverExecutionError,
    Problem,
)
from tests.providers import ScriptedProvider


class STOPTests(unittest.IsolatedAsyncioTestCase):
    async def test_recursive_improver_replacement_meta_utility_and_invalid_fallback(self):
        invocations = []

        async def evaluate(source):
            return 2.0 if source == "good solution" else 0.0

        async def execute(source, initial, caps):
            invocations.append((source, initial))
            if source == initial and source == SEED_IMPROVER:
                candidate = await caps.suggest(initial)
                await caps.evaluate(candidate)
                return candidate
            if initial == "base solution":
                return await caps.suggest(initial, "use the evolved strategy")
            return ""

        provider = ScriptedProvider(
            [json.dumps({"code": c}) for c in ["new optimizer", "good solution", "good solution"]]
        )
        with patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "stop_optimizer/prompts"
        ):
            result = await STOP(
                "task", provider, execute, [Problem("base solution", "utility", evaluate)]
            ).run(rounds=2)
        self.assertEqual(result["improver"], "new optimizer")
        self.assertIn(("new optimizer", "new optimizer"), invocations)
        self.assertEqual(result["history"][0]["checked_score"], 2.0)
        self.assertIsNone(result["history"][1]["checked_score"])
        self.assertEqual(result["meta_calls"], 2)
        self.assertIn("evolved strategy", provider.calls[1])

    async def test_budget_fails_before_call(self):
        async def evaluate(source):
            self.fail("budget should prevent callback")

        with self.assertRaises(BudgetExhausted):
            await Capabilities(None, evaluate, 0, 0).evaluate("source")

    async def test_failed_self_application_rolls_back_executable_only(self):
        calls = []

        async def evaluate(source):
            return 1.0

        async def execute(source, initial, caps):
            calls.append((source, initial))
            if initial == "problem":
                return "solution"
            if source == "A":
                raise ImproverExecutionError("cannot improve itself")
            return "A" if initial == "seed" else "B"

        result = await STOP(
            "task", ScriptedProvider([]), execute, [Problem("problem", "utility", evaluate)]
        ).run("seed", rounds=3)
        self.assertEqual(result["improver"], "B")
        self.assertIn(("seed", "A"), calls)
        self.assertEqual(calls.count(("A", "A")), 1)
