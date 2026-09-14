"""Offline checks for EvoX's coupled evolution loops and execution boundary."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts

from evox import Config, Evaluation, EvoX
from evox.runtime import run_python_strategy
from tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1] / "evox"
OPERATORS = '{"refine": "Polish the wording.", "diverge": "Try a different structure."}'
GREEDY = """def select(population, state, rng):
    parent = max(population, key=lambda p: p["quality"])
    return {"parent_id": parent["id"], "operator": "refine", "inspiration_ids": []}
"""


def strategy(code):
    import json

    return json.dumps({"code": code})


class EvoXTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", ROOT / "prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_stagnation_mutates_code_preserves_population_and_truncates_window(self):
        provider = ScriptedProvider(["flat", "flat", "better", "best", "last"])
        meta = ScriptedProvider([strategy(GREEDY)])
        guide = ScriptedProvider([OPERATORS])

        async def evaluate(text):
            return Evaluation({"flat": 1, "better": 2, "best": 3, "last": 2}[text], {"log": text})

        agent = EvoX(
            "Write any artifact",
            provider,
            evaluate,
            run_strategy=run_python_strategy,
            strategy_provider=meta,
            operator_provider=guide,
            initial_population=[("seed", Evaluation(1))],
            config=Config(iterations=5, window=2),
        )
        result = await agent.run()
        self.assertEqual(result.best.text, "best")
        self.assertEqual((result.steps, result.evaluation_calls, len(result.candidates)), (5, 5, 6))
        self.assertEqual([w.steps for w in result.windows], [2, 2, 1])
        self.assertEqual([w.strategy.id for w in result.windows], [0, 1, 1])
        self.assertAlmostEqual(result.windows[1].score, 2 * (1 + math.log(2)) / math.sqrt(2))
        self.assertEqual(len(meta.calls), 1)  # No unused meta generation after the budget.
        self.assertIn("seed", provider.calls[2])
        self.assertIn("Polish the wording.", provider.calls[2])
        self.assertIn("parent_counts", meta.calls[0])
        self.assertIn("start_state", meta.calls[0])
        self.assertTrue(result.strategy_attempts[0].accepted)

    async def test_invalid_strategies_retry_then_retain_previous_strategy(self):
        meta = ScriptedProvider(
            [
                "not json",
                strategy("def select(:"),
                strategy(
                    "def select(population, state, rng):\n"
                    '    return {"parent_id": 999, "operator": "refine", "inspiration_ids": []}'
                ),
            ]
        )

        async def evaluate(text):
            return Evaluation(1)

        agent = EvoX(
            "Choose a greeting",
            ScriptedProvider(["one", "two"]),
            evaluate,
            run_strategy=run_python_strategy,
            strategy_provider=meta,
            operator_provider=ScriptedProvider([OPERATORS]),
            initial_population=[("seed", Evaluation(1))],
            config=Config(iterations=2, window=1),
        )
        result = await agent.run()
        self.assertEqual(len(result.strategy_attempts), 3)
        self.assertTrue(all(a.error and not a.accepted for a in result.strategy_attempts))
        self.assertEqual([w.strategy.id for w in result.windows], [0, 0])
        self.assertTrue(any(r.response == "not json" for r in result.generations))
        self.assertIn("Previous validation failures", meta.calls[-1])

    async def test_empty_population_rejections_and_minimization(self):
        seen = []

        async def evaluate(text):
            seen.append(text)
            if text == "invalid":
                raise ValueError("not feasible")
            return Evaluation({"nan": math.nan, "first": -2, "best": -5}[text])

        provider = ScriptedProvider([" ", "invalid", "nan", "first", "best"])
        agent = EvoX(
            "Minimize a cost",
            provider,
            evaluate,
            run_strategy=run_python_strategy,
            operator_provider=ScriptedProvider([OPERATORS]),
            config=Config(iterations=5, window=5, maximize=False),
        )
        result = await agent.run()
        self.assertEqual(result.best.score, -5)
        self.assertEqual((result.steps, result.evaluation_calls), (5, 4))
        self.assertEqual(len([c for c in result.candidates if c.error]), 3)
        self.assertEqual(seen, ["invalid", "nan", "first", "best"])
        self.assertIn("minimize", provider.calls[-1])
        self.assertTrue(math.isfinite(result.windows[0].score))

    async def test_transport_and_programming_errors_propagate(self):
        async def evaluate(text):
            raise RuntimeError("broken evaluator")

        for response, error in [
            ("candidate", "broken evaluator"),
            (RuntimeError("offline"), "offline"),
        ]:
            agent = EvoX(
                "Task",
                ScriptedProvider([response]),
                evaluate,
                run_strategy=run_python_strategy,
                operator_provider=ScriptedProvider([OPERATORS]),
                config=Config(iterations=1),
            )
            with self.assertRaisesRegex(RuntimeError, error):
                await agent.run()

    async def test_worker_timeout_mutation_and_selection_contract(self):
        population = [{"id": 0, "quality": 1, "artifacts": {"log": "keep"}}]
        state = {"inspirations": 4}
        selection = await run_python_strategy(GREEDY, population, state, 0)
        self.assertEqual(selection["parent_id"], 0)
        for code, message in [
            ("def select(population, state, rng):\n    while True: pass", "timed out"),
            (
                'def select(population, state, rng):\n    population[0]["quality"] = 50\n'
                "    return {}",
                "modified",
            ),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                await run_python_strategy(code, population, state, 0, timeout=0.2)
        self.assertEqual(population[0]["quality"], 1)

    async def test_strategy_validation_descriptors_match_each_trial_population(self):
        code = """def select(population, state, rng):
    assert state["size"] == len(population)
    assert state["best"] == max(p["quality"] for p in population)
    assert all("strategy_id" in p and "error" in p for p in population)
    return {"parent_id": state["frontier"][0]["id"],
            "operator": "free", "inspiration_ids": []}
"""

        async def evaluate(text):
            return Evaluation(0)

        for seeds in [[], [("lower", Evaluation(1)), ("best", Evaluation(2))]]:
            agent = EvoX(
                "Task",
                ScriptedProvider([]),
                evaluate,
                run_strategy=run_python_strategy,
                initial_population=seeds,
                initial_strategy=code,
                config=Config(iterations=0),
            )
            result = await agent.run()
            self.assertEqual(result.strategies[0].code, code)

    async def test_all_operation_templates_render_from_another_directory(self):
        async def evaluate(text):
            return Evaluation(1)

        agent = EvoX("Task", ScriptedProvider([]), evaluate, run_strategy=run_python_strategy)
        import os
        import tempfile

        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                rendered = [
                    await EvoX.initialize.render(agent),
                    await EvoX.prepare_operators.render(agent),
                    await EvoX.refine.render(agent, {}, [], "guidance"),
                    await EvoX.diverge.render(agent, {}, [], "guidance"),
                    await EvoX.vary.render(agent, {}, []),
                    await EvoX.mutate_strategy.render(agent, {}, [], {}, []),
                ]
        finally:
            os.chdir(previous)
        self.assertTrue(all("Task" in text for text in rendered))
        for path in (ROOT / "prompts").glob("*.j2"):
            tree = Environment().parse(path.read_text())
            self.assertFalse(list(tree.find_all((nodes.If, nodes.CondExpr))))


if __name__ == "__main__":
    unittest.main()
