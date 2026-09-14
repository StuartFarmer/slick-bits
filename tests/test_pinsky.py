import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from jinja2 import Environment, nodes
from slick import prompts

from pinsky import PINSKY, Config, Evaluation, Pair
from tests.providers import ScriptedProvider


class PinskyTests(unittest.IsolatedAsyncioTestCase):
    async def test_fractional_rewards_after_integer_initialization_and_partial_generation(self):
        rewards = iter([0, 0, 0, 0, 0.1, 0.9, 0.2])
        evaluated = []

        async def evaluate(environment, parameters):
            evaluated.append(parameters.copy())
            return Evaluation(next(rewards), False)

        search = PINSKY("Task", None, evaluate, random_solve=None, strong_solve=None)
        pair = Pair(0, None, 0, "seed", np.zeros(2))
        await search.optimize(pair, Config(population_size=4, de_evaluations=3))
        np.testing.assert_array_equal(pair.parameters, evaluated[5])
        self.assertEqual(search.result.evaluation_calls, 7)

    async def test_de_improves_numeric_agent_with_exact_budget_and_isolation(self):
        async def evaluate(environment, parameters):
            score = -float(np.sum((parameters - float(environment)) ** 2))
            parameters[:] = 999  # Evaluators must not corrupt optimizer state.
            return Evaluation(score, score > -0.01)

        search = PINSKY("Fit targets", None, evaluate, random_solve=None, strong_solve=None)
        initial = np.array([4.0, 4.0])
        result = await search.run(
            "0.3",
            initial,
            config=Config(iterations=2, max_children=0, population_size=8, de_evaluations=160),
        )
        self.assertGreater(result.active[0].evaluation.score, -0.01)
        self.assertEqual(result.evaluation_calls, 1 + 2 * (8 + 160 + 1))
        np.testing.assert_array_equal(initial, [4, 4])

    async def test_viability_inheritance_global_attempt_budget_and_oldest_culling(self):
        candidates = iter(["easy", "hard", "child", "grandchild"])
        checks = []

        async def mutate(environment, operation, rng):
            return next(candidates)

        async def random_solve(environment):
            checks.append(("random", environment))
            return environment == "easy"

        async def strong_solve(environment):
            checks.append(("strong", environment))
            return environment != "hard"

        async def evaluate(environment, parameters):
            return Evaluation(float(parameters[0]), False)

        search = PINSKY(
            "Task",
            None,
            evaluate,
            random_solve=random_solve,
            strong_solve=strong_solve,
            mutate=mutate,
            seed=2,
        )
        result = await search.run(
            "seed",
            np.array([4.0]),
            config=Config(
                iterations=2,
                mutation_timer=1,
                max_children=2,
                max_environments=1,
                mutation_rate=1,
                continuation_rate=0,
                population_size=4,
                de_evaluations=0,
            ),
        )
        self.assertEqual(len(result.attempts), 4)
        self.assertEqual(len(checks), 8)
        self.assertEqual([a.accepted for a in result.attempts], [False, False, True, True])
        self.assertEqual(result.active[0].environment, "grandchild")
        self.assertEqual([p.environment for p in result.retired], ["seed", "child"])
        self.assertEqual(result.active[0].parent_id, 0)  # Both draw from phase-start parents.
        self.assertEqual(result.active[0].parameters[0], 4)
        self.assertFalse(
            np.shares_memory(result.active[0].parameters, result.retired[0].parameters)
        )

    async def test_transfer_uses_snapshot_and_strict_improvements(self):
        # Swapping both agents fails if replacements are applied during evaluation.
        async def evaluate(environment, parameters):
            return Evaluation(float(parameters[0] if environment == "a" else -parameters[0]), True)

        search = PINSKY("Task", None, evaluate, random_solve=None, strong_solve=None)
        search.result.active = [
            Pair(0, None, 0, "a", np.array([0.0]), Evaluation(999, False)),
            Pair(1, 0, 0, "b", np.array([1.0]), Evaluation(999, False)),
        ]
        await search.transfer(9)
        self.assertEqual([p.parameters[0] for p in search.result.active], [1, 0])
        self.assertEqual(search.result.evaluation_calls, 4)
        self.assertEqual(
            [(t.source_id, t.target_id) for t in search.result.transfers], [(1, 0), (0, 1)]
        )
        await search.transfer(19)
        self.assertEqual(len(search.result.transfers), 2)

    async def test_all_slick_operations_rejections_raw_records_and_external_root(self):
        async def evaluate(environment, parameters):
            return Evaluation(0, False)

        provider = ScriptedProvider(
            [
                '{"environment":"removed"}',
                '{"environment":"added"}',
                '{"environment":"moved"}',
                '{"environment":" "}',
                "not json",
            ]
        )
        search = PINSKY("Generic task", provider, evaluate, random_solve=None, strong_solve=None)
        root = Path(__file__).resolve().parents[1] / "pinsky" / "prompts"
        with patch.object(prompts, "TEMPLATE_ROOT", root), tempfile.TemporaryDirectory() as cwd:
            with patch("os.getcwd", return_value=cwd):
                for name in ("remove", "add", "move"):
                    rendered = await getattr(PINSKY, name).render(search, "seed")
                    self.assertIn("Generic task", rendered)
                    self.assertIn('"environment"', rendered)
                    parsed = Environment().parse((root / f"{name}.j2").read_text())
                    self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
                for name, expected in zip(("remove", "add", "move"), ("removed", "added", "moved")):
                    self.assertEqual(await search.edit("seed", name), expected)
                result = await search.run(
                    "seed",
                    np.array([0.0]),
                    config=Config(
                        iterations=1,
                        max_children=2,
                        mutation_rate=1,
                        continuation_rate=0,
                        population_size=4,
                        de_evaluations=0,
                    ),
                )
        self.assertTrue(all(a.error for a in result.attempts))
        self.assertEqual(len(result.generations), 5)
        self.assertEqual(result.generations[-1][1], "not json")
        self.assertEqual(len(result.active), 1)

    async def test_nonfinite_measurement_and_infrastructure_errors_propagate(self):
        async def evaluate(environment, parameters):
            return Evaluation(float("nan"), False)

        search = PINSKY("Task", None, evaluate, random_solve=None, strong_solve=None)
        with self.assertRaisesRegex(ValueError, "finite"):
            await search.run("seed", np.zeros(1), config=Config(iterations=0))
        self.assertEqual(search.result.evaluation_calls, 1)

        search.provider = ScriptedProvider([RuntimeError("offline")])
        with patch.object(
            prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[1] / "pinsky/prompts"
        ):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                await search.edit("seed", "add")

    async def test_zero_mutation_gate_retains_source_clone_behavior(self):
        async def solve(environment):
            return True

        async def random_solve(environment):
            return False

        async def evaluate(environment, parameters):
            return Evaluation(0, False)

        provider = ScriptedProvider([])
        search = PINSKY("Task", provider, evaluate, random_solve=random_solve, strong_solve=solve)
        result = await search.run(
            "seed",
            np.zeros(1),
            config=Config(
                iterations=1,
                max_children=1,
                mutation_rate=0,
                population_size=4,
                de_evaluations=0,
                transfer_timer=1,
            ),
        )
        self.assertEqual(len(provider.calls), 0)
        self.assertEqual([p.environment for p in result.active], ["seed", "seed"])
        self.assertEqual(result.evaluation_calls, 1 + 2 * 5 + 4)
