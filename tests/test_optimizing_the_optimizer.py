"""Offline checks for the paper's code-improvement workflow and generic CMSA."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts

from optimizing_the_optimizer import (
    CMSA,
    Evaluation,
    OptimizingTheOptimizer,
    Proposal,
    selection_probabilities,
)
from tests.providers import ScriptedProvider


class OptimizerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "optimizing_the_optimizer/prompts"
        self.root = root
        replacement = patch.object(prompts, "TEMPLATE_ROOT", root)
        replacement.start()
        self.addCleanup(replacement.stop)

    async def test_dialogue_measures_baseline_and_branches_performance(self):
        provider = ScriptedProvider(
            Proposal(code=code, rationale="Use an overlooked state variable")
            for code in ("v1", "v1-perf", "v2", "v2-perf")
        )
        measured = []

        async def evaluate(code):
            measured.append(code)
            return Evaluation({"base": 1, "v1": 5, "v1-perf": 4, "v2": 3, "v2-perf": 2}[code])

        agent = OptimizingTheOptimizer(
            "Optimize an arbitrary scheduling algorithm", provider, evaluate
        )
        result = await agent.run("base", target="construct", rounds=2, performance=True)
        self.assertEqual(result.code, "v1")
        self.assertEqual(measured, ["base", "v1", "v1-perf", "v2", "v2-perf"])
        self.assertEqual(
            [a["operation"] for a in agent.attempts],
            ["improve_heuristic", "optimize_code", "revise", "optimize_code"],
        )
        self.assertEqual(agent.attempts[2]["source"], "v1")
        self.assertIn("v1-perf", provider.calls[2])
        self.assertTrue(all("construct" in text and agent.task in text for text in provider.calls))
        self.assertTrue(all(a["response"] for a in agent.attempts))
        self.assertEqual(agent.evaluations, 5)

    async def test_repair_feedback_raw_logging_and_minimization(self):
        provider = ScriptedProvider(
            [
                "not JSON",
                Proposal(code="bad", rationale="candidate"),
                Proposal(code="nan", rationale="candidate"),
                Proposal(code="good", rationale="candidate"),
            ]
        )

        async def evaluate(code):
            return {
                "base": Evaluation(10),
                "bad": Evaluation(None, "infeasible"),
                "nan": Evaluation(math.nan),
                "good": Evaluation(2),
            }[code]

        agent = OptimizingTheOptimizer("Any task", provider, evaluate, maximize=False)
        result = await agent.run("base", target="choose", rounds=4)
        self.assertEqual(result.code, "good")
        self.assertEqual(agent.attempts[0]["response"], "not JSON")
        self.assertTrue(all(a["error"] for a in agent.attempts[:3]))
        self.assertIn("infeasible", provider.calls[2])
        self.assertIn("finite", provider.calls[3])
        self.assertEqual(agent.evaluations, 4)

    async def test_errors_propagate_and_ties_retain_baseline(self):
        async def evaluate(code):
            if code == "crash":
                raise RuntimeError("evaluator unavailable")
            return Evaluation(1)

        agent = OptimizingTheOptimizer(
            "task",
            ScriptedProvider(
                [Proposal(code="tie", rationale="same"), Proposal(code="crash", rationale="oops")]
            ),
            evaluate,
        )
        result = await agent.run("base", target="f", rounds=1)
        self.assertEqual(result.code, "base")
        with self.assertRaisesRegex(RuntimeError, "evaluator unavailable"):
            await agent.run("base", target="f", rounds=1)
        self.assertIn("evaluator unavailable", agent.attempts[0]["error"])

    async def test_all_templates_render_and_reject_blank_code(self):
        agent = OptimizingTheOptimizer(
            "task", ScriptedProvider([Proposal(code="  ", rationale="blank")]), None
        )
        for operation in (agent.improve_heuristic, agent.revise, agent.optimize_code):
            rendered = await operation.render(agent, "base", "feedback", [])
            self.assertIn('"code"', rendered)
        for path in self.root.glob("*.j2"):
            tree = Environment().parse(path.read_text())
            self.assertFalse(list(tree.find_all((nodes.If, nodes.CondExpr))))
        with self.assertRaisesRegex(ValueError, "blank"):
            await agent.improve_heuristic("base", "", [], provider=agent.provider)

    async def test_provider_errors_are_not_candidate_rejections(self):
        async def evaluate(source):
            return Evaluation(1)

        agent = OptimizingTheOptimizer(
            "task", ScriptedProvider([ValueError("bad provider config")]), evaluate
        )
        with self.assertRaisesRegex(ValueError, "bad provider config"):
            await agent.run("base", target="f", rounds=3)
        self.assertEqual(len(agent.attempts), 1)
        self.assertEqual(agent.evaluations, 1)


class CMSATests(unittest.IsolatedAsyncioTestCase):
    def test_weights_match_official_code(self):
        p = selection_probabilities([0, 3], [-1, 2])
        self.assertEqual(p, [0.8, 0.2])
        entropy = -sum(value * math.log(value) for value in p)
        adjusted = selection_probabilities([0, 3], [-1, 2], entropy=True)
        self.assertAlmostEqual(adjusted[0], (0.8 + entropy) / (1 + 2 * entropy))
        self.assertAlmostEqual(sum(adjusted), 1)
        self.assertEqual(selection_probabilities([10], [-1], entropy=True), [1])
        self.assertEqual(selection_probabilities([], [], entropy=True), [])

    async def test_merge_age_reset_expiry_and_missing_solver_solution(self):
        pools = []
        answers = iter([frozenset({"a"}), None, frozenset({"a"})])

        async def solve(pool, seconds):
            pools.append(pool)
            self.assertEqual(seconds, 7)
            return next(answers)

        async def evaluate(solution):
            return len(solution)

        agent = CMSA(["a", "b"], {"a": 0, "b": 1}, lambda chosen, c: True, solve, evaluate)
        result = await agent.run(
            iterations=3, constructions=1, age_max=2, determinism_rate=1, solve_time_limit=7
        )
        self.assertEqual(result.components, frozenset({"a", "b"}))
        self.assertEqual(pools, [frozenset({"a", "b"})] * 3)
        self.assertEqual([row["ages"]["b"] for row in agent.history], [1, 1, -1])
        self.assertEqual(agent.age["a"], 0)

    async def test_generic_capacity_problem_and_seeded_variants(self):
        costs = {"red": 1, "blue": 2, "green": 3}

        def can_add(chosen, component):
            return sum(costs[c] for c in chosen) + costs[component] <= 3

        async def evaluate(solution):
            self.assertLessEqual(sum(costs[c] for c in solution), 3)
            return sum(costs[c] for c in solution)

        async def solve(pool, seconds):
            # Exact enumeration is deliberately confined to this tiny test problem.
            from itertools import combinations

            subsets = (frozenset(s) for n in range(len(pool) + 1) for s in combinations(pool, n))
            return max(
                (s for s in subsets if sum(costs[c] for c in s) <= 3),
                key=lambda s: sum(costs[c] for c in s),
            )

        for variant in ("baseline", "v1", "v2"):
            runs = []
            for _ in range(2):
                agent = CMSA(list(costs), costs, can_add, solve, evaluate, variant=variant)
                result = await agent.run(iterations=3, constructions=2, seed=42)
                self.assertEqual(result.score, 3)
                runs.append(agent.history)
            self.assertEqual(*runs)

    async def test_solver_boundary_minimization_and_zero_budget(self):
        async def evaluate(solution):
            return len(solution)

        async def solve(pool, seconds):
            return frozenset()

        agent = CMSA(["a"], {"a": 0}, lambda chosen, c: True, solve, evaluate, maximize=False)
        result = await agent.run(iterations=1, constructions=1)
        self.assertEqual(result.score, 0)
        self.assertEqual(result.components, frozenset())
        self.assertIsNone(await agent.run(iterations=1, time_limit=0))
        self.assertEqual(agent.history, [])

        async def invalid(pool, seconds):
            return frozenset({"outside"})

        agent.solve = invalid
        with self.assertRaisesRegex(ValueError, "outside"):
            await agent.run(iterations=1, constructions=1)

    def test_determinism_ignores_age_and_random_support_matches_variants(self):
        from random import Random

        agent = CMSA(["cheap", "costly"], {"cheap": 0, "costly": 1}, None, None, None)
        agent.rng, agent.age = Random(1), {"cheap": 99, "costly": -1}
        self.assertEqual(agent._select(["cheap", "costly"], 1, 1), "cheap")
        with patch.object(agent.rng, "choices", return_value=["costly"]) as choices:
            self.assertEqual(agent._select(["cheap", "costly"], 0, 1), "costly")
            self.assertEqual(choices.call_args.args[0], ["cheap", "costly"])
        agent.variant = "baseline"
        self.assertEqual(agent._select(["cheap", "costly"], 0, 1), "cheap")

    async def test_remaining_time_is_passed_to_solver(self):
        async def evaluate(solution):
            return len(solution)

        async def solve(pool, seconds):
            self.assertEqual(seconds, 2.0)
            return pool

        agent = CMSA(["x"], {"x": 0}, lambda chosen, c: True, solve, evaluate)
        with patch("optimizing_the_optimizer.cmsa.time.monotonic", side_effect=[0, 1, 3]):
            result = await agent.run(
                iterations=1, constructions=1, time_limit=5, solve_time_limit=10
            )
        self.assertEqual(result.score, 1)

    async def test_expired_components_reenter_and_nonfinite_scores_abort(self):
        async def evaluate(solution):
            return len(solution)

        async def solve(pool, seconds):
            return frozenset()

        agent = CMSA(["x"], {"x": 0}, lambda chosen, c: True, solve, evaluate)
        await agent.run(iterations=2, constructions=1, age_max=1)
        self.assertEqual([row["pool"] for row in agent.history], [frozenset({"x"})] * 2)
        self.assertEqual(agent.age, {"x": -1})

        async def invalid(solution):
            return math.inf

        agent.evaluate = invalid
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run(iterations=1, constructions=1)


if __name__ == "__main__":
    unittest.main()
