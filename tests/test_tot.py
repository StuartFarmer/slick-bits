"""Exercise ToT search decisions through the real Slick prompt boundary."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from tests.providers import ScriptedProvider
from tot import State, TreeOfThoughts


class ToTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "tot/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_bfs_selects_globally_and_finalizes_only_best_leaf(self):
        provider = ScriptedProvider(
            [
                '{"thoughts": ["A", "B", "C"]}',
                '{"score": 0.9}',
                '{"score": 0.8}',
                '{"score": 0.1}',
                '{"thoughts": ["A1", "A2"]}',
                '{"thoughts": ["B1", "B2"]}',
                '{"score": 0.2}',
                '{"score": 0.3}',
                '{"score": 0.7}',
                '{"score": 0.6}',
                "Final artifact",
            ]
        )
        agent = TreeOfThoughts("Arbitrary task", provider)
        result = await agent.run(depth=2, breadth=2, candidates=3, evaluation_samples=1)
        self.assertEqual(result.solutions[0].state.thoughts, ("B", "B1"))
        self.assertEqual(result.solutions[0].answer, "Final artifact")
        self.assertEqual(result.expansions, 3)
        self.assertEqual([s.thoughts for s in agent.history[0]], [("A",), ("B",)])
        self.assertEqual(len(agent.calls), len(provider.calls))
        self.assertNotIn('"C"', provider.calls[-1])

    async def test_dfs_prunes_backtracks_and_records_all_terminal_outputs(self):
        provider = ScriptedProvider(
            [
                '{"thoughts": ["B", "A", "prune"]}',
                '{"score": 0.8}',
                '{"score": 0.9}',
                '{"score": 0.5}',
                '{"thoughts": ["dead"]}',
                '{"score": 0.1}',
                '{"thoughts": ["B1", "B2"]}',
                '{"score": 0.7}',
                '{"score": 0.6}',
                "one",
                "two",
            ]
        )
        agent = TreeOfThoughts("Task", provider)
        result = await agent.run(
            search="dfs", depth=2, candidates=3, threshold=0.5, evaluation_samples=1
        )
        self.assertEqual([s.state.thoughts for s in result.solutions], [("B", "B1"), ("B", "B2")])
        self.assertEqual(result.expansions, 3)
        self.assertIn('"A"', provider.calls[4])
        self.assertFalse(result.budget_exhausted)

    async def test_independent_samples_and_repeated_votes(self):
        provider = ScriptedProvider(
            ["A", "A", "B", '{"choice": 2}', '{"choice": 1}', '{"choice": 2}', "answer"]
        )
        agent = TreeOfThoughts("Task", provider, thought="One short plan")
        result = await agent.run(
            depth=1,
            generation="sample",
            evaluation="vote",
            candidates=3,
            evaluation_samples=3,
        )
        self.assertEqual(result.solutions[0].state.thoughts, ("B",))
        self.assertAlmostEqual(result.solutions[0].state.value, 2 / 3)
        self.assertEqual(provider.calls[0], provider.calls[1])
        self.assertEqual(provider.calls[1], provider.calls[2])

    async def test_value_averages_and_stable_ties(self):
        provider = ScriptedProvider(
            [
                '{"thoughts": ["A", "B"]}',
                '{"score": 0.2}',
                '{"score": 0.8}',
                '{"score": 0.5}',
                '{"score": 0.5}',
                "answer",
            ]
        )
        result = await TreeOfThoughts("Task", provider).run(depth=1, evaluation_samples=2)
        self.assertEqual(result.solutions[0].state, State(("A",), 0.5))

    async def test_custom_evaluator_and_expansion_budget(self):
        assessed = []

        async def evaluate(state):
            assessed.append(state.thoughts)
            return 0.8

        for search in ("bfs", "dfs"):
            agent = TreeOfThoughts("Task", ScriptedProvider(['{"thoughts": ["A"]}']), evaluate)
            result = await agent.run(search=search, depth=2, max_expansions=1)
            self.assertEqual(result.solutions, ())
            self.assertEqual(result.best_state.thoughts, ("A",))
            self.assertTrue(result.budget_exhausted)
            self.assertEqual(len(agent.calls), 1)
        self.assertEqual(assessed, [("A",), ("A",)])

    async def test_dead_end_and_zero_depth(self):
        agent = TreeOfThoughts("Task", ScriptedProvider(['{"thoughts": []}', "direct"]))
        result = await agent.run(depth=2)
        self.assertEqual(result.solutions, ())
        self.assertFalse(result.budget_exhausted)
        result = await agent.run(depth=0)
        self.assertEqual(result.solutions[0].state.thoughts, ())
        self.assertEqual(len(agent.calls), 1)

    async def test_budget_preserves_whole_bfs_levels_and_all_generated_dfs_leaves(self):
        responses = ['{"thoughts": ["A", "B"]}', '{"score": 0.8}', '{"score": 0.7}']
        agent = TreeOfThoughts("Task", ScriptedProvider(responses))
        result = await agent.run(depth=2, breadth=2, max_expansions=2, evaluation_samples=1)
        self.assertEqual(result.expansions, 1)
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.solutions, ())
        self.assertEqual(len(agent.calls), 3)
        agent = TreeOfThoughts("Task", ScriptedProvider(responses + ["answer A", "answer B"]))
        result = await agent.run(search="dfs", depth=1, max_expansions=1, evaluation_samples=1)
        self.assertEqual(len(result.solutions), 2)
        self.assertFalse(result.budget_exhausted)

    async def test_generated_rejection_retains_raw_response_without_retry(self):
        for bad in ("not json", '{"thoughts": [" "]}', '{"thoughts": ["A", "B", "C", "D"]}'):
            agent = TreeOfThoughts("Task", ScriptedProvider([bad]))
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await agent.run(candidates=3)
            self.assertEqual(agent.calls[0]["response"], bad)
            self.assertIn("error", agent.calls[0])
            self.assertEqual(len(agent.calls), 1)
        for bad in ('{"choice": 2}', '{"choice": true}', '{"choice": "1"}'):
            agent = TreeOfThoughts("Task", ScriptedProvider(['{"thoughts": ["A"]}', bad]))
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await agent.run(depth=1, evaluation="vote", evaluation_samples=1)
            self.assertEqual(agent.calls[-1]["response"], bad)
        for bad in ('{"score": NaN}', '{"score": 1.1}', '{"score": -0.1}'):
            agent = TreeOfThoughts("Task", ScriptedProvider(['{"thoughts": ["A"]}', bad]))
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await agent.run(depth=1, evaluation_samples=1)
            self.assertEqual(agent.calls[-1]["response"], bad)
        for options in ({"depth": 0}, {"generation": "sample"}):
            agent = TreeOfThoughts("Task", ScriptedProvider([" "]))
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "blank"):
                await agent.run(**options)
            self.assertEqual(agent.calls[-1]["response"], " ")

    async def test_failures_propagate_and_nonfinite_evaluations_fail(self):
        agent = TreeOfThoughts("Task", ScriptedProvider([OSError("transport")]))
        with self.assertRaisesRegex(OSError, "transport"):
            await agent.run()
        self.assertIn("error", agent.calls[0])

        async def evaluate(state):
            return float("nan")

        agent = TreeOfThoughts("Task", ScriptedProvider(['{"thoughts": ["A"]}']), evaluate)
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run()
        self.assertIn("error", agent.assessments[0])

    async def test_all_templates_render_from_another_directory(self):
        agent = TreeOfThoughts("Task", ScriptedProvider([]))
        previous = Path.cwd()
        try:
            os.chdir(self.templates)
            for operation, args in (
                (TreeOfThoughts.sample, (State(),)),
                (TreeOfThoughts.propose, (State(), 3)),
                (TreeOfThoughts.value, (State(("A",)),)),
                (TreeOfThoughts.vote, ([State(("A",)), State(("B",))],)),
                (TreeOfThoughts.finish, (State(("A",)),)),
            ):
                rendered = await operation.render(agent, *args)
                self.assertIn("Task", rendered)
            for template in self.templates.glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
