"""Check BoT search and experience through Slick with no model requests."""

import asyncio
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from bot import BoostingOfThoughts, Scores, Thought
from tests.providers import ScriptedProvider


class BoTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "bot/prompts"
        patcher = patch("slick.prompts.TEMPLATE_ROOT", root)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_experience_accumulates_and_final_answer_uses_last_chain(self):
        async def evaluate(chain):
            return Scores(node=0.9, edge=0.9)

        provider = ScriptedProvider(
            [
                "bad approach",
                "other approach",
                "Error: wrong constraint. Advice: revise it.",
                "revised approach",
                "alternative",
                "Keep the revised approach.",
                "Final artifact",
            ]
        )
        agent = BoostingOfThoughts("Design a workflow", provider, evaluate)
        result = await agent.run(iterations=2, trees=1, aggregation="best_first")
        self.assertEqual(result.answer, "Final artifact")
        self.assertEqual(len(result.experiences), 2)
        self.assertEqual(result.chain[0].text, "revised approach")
        self.assertNotIn("wrong constraint", provider.calls[0])
        self.assertIn("wrong constraint", provider.calls[3])
        self.assertIn("bad approach", provider.calls[3])
        self.assertIn("Keep the revised approach", provider.calls[-1])
        self.assertEqual(result.calls, 7)

    async def test_greedy_can_join_different_chains_without_repeating_a_step(self):
        agent = BoostingOfThoughts("Task", ScriptedProvider(['{"value": 0.8}']))
        chains = [
            (Thought("A", 0.8, 0.8), Thought("weak", 0.2, 0.2)),
            (Thought("equivalent A", 0.7, 0.7), Thought("B", 0.9, 0.9)),
        ]
        joined = await agent.aggregate(chains, "greedy", 2, 0.7)
        self.assertEqual([t.text for t in joined], ["A", "B"])
        best = await agent.aggregate(chains, "best_first", 2, 0.7)
        self.assertEqual(best, chains[1])

    async def test_growth_orders_and_expansion_budget(self):
        async def evaluate(chain):
            value = 0.7 if chain[-1] == "B" else 0.5
            return Scores(node=value, edge=value)

        for strategy, expected in (("level", "A"), ("leaf", "B")):
            provider = ScriptedProvider(["A", "B", "child 1", "child 2"])
            agent = BoostingOfThoughts("Task", provider, evaluate)
            tree = await agent.build_tree(provider, strategy, 3, 2, (0.3, 0.8))
            self.assertEqual(tree.expansions, 2)
            self.assertEqual(len(tree.leaves), 3)
            self.assertEqual(tree.best[0].text, expected)
            self.assertIn(expected, provider.calls[2])

    async def test_scores_outside_range_stop_but_boundary_scores_expand(self):
        async def evaluate(chain):
            return Scores(node=0.3, edge=0.8 if chain[-1] == "expand" else 0.9)

        provider = ScriptedProvider(["stop", "expand", "leaf 1", "leaf 2"])
        agent = BoostingOfThoughts("Task", provider, evaluate)
        tree = await agent.build_tree(provider, "level", 2, 9, (0.3, 0.8))
        self.assertEqual(tree.expansions, 2)
        self.assertEqual(sorted(map(len, tree.leaves)), [1, 2, 2])

    async def test_selection_counts_node_and_edge_scores(self):
        async def evaluate(chain):
            if chain[-1] == "strong edge":
                return Scores(node=0.2, edge=0.9)
            return Scores(node=0.8, edge=0.7)

        provider = ScriptedProvider(["strong edge", "strong sum", "Feedback", "Answer"])
        agent = BoostingOfThoughts("Task", provider, evaluate)
        result = await agent.run(iterations=1, trees=1, max_depth=1, aggregation="best_first")
        self.assertEqual(result.chain[0].text, "strong sum")
        self.assertEqual(agent.forests[0][0].best, result.chain)

    async def test_generated_scores_and_raw_failure_records(self):
        provider = ScriptedProvider(
            [
                "first",
                '{"value": 0.5}',
                '{"value": 0.9}',
                "second",
                '{"value": 0.4}',
                '{"value": 0.9}',
                "Advice",
                "Answer",
            ]
        )
        agent = BoostingOfThoughts("Task", provider)
        result = await agent.run(iterations=1, trees=1, aggregation="best_first")
        self.assertEqual(result.chain[0], Thought("first", 0.5, 0.9))
        for bad in ("not json", '{"value": 2}', '{"value": NaN}'):
            provider = ScriptedProvider(["step", bad])
            agent = BoostingOfThoughts("Task", provider)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await agent.run(iterations=1, trees=1)
            self.assertEqual(agent.calls[-1]["response"], bad)
            self.assertIn("error", agent.calls[-1])
            self.assertEqual(len(agent.calls), 2)

    async def test_parallel_trees_use_seeded_sampling_and_independent_providers(self):
        async def evaluate(chain):
            return Scores(node=0.9, edge=0.9)

        settings, providers = [], []
        started = 0
        both_started = asyncio.Event()

        def make_provider(temperature, top_p):
            settings.append((temperature, top_p))
            source = ScriptedProvider([f"tree {len(settings)}", "alternative"])
            original = source.acall

            async def concurrent(context):
                nonlocal started
                started += 1
                if started == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), timeout=1)
                return await original(context)

            source.acall = concurrent
            providers.append(source)
            return source

        provider = ScriptedProvider(["Feedback", "Final"])
        agent = BoostingOfThoughts("Task", provider, evaluate, tree_provider=make_provider)
        result = await agent.run(iterations=1, trees=2, aggregation="best_first", seed=12)
        rng = random.Random(12)
        expected = [
            (rng.choice((0.2, 0.4, 0.6, 0.7, 0.9, 1.1, 1.5)), rng.choice((0.1, 0.3, 0.5, 0.7, 0.9)))
            for _ in range(2)
        ]
        self.assertEqual(settings, expected)
        self.assertEqual([t.strategy for t in agent.forests[0]], ["level", "leaf"])
        self.assertEqual([(t.temperature, t.top_p) for t in agent.forests[0]], expected)
        self.assertEqual(result.chain[0].text, "tree 1")
        self.assertEqual(result.calls, 6)
        self.assertEqual([len(p.calls) for p in providers], [2, 2])

    async def test_failure_cancels_other_trees_before_returning(self):
        pending = asyncio.Event()
        cancelled = asyncio.Event()
        first, second = ScriptedProvider([]), ScriptedProvider([])

        async def fail(context):
            await pending.wait()
            raise OSError("transport")

        async def block(context):
            pending.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        first.acall, second.acall = fail, block
        sources = iter((first, second))
        agent = BoostingOfThoughts(
            "Task", ScriptedProvider([]), tree_provider=lambda temperature, top_p: next(sources)
        )
        with self.assertRaisesRegex(OSError, "transport"):
            await asyncio.wait_for(agent.run(iterations=1, trees=2), timeout=1)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(agent.experiences, [])

    async def test_greedy_threshold_and_cycles(self):
        a, b = Thought("A", 0.8, 0.8), Thought("B", 0.7, 0.7)
        agent = BoostingOfThoughts("Task", ScriptedProvider([]))
        chain = await agent.aggregate([(a, b, a)], "greedy", 10, 0.7)
        self.assertEqual(chain, (a, b))
        agent = BoostingOfThoughts("Task", ScriptedProvider(['{"value": 0.7}']))
        chain = await agent.aggregate([(a,), (Thought("other", 0.1, 0.1), b)], "greedy", 3, 0.7)
        self.assertEqual(chain, (a,))

    async def test_external_invalid_scores_propagate_and_are_recorded(self):
        async def evaluate(chain):
            return Scores.model_construct(node=float("nan"), edge=0.5)

        agent = BoostingOfThoughts("Task", ScriptedProvider(["step"]), evaluate)
        with self.assertRaises(ValueError):
            await agent.run(iterations=1, trees=1)
        self.assertIn("error", agent.assessments[0])
        self.assertEqual(len(agent.calls), 1)

    async def test_empty_generated_text_and_zero_iterations(self):
        agent = BoostingOfThoughts("Task", ScriptedProvider([" ", "Direct answer"]))
        with self.assertRaisesRegex(ValueError, "blank"):
            await agent.run(iterations=1, trees=1)
        self.assertEqual(agent.calls[0]["response"], " ")
        result = await agent.run(iterations=0)
        self.assertEqual(result.answer, "Direct answer")
        self.assertEqual(result.chain, ())
        self.assertEqual(result.experiences, ())
        self.assertEqual(result.calls, 1)

    async def test_all_templates_render_from_another_directory(self):
        root = Path(__file__).resolve().parents[1] / "bot/prompts"
        agent = BoostingOfThoughts("Arbitrary task", ScriptedProvider([]))
        previous = Path.cwd()
        try:
            os.chdir(root)
            for operation, args in (
                (BoostingOfThoughts.generate, ((),)),
                (BoostingOfThoughts.score_thought, ((), "step")),
                (BoostingOfThoughts.score_edge, ((), "step")),
                (BoostingOfThoughts.similarity, ("one", "two")),
                (BoostingOfThoughts.analyze, ((),)),
                (BoostingOfThoughts.finish, ((),)),
            ):
                rendered = await operation.render(agent, *args)
                self.assertIn("Arbitrary task", rendered)
                if operation in (
                    BoostingOfThoughts.score_thought,
                    BoostingOfThoughts.score_edge,
                    BoostingOfThoughts.similarity,
                ):
                    self.assertIn('"properties"', rendered)
            for path in root.glob("*.j2"):
                parsed = Environment().parse(path.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
