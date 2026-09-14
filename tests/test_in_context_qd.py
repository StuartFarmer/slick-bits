"""Offline checks of archive selection, prompt variants, and rejection accounting."""

import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from in_context_qd import CandidateRejected, Config, Evaluation, Grid, InContextQD, NumericSpace
from tests.providers import ScriptedProvider


class InContextQDTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "in_context_qd/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)

    def agent(self, responses=(), **settings):
        async def evaluate(text):
            fitness, feature = map(float, text.split(","))
            return Evaluation(fitness, (feature,))

        return InContextQD(
            "Return fitness, feature",
            ScriptedProvider(responses),
            evaluate,
            Grid(((0, 1),), (4,)),
            config=Config(**settings),
        )

    async def test_batch_snapshot_actual_cells_strict_elitism_and_metrics(self):
        agent = self.agent(["3,0.1", "3,0.1", "2,0.8", "1,0.8"], batch_size=2)
        result = await agent.run(["1,0.1"], generations=2)
        self.assertEqual(result.archive[(0,)].candidate, "3,0.1")
        self.assertEqual(result.archive[(3,)].candidate, "2,0.8")
        self.assertEqual((result.evaluations, result.model_calls), (5, 4))
        self.assertEqual((result.coverage, result.qd_score, result.max_fitness), (0.5, 5, 3))
        self.assertEqual([m.occupied for m in result.history], [1, 1, 2])
        self.assertEqual([a.accepted for a in result.attempts], [True, True, False, True, False])
        # Both queries read the old archive, even if the first child improves it.
        self.assertEqual([a.query.fitness for a in result.attempts[1:3]], [1.2, 1.2])
        self.assertEqual([a.query.fitness for a in result.attempts[3:]], [3.6, 3.6])

    async def test_empty_query_fallback_and_context_order(self):
        agent = self.agent(batch_size=2, context_size=10, seed=5)
        await agent.run(["-10,0.1", "-5,0.9"], generations=0)
        queries = agent.build_queries()
        self.assertEqual({q.cell for q in queries}, {(1,), (2,)})
        for query in queries:
            distances = [abs(p.features[0] - query.features[0]) for p in query.context]
            self.assertEqual(distances, sorted(distances, reverse=True))
            self.assertEqual(query.fitness, -4)
        # With fewer empty cells than the batch, sample from the entire grid.
        agent.archive[(1,)] = agent.archive[(0,)]
        with patch.object(agent.rng, "choices", return_value=[(0,), (0,)]) as choices:
            self.assertEqual([q.cell for q in agent.build_queries()], [(0,), (0,)])
        self.assertEqual(len(choices.call_args.args[0]), 4)

    async def test_zero_context_zero_fitness_and_all_prompt_variants(self):
        for template in ("qd", "fitness", "feature", "lmx"):
            for order in ("distance", "fitness", "random"):
                agent = self.agent(["1,0.1"], template=template, order=order, batch_size=1)
                result = await agent.run(["0,0.1"])
                self.assertEqual(result.model_calls, 1)
                self.assertGreater(result.attempts[-1].query.fitness, 0)
                prompt = agent.provider.calls[0]
                self.assertIn("0,0.1", prompt)
                self.assertEqual(prompt.count("Return fitness, feature"), 1)
                query = result.attempts[-1].query
                descriptor = agent.grid.encode(query.features)
                expected_rows = {
                    "qd": ["0.0 : 100 : 0,0.1", f"{query.fitness} : {descriptor} :"],
                    "fitness": ["0.0 : 0,0.1", f"{query.fitness} :"],
                    "feature": ["100 : 0,0.1", f"{descriptor} :"],
                    "lmx": ["0,0.1"],
                }
                self.assertEqual(prompt.strip().splitlines()[1:], expected_rows[template])
        agent = self.agent(["2,0.1"], context_size=0, batch_size=1, initial_fitness=7)
        result = await agent.run(["1,0.1"])
        self.assertEqual(result.attempts[-1].query.context, ())
        self.assertEqual(result.attempts[-1].query.fitness, 7)

    async def test_arbitrary_text_candidates_and_repeatable_run(self):
        measurements = {
            "quiet garden": Evaluation(2, (0.1,)),
            "busy market": Evaluation(4, (0.9,)),
            "shaded courtyard": Evaluation(5, (0.2,)),
        }

        async def evaluate(candidate):
            return measurements[candidate]

        agent = InContextQD(
            "Invent a place",
            ScriptedProvider(["shaded courtyard"]),
            evaluate,
            Grid(((0, 1),), (2,)),
            config=Config(batch_size=1, context_size=1, seed=3),
        )
        first = await agent.run(["quiet garden", "busy market"])
        self.assertEqual(
            {p.candidate for p in first.archive.values()}, {"shaded courtyard", "busy market"}
        )
        self.assertEqual(len(first.attempts[-1].query.context), 1)
        agent.provider = ScriptedProvider(["shaded courtyard"])
        self.assertEqual(first, await agent.run(["quiet garden", "busy market"]))

    async def test_rejections_are_bounded_and_raw_output_is_retained(self):
        agent = self.agent(["  ", "nan,0.1", "2,2", "2,0.1"], batch_size=4)
        result = await agent.run(["1,0.1"])
        self.assertEqual((result.model_calls, result.evaluations), (4, 4))
        self.assertEqual(result.attempts[1].raw, "  ")
        self.assertTrue(all(a.error for a in result.attempts[1:4]))
        self.assertEqual(result.max_fitness, 2)

        async def rejected(_):
            raise CandidateRejected("infeasible")

        agent.evaluate = rejected
        with self.assertRaisesRegex(RuntimeError, "empty"):
            await agent.run(["seed"])
        self.assertEqual(agent.evaluations, 1)

    async def test_errors_propagate_with_records_and_no_hidden_retry(self):
        agent = self.agent([TimeoutError("offline")], batch_size=1)
        with self.assertRaisesRegex(TimeoutError, "offline"):
            await agent.run(["1,0.1"])
        self.assertIn("offline", agent.attempts[-1].error)
        self.assertEqual(agent.model_calls, 1)

        async def broken(_):
            raise LookupError("broken evaluator")

        agent.evaluate = broken
        with self.assertRaisesRegex(LookupError, "broken evaluator"):
            await agent.run(["seed"])
        self.assertIn("broken evaluator", agent.attempts[-1].error)

    async def test_templates_render_from_any_directory_without_jinja_control_flow(self):
        agent = self.agent(batch_size=1)
        await agent.run(["1,0.1"], generations=0)
        query = agent.build_queries()[0]
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for name in ("qd", "fitness", "feature", "lmx"):
                    rendered = await getattr(InContextQD, "generate_" + name).render(agent, query)
                    self.assertIn("1,0.1", rendered)
            finally:
                os.chdir(previous)
        for path in self.root.glob("*.j2"):
            tree = Environment().parse(path.read_text())
            self.assertEqual(list(tree.find_all((nodes.If, nodes.CondExpr))), [])

    def test_grid_edges_and_numeric_representation(self):
        grid = Grid(((-1, 1), (0, 10)), (2, 5))
        self.assertEqual(grid.locate((-1, 10)), (0, 4))
        self.assertEqual(grid.centroid((0, 4)), (-0.5, 9))
        self.assertEqual(grid.encode((-0.5, 9)), "250, 900")
        for features in ((float("nan"), 1), (0,), (2, 3)):
            with self.assertRaises(CandidateRejected):
                grid.locate(features)
        space = NumericSpace(((-2, 2), (10, 20), (3, 3)))
        self.assertEqual(space.encode((-1, 17.5, 3)), "250, 750, 0")
        self.assertEqual(space.decode("250, 750, 0"), (-1, 17.5, 3))
        self.assertEqual(space.sample(random.Random(2)), space.sample(random.Random(2)))
        for text in ("1.2, 3, 0", "-1, 3, 0", "1001, 3, 0", "nan, 3, 0", "3, 0"):
            with self.assertRaises(CandidateRejected):
                space.decode(text)


if __name__ == "__main__":
    unittest.main()
