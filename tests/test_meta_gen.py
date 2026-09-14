"""Exercise the paper loops against the official MetaGen representation."""

import math
import random
import unittest
from copy import deepcopy
from unittest.mock import patch

from meta_gen import BaseConnector, Domain, RandomSearch, SimulatedAnnealing, Solution


class MetaGenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.random_state = random.getstate()
        random.seed(7)
        self.addCleanup(random.setstate, self.random_state)
        self.domain = Domain()
        self.domain.define_real("x", -5.0, 5.0)

    async def test_random_search_counts_and_preserves_best_snapshot(self):
        scores = iter([3, 2, 1, 8, 9, 10])
        seen = []

        async def evaluate(solution):
            seen.append(deepcopy(solution))
            return next(scores)

        agent = RandomSearch("Any objective", self.domain, evaluate)
        best = await agent.run(population_size=2, iterations=2)
        self.assertEqual(agent.evaluations, 6)
        self.assertEqual(agent.history, [2, 1, 1])
        self.assertEqual(best.fitness, 1)
        self.assertEqual(best["x"], seen[2]["x"])
        self.assertEqual([s.fitness for s in agent.population], [9, 10])
        best["x"] = 5.0
        self.assertEqual(agent.best["x"], seen[2]["x"])

    async def test_annealing_accepts_worse_but_returns_best(self):
        scores = iter([0, 1, -2, 100])

        async def evaluate(solution):
            return next(scores)

        agent = SimulatedAnnealing("Cost", self.domain, evaluate)
        # At T=1, accept delta=1 with draw=.1; reject delta=102 with draw=.9.
        with patch("meta_gen.agent.random.random", side_effect=[0.1, 0.9]):
            best = await agent.run(iterations=3, initial_temp=1.0, cooling_rate=1.0)
        self.assertEqual(agent.evaluations, 4)
        self.assertEqual(agent.accepted, [True, True, False])
        self.assertEqual(agent.history, [0, 0, -2, -2])
        self.assertEqual(best.fitness, -2)
        self.assertEqual(agent.current.fitness, -2)

        scores = iter([0, 1])
        with patch("meta_gen.agent.random.random", return_value=0.1):
            best = await agent.run(iterations=1, initial_temp=1.0)
        self.assertEqual(best.fitness, 0)
        self.assertEqual(agent.current.fitness, 1)
        self.assertEqual(agent.evaluations, 2)

    async def test_zero_iterations_and_temperature_underflow(self):
        async def evaluate(solution):
            return solution["x"] ** 2

        for agent in (
            RandomSearch("Cost", self.domain, evaluate),
            SimulatedAnnealing("Cost", self.domain, evaluate),
        ):
            best = await agent.run(iterations=0)
            self.assertTrue(math.isfinite(best.fitness))
            self.assertEqual(len(agent.history), 1)

        scores = iter([0, 1, 1, -1])

        async def extreme_score(solution):
            return next(scores)

        agent = SimulatedAnnealing("Cost", self.domain, extreme_score)
        best = await agent.run(iterations=3, initial_temp=1e-300, cooling_rate=1e-300)
        self.assertEqual(agent.accepted, [False, False, True])
        self.assertEqual(best.fitness, -1)

    async def test_evaluator_isolation_and_failures(self):
        seen = []

        async def destructive_evaluator(solution):
            seen.append(solution["x"])
            solution["x"] = 5.0
            return 0.0

        agent = RandomSearch("Cost", self.domain, destructive_evaluator)
        best = await agent.run(population_size=1, iterations=0)
        self.assertEqual(best["x"], seen[0])

        for optimizer in (RandomSearch, SimulatedAnnealing):
            for score in (float("nan"), float("inf"), -float("inf")):

                async def invalid(solution):
                    return score

                agent = optimizer("Cost", self.domain, invalid)
                with self.assertRaisesRegex(ValueError, "finite"):
                    await agent.run(iterations=0)
                self.assertEqual(agent.evaluations, 1)

            async def failed(solution):
                raise RuntimeError("evaluator failed")

            agent = optimizer("Cost", self.domain, failed)
            with self.assertRaisesRegex(RuntimeError, "evaluator failed"):
                await agent.run(iterations=0)
            self.assertEqual(agent.evaluations, 1)

    async def test_official_domain_all_six_types_and_custom_connector(self):
        from metagen.framework.domain import BaseDefinition

        class CustomSolution(Solution):
            pass

        connector = BaseConnector()
        connector.register(BaseDefinition, CustomSolution, dict)
        domain = Domain(connector)
        domain.define_integer("count", 1, 10)
        domain.define_real("rate", 0.0, 1.0)
        domain.define_categorical("mode", ["a", "b"])
        domain.define_group("settings")
        domain.define_integer_in_group("settings", "width", 1, 8)
        domain.define_dynamic_structure("stages", 1, 4)
        domain.define_group("stage")
        domain.define_real_in_group("stage", "weight", 0.0, 1.0)
        domain.define_categorical_in_group("stage", "kind", ["x", "y"])
        domain.set_structure_to_variable("stages", "stage")
        domain.define_static_structure("offsets", 3)
        domain.set_structure_to_integer("offsets", -2, 2)
        lengths = set()

        async def evaluate(solution):
            self.assertIsInstance(solution, CustomSolution)
            self.assertTrue(1 <= solution["count"] <= 10)
            self.assertTrue(0 <= solution["rate"] <= 1)
            self.assertIn(solution["mode"], ["a", "b"])
            self.assertTrue(1 <= solution["settings"]["width"] <= 8)
            self.assertTrue(1 <= len(solution["stages"]) <= 4)
            lengths.add(len(solution["stages"]))
            for stage in solution["stages"]:
                self.assertTrue(0 <= stage["weight"] <= 1)
                self.assertIn(stage["kind"], ["x", "y"])
            self.assertEqual(len(solution["offsets"]), 3)
            for offset in solution["offsets"]:
                self.assertTrue(-2 <= offset.get() <= 2)
            return solution["rate"] + len(solution["stages"])

        for optimizer in (RandomSearch, SimulatedAnnealing):
            best = await optimizer("Structured cost", domain, evaluate).run(iterations=20)
            self.assertTrue(math.isfinite(best.fitness))
        self.assertGreater(len(lengths), 1)


if __name__ == "__main__":
    unittest.main()
