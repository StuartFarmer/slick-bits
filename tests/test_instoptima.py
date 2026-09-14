"""Check component-preserving variation and Pareto selection, including shared numerics."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from instoptima import Individual, InstOptima, Instruction
from prompt_optimization.pareto import checked_scores, crowding, fronts, nsga_select
from tests.providers import ScriptedProvider


class InstOptimaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "instoptima/prompts"
        )
        root.start()
        self.addCleanup(root.stop)

    def test_pareto_fronts_extremes_ties_and_finite_scores(self):
        scores = [(1, 0), (0, 1), (0.6, 0.6), (0, 0)]
        self.assertEqual(fronts(scores), [[0, 1, 2], [3]])
        self.assertEqual(set(nsga_select(scores, 2)), {0, 1})
        self.assertEqual(nsga_select([(1, 1)] * 3, 2), [0, 1])
        self.assertEqual(crowding([(1, 1)] * 3, [0, 1, 2]), {0: 0, 1: 0, 2: 0})
        self.assertTrue(math.isfinite(crowding([(-1e308, 1), (0, 1), (1e308, 1)], [0, 1, 2])[1]))
        for values in ((math.inf, 1), (math.nan, 1), (1,)):
            with self.assertRaises(ValueError):
                checked_scores(values, 2)

    async def test_all_component_operators_and_initialization(self):
        async def evaluate(instruction):
            return (len(instruction.definition), -len(instruction.examples))

        agent = InstOptima("Task", ScriptedProvider([]), evaluate, ["quality", "brevity"])
        first = Individual(Instruction(definition="old", examples=["a"]), (1, 2))
        second = Individual(Instruction(definition="other", examples=["b"]), (2, 1))
        for method, args in (
            (agent.mutate_definition, (first,)),
            (agent.cross_definition, (first, second)),
        ):
            result = await method(*args, provider=ScriptedProvider(['{"text":"new"}']))
            self.assertEqual(result.examples, ["a"])
            self.assertEqual(result.definition, "new")
        for method, args in (
            (agent.mutate_examples, (first,)),
            (agent.cross_examples, (first, second)),
        ):
            result = await method(*args, provider=ScriptedProvider(['{"examples":["b"]}']))
            self.assertEqual(result.definition, "old")
        with self.assertRaisesRegex(ValueError, "parents"):
            await agent.cross_examples(
                first, second, provider=ScriptedProvider(['{"examples":["invented"]}'])
            )
        provider = ScriptedProvider([Instruction(definition="new", examples=[])])
        agent.provider = provider
        result = await agent.run(population_size=1, generations=0)
        self.assertEqual(result["evaluations"], 1)
        self.assertIn("quality", provider.calls[0])

    async def test_generation_and_refresh(self):
        async def evaluate(instruction):
            return (len(instruction.definition), -len(instruction.examples))

        # Seed 1 chooses a unary definition operation, then refreshes the Pareto front.
        provider = ScriptedProvider(
            ['{"text":"better"}', Instruction(definition="fresh", examples=[])]
        )
        agent = InstOptima("Task", provider, evaluate, ["quality", "brevity"])
        result = await agent.run(
            [Instruction(definition="old", examples=[])],
            population_size=1,
            generations=1,
            refresh_probability=1,
            seed=1,
        )
        self.assertEqual(result["population"][0].instruction.definition, "fresh")
        self.assertEqual(result["evaluations"], 3)
