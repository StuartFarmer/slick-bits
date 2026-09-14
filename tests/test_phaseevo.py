"""Check PhaseEvo phase order, joint candidates, and performance-vector selection."""

import json
import math
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from phaseevo import Candidate, Evaluation, Individual, PhaseEvo
from tests.providers import ScriptedProvider


def candidate(text):
    return Candidate(instruction=text, examples=["input -> output"])


class PhaseEvoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "phaseevo/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_phases_and_elitism(self):
        async def evaluate(value):
            return Evaluation(len(value.instruction), (True, False), "second example failed")

        responses = [
            candidate("aa"),
            candidate("bbb"),
            '{"text":"fix"}',
            candidate("long"),
            '{"text":"fix"}',
            candidate("longer"),
            candidate("x"),
            candidate("x"),
            candidate("longest"),
            candidate("x"),
        ]
        provider = ScriptedProvider(responses)
        result = await PhaseEvo("Task", provider, evaluate).run(
            ["input -> output"], population_size=2, max_rounds=1
        )
        self.assertEqual(result["history"], [3, 6, 6, 7])
        self.assertEqual(result["evaluations"], 8)
        self.assertIn("observed errors", provider.calls[2])
        self.assertIn("fix", provider.calls[3])
        self.assertIn("Paraphrase", provider.calls[-1])

    async def test_hamming_complementarity_and_all_templates(self):
        async def evaluate(value):
            return Evaluation(math.inf, (), "")

        agent = PhaseEvo("Task", ScriptedProvider([]), evaluate)
        agent.rng = random.Random(0)
        population = [
            Individual(candidate(str(i)), Evaluation(0, v, ""))
            for i, v in enumerate([(True, False), (False, True), (True, True)])
        ]
        parents = agent._distinct_parents(population, 2)
        self.assertEqual(
            sum(
                a != b
                for a, b in zip(parents[0].evaluation.outcomes, parents[1].evaluation.outcomes)
            ),
            2,
        )
        for method, args in (
            (PhaseEvo.crossover, (candidate("a"), candidate("b"))),
            (PhaseEvo.distribution, ([candidate("a")],)),
        ):
            rendered = await method.render(agent, *args)
            self.assertIn("Task", rendered)
            self.assertIn("instruction", rendered)
        with self.assertRaises(ValueError):
            await agent.run([], initial=[candidate("a")], population_size=1, max_rounds=0)
        with self.assertRaises(ValueError):
            await agent.semantic(
                candidate("a"),
                provider=ScriptedProvider([json.dumps({"instruction": " ", "examples": []})]),
            )
