"""Check Plum's distinct metaheuristic update rules."""

import unittest
from pathlib import Path
from unittest.mock import patch

from plum import Plum
from tests.providers import ScriptedProvider
from tests.test_grips import spans


class PlumTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "plum/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_ga_archive_appends_best_losing_neighbor(self):
        values = {"seed": 10, "weak": 2, "strong": 4, "winner": 20, "low": 1}

        async def evaluate(text):
            return values[text]

        provider = ScriptedProvider(["weak", "strong", "winner", "low"])
        result = await Plum("Teach music", provider, evaluate, spans).run(
            "seed", iterations=2, candidates=2, operations=("substitute",), tournament=2
        )
        self.assertEqual([p["prompt"] for p in result["population"]], ["seed", "strong", "winner"])
        self.assertEqual(result["best"]["score"], 20)
        self.assertEqual(result["evaluations"], 5)
        self.assertFalse(result["history"][0]["accepted"])
        self.assertIn("Teach music", provider.calls[0])

    async def test_hill_climb_ties_annealing_and_tabu_rejections(self):
        async def score(text):
            return 1 if text == "seed" else 0

        for algorithm in ("hc", "sa"):
            result = await Plum("Task", ScriptedProvider(["worse"]), score, spans).run(
                "seed",
                algorithm=algorithm,
                iterations=1,
                candidates=1,
                operations=("substitute",),
                temperature=1e9,
            )
            self.assertEqual(result["current"]["prompt"], "worse" if algorithm == "sa" else "seed")
            self.assertEqual(result["best"]["prompt"], "seed")

        result = await Plum("Task", ScriptedProvider(["worse", "worse"]), score, spans).run(
            "seed",
            algorithm="tabu",
            iterations=2,
            candidates=1,
            operations=("substitute",),
            tabu_accept=0,
        )
        self.assertEqual(result["tabu"], ["worse"])
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(result["rejections"][0]["reason"], "tabu")

    async def test_harmony_pitch_adjustment_and_memory_selection(self):
        async def score(text):
            return len(text)

        provider = ScriptedProvider(["longer", "longest"])
        result = await Plum("Task", provider, score, spans).run(
            "a b",
            algorithm="hs",
            iterations=1,
            candidates=1,
            segments=2,
            memory_rate=1,
            pitch_rate=1,
            harmony_size=1,
        )
        self.assertEqual(result["best"]["prompt"], "longer longest")
        self.assertEqual(len(result["population"]), 1)
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(result["optimizer_calls"], 2)

    async def test_empty_candidates_are_bounded_and_transport_propagates(self):
        async def score(text):
            return 1

        agent = Plum("Task", ScriptedProvider([]), score, spans)
        result = await agent.run("a", operations=("delete",), candidates=3, patience=1)
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(len(result["rejections"]), 3)
        agent = Plum("Task", ScriptedProvider([RuntimeError("offline")]), score, spans)
        with self.assertRaisesRegex(RuntimeError, "offline"):
            await agent.run("a", operations=("substitute",), candidates=1)


if __name__ == "__main__":
    unittest.main()
