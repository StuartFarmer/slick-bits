"""Offline checks for feedback-gated, steady-state text evolution."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from interactive_evolution import Evaluation, InteractiveEvolution
from tests.providers import ScriptedProvider

PROMPTS = Path(__file__).resolve().parents[1] / "interactive_evolution/prompts"


class InteractiveEvolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_feedback_gates_crossover_mutation_and_protects_unrated_child(self):
        provider = ScriptedProvider(["seed B", "combined", "child"])
        snapshots = []
        batches = iter(
            [
                [Evaluation(0, 1)] * 3,
                [Evaluation(1, -1)],
                [Evaluation(2, -1)],
            ]
        )

        async def evaluate(population):
            snapshots.append(population)
            # Three votes alone must not permit selection of an unrated seed.
            self.assertEqual(len(provider.calls), 1 if len(snapshots) < 3 else 3)
            return next(batches)

        with patch("slick.prompts.TEMPLATE_ROOT", PROMPTS):
            agent = InteractiveEvolution("brief", provider, evaluate, mutation_topics=["resources"])
            result = await agent.run(
                initial=["seed A"],
                population_size=2,
                iterations=1,
                new_evaluations=3,
                crossover_probability=1,
                seed=4,
            )
        self.assertEqual([p.content for p in result], ["seed A", "child"])
        self.assertEqual(agent.completed_iterations, 1)
        self.assertEqual(len(agent.archive), 3)
        self.assertEqual(len(agent.evaluations), 5)
        self.assertEqual(agent.archive[1].fitness, -1)
        self.assertEqual(agent.archive[2].parents, (0, 0))
        self.assertEqual(agent.archive[2].mutation_topic, "resources")
        self.assertIn("combined", provider.calls[-1])
        self.assertNotIn("combined", [p.content for p in agent.archive.values()])
        self.assertEqual(snapshots[0][0].ratings, ())
        self.assertEqual(len(agent.history), 2)

    async def test_new_votes_and_child_coverage_required_each_iteration(self):
        provider = ScriptedProvider(["child", "grandchild"])
        batches = iter(
            [
                [Evaluation(0, 1), Evaluation(1, 0)],
                [Evaluation(0, 1), Evaluation(0, 1)],
                [Evaluation(2, -1)],
                [Evaluation(3, 0)],
            ]
        )
        seen = []

        async def evaluate(population):
            seen.append((tuple(p.id for p in population), len(provider.calls)))
            return next(batches)

        with patch("slick.prompts.TEMPLATE_ROOT", PROMPTS):
            agent = InteractiveEvolution("brief", provider, evaluate, mutation_topics=["method"])
            result = await agent.run(
                initial=["best", "other"],
                population_size=2,
                iterations=2,
                new_evaluations=2,
                crossover_probability=0,
                seed=2,
            )
        self.assertEqual([calls for _, calls in seen], [0, 1, 1, 2])
        self.assertEqual([p.id for p in result], [0, 3])
        self.assertEqual([r["operation"] for r in agent.attempts], ["mutate", "mutate"])
        self.assertEqual(agent.archive[2].parents, (0,))
        self.assertEqual(agent.completed_iterations, 2)

    async def test_late_votes_count_toward_trigger_but_child_still_needs_feedback(self):
        batches = iter(
            [
                [Evaluation(0, 1), Evaluation(1, -1)],
                [Evaluation(1, 1)] * 10,
                [Evaluation(2, 0)],
                [Evaluation(3, 0)],
            ]
        )
        calls = []
        provider = ScriptedProvider(["child", "grandchild"])

        async def evaluate(population):
            calls.append(len(provider.calls))
            return next(batches)

        with patch("slick.prompts.TEMPLATE_ROOT", PROMPTS):
            agent = InteractiveEvolution("brief", provider, evaluate, mutation_topics=["method"])
            await agent.run(
                initial=["best", "worst"],
                population_size=2,
                iterations=2,
                new_evaluations=2,
                crossover_probability=0,
            )
        self.assertEqual(calls, [0, 1, 1, 2])
        self.assertEqual(len(agent.archive[1].ratings), 11)
        self.assertNotIn(1, [p.id for p in agent.population])

    async def test_generation_failure_retains_population_and_raw_response(self):
        async def evaluate(population):
            return [Evaluation(p.id, 0) for p in population]

        for response, error in [("  ", ValueError), (RuntimeError("offline"), RuntimeError)]:
            with self.subTest(response=response), patch("slick.prompts.TEMPLATE_ROOT", PROMPTS):
                agent = InteractiveEvolution(
                    "brief", ScriptedProvider([response]), evaluate, mutation_topics=["method"]
                )
                with self.assertRaises(error):
                    await agent.run(
                        initial=["a", "b"],
                        population_size=2,
                        iterations=1,
                        new_evaluations=2,
                        crossover_probability=0,
                    )
                self.assertEqual([p.content for p in agent.population], ["a", "b"])
                self.assertEqual(agent.completed_iterations, 0)
                self.assertEqual(len(agent.attempts), 1)
                self.assertIn("error", agent.attempts[0])
                if response == "  ":
                    self.assertEqual(agent.attempts[0]["raw_response"], response)

        with patch("slick.prompts.TEMPLATE_ROOT", PROMPTS):
            agent = InteractiveEvolution(
                "brief",
                ScriptedProvider(["combined", "  "]),
                evaluate,
                mutation_topics=["method"],
            )
            with self.assertRaisesRegex(ValueError, "blank"):
                await agent.run(
                    initial=["a", "b"],
                    population_size=2,
                    iterations=1,
                    new_evaluations=2,
                    crossover_probability=1,
                )
        self.assertEqual([p.content for p in agent.population], ["a", "b"])
        self.assertEqual(agent.completed_iterations, 0)
        self.assertEqual([r["raw_response"] for r in agent.attempts], ["combined", "  "])

    async def test_mean_vote_selection_and_minimum_coverage_including_final_child(self):
        provider = ScriptedProvider(["child"])
        batches = iter(
            [
                [Evaluation(0, 1)] * 5 + [Evaluation(0, -1), Evaluation(1, 1)],
                [Evaluation(1, 1)],
                [Evaluation(2, -1)],
                [Evaluation(2, 1)],
            ]
        )
        calls = []

        async def evaluate(population):
            calls.append(len(provider.calls))
            return next(batches)

        with patch("slick.prompts.TEMPLATE_ROOT", PROMPTS):
            agent = InteractiveEvolution("brief", provider, evaluate, mutation_topics=["method"])
            result = await agent.run(
                initial=["many votes", "higher mean"],
                population_size=2,
                iterations=1,
                new_evaluations=2,
                min_evaluations=2,
                crossover_probability=0,
            )
        self.assertEqual(calls, [0, 0, 1, 1])
        self.assertEqual([p.id for p in result], [1, 2])
        self.assertEqual(agent.archive[0].fitness, 2 / 3)
        self.assertEqual(agent.archive[2].fitness, 0)
        self.assertEqual(agent.archive[2].parents, (1,))

    async def test_bad_feedback_batch_is_atomic_and_empty_batch_stops(self):
        for batch, error in [
            ([Evaluation(0, 1), Evaluation(9, 1)], KeyError),
            ([Evaluation(0, 1), Evaluation(1, 9)], KeyError),
            ([], RuntimeError),
        ]:

            async def evaluate(population):
                return batch

            with self.subTest(batch=batch), patch("slick.prompts.TEMPLATE_ROOT", PROMPTS):
                agent = InteractiveEvolution(
                    "brief", ScriptedProvider([]), evaluate, mutation_topics=["method"]
                )
                with self.assertRaises(error):
                    await agent.run(initial=["a", "b"], population_size=2, iterations=0)
                self.assertEqual(agent.evaluations, [])
                self.assertTrue(all(not p.ratings for p in agent.population))

    async def test_plain_text_templates_from_other_directory_and_fresh_run(self):
        async def evaluate(population):
            return [Evaluation(p.id, 0) for p in population]

        provider = ScriptedProvider([" first ", "second", "third", "fourth"])
        agent = InteractiveEvolution(
            "custom brief", provider, evaluate, mutation_topics=["structure"]
        )
        previous_directory = Path.cwd()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("slick.prompts.TEMPLATE_ROOT", PROMPTS),
        ):
            try:
                os.chdir(directory)
                for _ in range(2):
                    result = await agent.run(population_size=2, iterations=0)
                    self.assertEqual(len(agent.archive), 2)
                    self.assertEqual(len(agent.attempts), 2)
                    self.assertEqual(len(agent.evaluations), 2)
                self.assertEqual([p.content for p in result], ["third", "fourth"])
                rendered = [
                    await InteractiveEvolution.initialize.render(agent),
                    await InteractiveEvolution.crossover.render(agent, "parent A", "parent B"),
                    await InteractiveEvolution.mutate.render(agent, "parent A", "structure"),
                ]
            finally:
                os.chdir(previous_directory)
        for text in rendered:
            self.assertIn("custom brief", text)
            self.assertNotIn("JSON", text)
        self.assertIn("parent B", rendered[1])
        self.assertIn("structure", rendered[2])
        for file in PROMPTS.glob("*.j2"):
            parsed = Environment().parse(file.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
