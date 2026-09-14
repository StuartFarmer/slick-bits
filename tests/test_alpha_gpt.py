"""Exercise Alpha-GPT's search and feedback loop without model calls."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from alpha_gpt import AlphaGPT, CandidateRejected, Document, Evaluation
from tests.providers import ScriptedProvider

IDEA = '{"idea":"Find short rules", "directions":"Simplify", "crossover_rate":1, "mutation_rate":1}'
SEEDS = '{"candidates":[{"content":"a", "explanation":"baseline"}, {"content":"bb", "explanation":"alternative"}]}'
REVIEW = '{"summary":"Measured improvement", "next_direction":"Try a different rule"}'


class AlphaGPTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "alpha_gpt/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)

    def agent(self, responses, **options):
        async def evaluate(content):
            return Evaluation(float(len(content)), feedback="Measured length")

        async def mutate(content, directions, rng):
            return content + "m"

        async def crossover(left, right, directions, rng):
            return left + right

        return AlphaGPT(
            "Optimize a text rule",
            ScriptedProvider(responses),
            options.pop("evaluate", evaluate),
            mutate=options.pop("mutate", mutate),
            crossover=options.pop("crossover", crossover),
            seed_count=2,
            population_size=2,
            **options,
        )

    async def test_interactive_search_and_feedback(self):
        directions = []

        async def mutate(content, direction, rng):
            directions.append(direction)
            return content + "m"

        async def feedback(report):
            return "Keep improving but avoid long rules"

        agent = self.agent([IDEA, SEEDS, REVIEW] * 2, mutate=mutate, generations=1)
        result = await agent.run("Find rules", rounds=2, feedback=feedback)
        self.assertEqual(len(result.rounds), 2)
        self.assertGreater(result.best.evaluation.score, 2)
        self.assertEqual(result.calls, 6)
        self.assertEqual(directions, ["Simplify"] * 4)
        self.assertIn("avoid long rules", agent.calls[3]["prompt"])
        self.assertIn("Measured length", agent.calls[2]["prompt"])
        self.assertEqual(result.rounds[0].seeds[0].content, "a")
        self.assertEqual(len(agent.generations), 2)

    async def test_autonomous_hierarchy_scopes_and_memory(self):
        queries = []

        async def retrieve(query):
            queries.append(query)
            return [Document(query.stage, query.stage + " document")]

        agent = self.agent(
            [
                '{"id":"categories"}',
                '{"id":"subcategories"}',
                '{"content":"Explore fields", "explanation":"New idea"}',
                IDEA,
                SEEDS,
                REVIEW,
            ],
            retrieve=retrieve,
            generations=0,
        )
        result = await agent.run(rounds=1)
        self.assertEqual(result.calls, 6)
        self.assertEqual(
            [q.stage for q in queries],
            [
                "successes",
                "categories",
                "subcategories",
                "fields",
                "literature",
            ],
        )
        self.assertEqual(queries[2].path, ("categories",))
        self.assertEqual(queries[3].path, ("categories", "subcategories"))
        self.assertIn("successes document", agent.calls[0]["prompt"])
        self.assertIn("fields document", agent.calls[3]["prompt"])

    async def test_rejection_repair_dedup_and_minimization(self):
        seen = []

        async def evaluate(content):
            seen.append(content)
            if content == "a":
                raise CandidateRejected("Unknown symbol")
            return Evaluation(float(len(content)))

        agent = self.agent(
            [
                IDEA,
                SEEDS,
                '{"content":"bb", "explanation":"Correct symbol"}',
                REVIEW,
            ],
            evaluate=evaluate,
            maximize=False,
            generations=0,
        )
        result = await agent.run("Rules")
        self.assertEqual(seen, ["a", "bb"])
        self.assertEqual(result.best.content, "bb")
        self.assertEqual(result.evaluations, 2)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("Unknown symbol", agent.calls[2]["prompt"])

    async def test_budgets_and_invalid_children_do_not_loop(self):
        async def mutate(content, directions, rng):
            return "bad"

        async def evaluate(content):
            return Evaluation(float("nan") if content == "bad" else 1.0)

        agent = self.agent([IDEA, SEEDS, REVIEW], evaluate=evaluate, mutate=mutate, generations=3)
        result = await agent.run("Rules")
        self.assertEqual(result.evaluations, 3)
        self.assertEqual(len(agent.generations), 3)
        self.assertEqual(len(result.failures), 1)
        limited = self.agent([IDEA, SEEDS], generations=0, max_calls=2)
        partial = await limited.run("Rules")
        self.assertEqual(partial.stop_reason, "budget")
        self.assertIsNotNone(partial.best)
        self.assertEqual(partial.rounds, ())

    async def test_bad_generation_and_callback_errors_retain_state(self):
        agent = self.agent([IDEA, "not json"], generations=0)
        with self.assertRaises(ValueError):
            await agent.run("Rules")
        self.assertEqual(agent.calls[-1]["response"], "not json")
        self.assertIn("error", agent.calls[-1])

        async def broken(content):
            raise RuntimeError("Evaluator offline")

        agent = self.agent([IDEA, SEEDS], evaluate=broken, generations=0)
        with self.assertRaisesRegex(RuntimeError, "Evaluator offline"):
            await agent.run("Rules")
        self.assertEqual(agent.evaluations, 1)

    async def test_unknown_retrieved_id_is_rejected(self):
        async def retrieve(query):
            return [Document("known", "description")]

        agent = self.agent(['{"id":"invented"}'], retrieve=retrieve)
        with self.assertRaisesRegex(ValueError, "unknown"):
            await agent.run(rounds=1)

    async def test_elitism_minimization_and_snapshot_parents(self):
        parents = []

        async def crossover(left, right, direction, rng):
            parents.extend([left, right])
            return "long-child"

        async def mutate(content, direction, rng):
            return content

        agent = self.agent(
            [IDEA, SEEDS, REVIEW], crossover=crossover, mutate=mutate, maximize=False, generations=1
        )
        result = await agent.run("Rules")
        self.assertEqual(result.best.content, "a")
        self.assertTrue(set(parents) <= {"a", "bb"})

    async def test_empty_offspring_and_evaluation_cap(self):
        async def mutate(content, directions, rng):
            return ""

        agent = self.agent([IDEA, SEEDS, REVIEW], mutate=mutate, generations=2)
        result = await agent.run("Rules")
        self.assertEqual(result.evaluations, 2)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("blank", result.failures[0].error)
        capped = self.agent([IDEA, SEEDS], max_evaluations=1)
        result = await capped.run("Rules")
        self.assertEqual(result.evaluations, 1)
        self.assertEqual(result.best.content, "a")
        self.assertEqual(result.stop_reason, "budget")

    async def test_qualification_and_redundancy(self):
        async def evaluate(content):
            return Evaluation(len(content), qualified=content != "a", feedback="Not allowed")

        agent = self.agent(
            [IDEA, SEEDS, REVIEW], evaluate=evaluate, repair_attempts=0, generations=0
        )
        result = await agent.run("Rules")
        self.assertEqual(result.best.content, "bb")
        self.assertEqual(result.failures[0].error, "Not allowed")
        agent = self.agent([IDEA, SEEDS, REVIEW], generations=0, redundant=lambda left, right: True)
        self.assertEqual(len((await agent.run("Rules")).population), 1)

    async def test_feedback_stop_and_tool_failure(self):
        async def stop(report):
            return None

        agent = self.agent([IDEA, SEEDS, REVIEW], generations=0)
        self.assertEqual((await agent.run("Rules", feedback=stop)).stop_reason, "feedback")
        agent = self.agent([(IDEA, ["tool request"])])
        with self.assertRaisesRegex(ValueError, "tool requests"):
            await agent.run("Rules")
        self.assertEqual(agent.calls[-1]["response"], IDEA)

    async def test_all_templates_from_another_launch_directory(self):
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root.parent)
        # End-to-end tests above render every operation including repair and navigation.
        agent = self.agent([IDEA, SEEDS, REVIEW], generations=0)
        await agent.run("Other task")
        from alpha_gpt import Idea

        idea = Idea.model_validate_json(IDEA)
        rendered = await AlphaGPT.initialize.render(agent, idea, [])
        self.assertIn("Optimize a text rule", rendered)
        self.assertIn('"candidates"', rendered)
        for template in self.root.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "optional faiss-cpu not installed")
    async def test_real_faiss_scopes_and_nearest_neighbors(self):
        from alpha_gpt import Query
        from alpha_gpt.retrieval import FaissRetriever

        async def embed(text):
            return [1.0, 0.0]

        retrieve = FaissRetriever(
            {
                ("fields", ("a", "b")): (
                    [Document("near", "near"), Document("far", "far")],
                    [[1, 0], [0, 1]],
                ),
                ("fields", ("other", "b")): ([Document("other", "other")], [[1, 0]]),
            },
            embed,
        )
        self.assertEqual(
            await retrieve(Query("fields", "q", ("a", "b"), 1)), (Document("near", "near"),)
        )
        self.assertEqual(len(await retrieve(Query("fields", "q", ("a", "b"), 9))), 2)
        self.assertEqual(await retrieve(Query("fields", "q", ("missing",))), ())


if __name__ == "__main__":
    unittest.main()
