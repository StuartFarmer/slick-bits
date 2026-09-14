"""Offline checks for topic-guided evolution and its failure boundaries."""

import math
import random
import unittest
from pathlib import Path

from jinja2 import Environment, nodes
from slick import prompts

from greentea import ErrorCase, Evaluation, GreenTEA, KMeansTopics
from tests.providers import ScriptedProvider

FEEDBACK = "<erroranalysis>Missing checks</erroranalysis><suggestion>Check units</suggestion>"


class GreenTEAChecks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        old = prompts.TEMPLATE_ROOT
        prompts.TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "greentea/prompts"
        self.addCleanup(setattr, prompts, "TEMPLATE_ROOT", old)

    async def test_topic_feedback_roulette_elitism_and_cache(self):
        errors = (
            ErrorCase("minor case", "rare reference", "wrong"),
            ErrorCase("major one", "common reference", "wrong", "model reasoning"),
            ErrorCase("major two", "common reference", "wrong"),
        )
        embedded = []

        def encode(texts):
            embedded.append(list(texts))
            return [[100.0] if text == "rare reference" else [0.0] for text in texts]

        evaluated = []

        async def evaluate(text):
            evaluated.append(text)
            return Evaluation({"weak": 0, "strong": 1, "better": 2}[text], errors)

        analyzer = ScriptedProvider([FEEDBACK] * 3)
        generator = ScriptedProvider(
            ["<ChildPrompt>combined</ChildPrompt>", "<OptimizedPrompt>better</OptimizedPrompt>"] * 4
        )
        agent = GreenTEA(
            "Improve arbitrary instructions.",
            generator,
            evaluate,
            KMeansTopics(encode, min_clusters=2, max_clusters=2),
            analyzer_provider=analyzer,
        )
        result = await agent.run(["weak", "strong"], iterations=2)
        self.assertEqual(evaluated, ["weak", "strong", "better"])
        self.assertEqual([p.prompt for p in result["population"]], ["better", "strong"])
        self.assertEqual([h["best_score"] for h in result["history"]], [1, 2, 2])
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(result["optimizer_calls"], 11)
        self.assertEqual(result["cache_hits"], 3)
        self.assertEqual(embedded, [[e.expected for e in errors]] * 3)
        for context in analyzer.calls:
            self.assertIn("major one", context)
            self.assertIn("major two", context)
            self.assertIn("model reasoning", context)
            self.assertNotIn("minor case", context)
        self.assertIn("Parent 1:\nstrong", generator.calls[0])
        self.assertIn("Parent 2:\nstrong", generator.calls[0])
        self.assertIn("Check units", generator.calls[1])
        self.assertIn("combined", generator.calls[1])
        self.assertNotIn("better", generator.calls[2])  # frozen first generation

    async def test_zero_fitness_ties_no_errors_and_seed(self):
        async def evaluate(text):
            return Evaluation(0)

        def unused_topics(texts, rng):
            self.fail("No errors must skip topic modeling")

        runs = []
        for _ in range(2):
            provider = ScriptedProvider(
                ["<ChildPrompt>x</ChildPrompt>", "<OptimizedPrompt>new</OptimizedPrompt>"] * 4
            )
            agent = GreenTEA("Write a poem.", provider, evaluate, unused_topics)
            result = await agent.run(["old", "kept"], iterations=2, seed=9)
            self.assertEqual([p.prompt for p in result["population"]], ["old", "kept"])
            self.assertEqual(result["optimizer_calls"], 8)
            runs.append(provider.calls)
        self.assertEqual(*runs)

    async def test_malformed_output_keeps_raw_and_never_evaluates_child(self):
        for response in (
            "untagged",
            "<ChildPrompt> </ChildPrompt>",
            "<ChildPrompt>x</ChildPrompt><ChildPrompt>y</ChildPrompt>",
        ):
            evaluated = []

            async def evaluate(text):
                evaluated.append(text)
                return Evaluation(1)

            agent = GreenTEA("Task", ScriptedProvider([response]), evaluate, None)
            with self.subTest(response=response), self.assertRaises(ValueError) as caught:
                await agent.run(["initial"], iterations=1)
            self.assertIn(repr(response), str(caught.exception))
            self.assertEqual(evaluated, ["initial"])
            self.assertEqual(agent.responses[-1]["raw"], response)
            self.assertEqual(agent.optimizer_calls, 1)

    async def test_fitness_and_evaluator_failures_propagate(self):
        for score in (-1, math.nan, math.inf):

            async def evaluate(text):
                return Evaluation(score)

            agent = GreenTEA("Task", ScriptedProvider([]), evaluate, None)
            with self.subTest(score=score), self.assertRaises(ValueError):
                await agent.run(["initial"], iterations=0)
            self.assertEqual(agent.evaluations, 1)

        async def broken(text):
            raise RuntimeError("evaluation failed")

        agent = GreenTEA("Task", ScriptedProvider([]), broken, None)
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            await agent.run(["initial"])

    async def test_final_generation_is_evaluated_and_runs_reset(self):
        evaluated = []

        async def evaluate(text):
            evaluated.append(text)
            return Evaluation({"initial": 1, "worse": 0, "best": 2}[text])

        provider = ScriptedProvider(
            [
                "<ChildPrompt>one</ChildPrompt>",
                "<OptimizedPrompt>worse</OptimizedPrompt>",
                "<ChildPrompt>two</ChildPrompt>",
                "<OptimizedPrompt>best</OptimizedPrompt>",
            ]
            * 2
        )
        agent = GreenTEA("Task", provider, evaluate, None)
        first = await agent.run(["initial"], iterations=2)
        second = await agent.run(["initial"], iterations=2)
        self.assertEqual(first, second)
        self.assertEqual(first["best"].prompt, "best")
        self.assertEqual([h["best_score"] for h in first["history"]], [1, 1, 2])
        self.assertEqual(evaluated, ["initial", "worse", "best"] * 2)

    async def test_analysis_mutation_and_transport_failures_stop_without_retry(self):
        async def evaluate(text):
            return Evaluation(1)

        malformed = "<OptimizedPrompt></OptimizedPrompt>"
        for failure in (malformed, RuntimeError("transport failed")):
            provider = ScriptedProvider(["<ChildPrompt>child</ChildPrompt>", failure])
            agent = GreenTEA("Task", provider, evaluate, None)
            with self.assertRaises((ValueError, RuntimeError)):
                await agent.run(["initial"], iterations=1)
            self.assertEqual(agent.evaluations, 1)
            self.assertEqual(agent.optimizer_calls, 2)
            self.assertEqual(len(provider.calls), 2)
        agent = GreenTEA("Task", ScriptedProvider([]), evaluate, None)
        raw = "<erroranalysis>Analysis without suggestion</erroranalysis>"
        with self.assertRaises(ValueError):
            await agent.analyze("initial", [], provider=ScriptedProvider([raw]))
        self.assertEqual(agent.responses, [{"operation": "analyze", "raw": raw}])

    async def test_all_operations_render_without_conditional_templates(self):
        agent = GreenTEA("Sort names.", ScriptedProvider([]), None, None)
        cases = [ErrorCase("input", "expected", "predicted")]
        contexts = [
            await GreenTEA.analyze.render(agent, "instruction", cases),
            await GreenTEA.crossover.render(agent, "first", "second"),
            await GreenTEA.mutate.render(
                agent, "child", "first", "second", cases, cases, FEEDBACK, FEEDBACK
            ),
        ]
        for context in contexts:
            self.assertIn(agent.task, context)
        for path in prompts.TEMPLATE_ROOT.glob("*.j2"):
            self.assertEqual(
                list(Environment().parse(path.read_text()).find_all((nodes.If, nodes.CondExpr))), []
            )


class TopicChecks(unittest.TestCase):
    def test_degenerate_embeddings_and_small_pools(self):
        topics = KMeansTopics(lambda texts: [[1.0, 1.0] for _ in texts], max_examples=2)
        self.assertEqual(topics([], random.Random(0)), [])
        self.assertEqual(topics(["one"], random.Random(0)), [0])
        self.assertEqual(topics(["same"] * 5, random.Random(0)), [0, 1])

    def test_reference_embedding_majority_and_sampling(self):
        topics = KMeansTopics(
            lambda texts: [[float(t)] for t in texts], min_clusters=1, max_clusters=2
        )
        self.assertEqual(topics(["0", "0", "0", "100"], random.Random(0)), [0, 1, 2])
        topics.penalty = 2
        self.assertEqual(topics(["0", "0", "0", "100"], random.Random(0)), [0, 1, 2, 3])
        topics = KMeansTopics(lambda texts: [[1.0] for _ in texts], sample_size=4)
        selected = topics(["x"] * 10, random.Random(3))
        self.assertEqual(selected, topics(["x"] * 10, random.Random(3)))
        self.assertEqual(len(set(selected)), 4)


if __name__ == "__main__":
    unittest.main()
