"""Offline behavior checks for task-agnostic quality-uncertainty evolution."""

import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts

from qube import QUBE, Evaluation
from tests.providers import ScriptedProvider


class QUBETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = patch.object(
            prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[1] / "qube" / "prompts"
        )
        self.root.start()
        self.addCleanup(self.root.stop)

    async def test_extreme_finite_offspring_quality_does_not_overflow(self):
        import math

        for scores in [(1e308, 1e308), (-1e308, 1e308)]:
            values = iter((0.0, *scores))

            async def evaluate(text):
                return Evaluation(next(values), (1.0,))

            agent = QUBE(
                "Compare text",
                ScriptedProvider(["first", "second"]),
                evaluate,
                seed_candidate="seed",
                samples=2,
                islands=1,
                k=0,
                reset_interval=0,
            )
            await agent.run()
            cluster = next(iter(agent.islands[0].clusters.values()))
            quality = cluster.uiq(3, 0)
            self.assertTrue(math.isfinite(quality))
            self.assertEqual(quality, scores[0] / 2 + scores[1] / 2)

    async def test_unrelated_tasks(self):
        for task, candidate in [
            ("Write a greeting", "hello, friend"),
            ("Design a menu", "rice; beans"),
        ]:
            seen = []

            async def evaluate(text):
                seen.append(text)
                return Evaluation(score=len(text), signature=(float(len(text)),))

            provider = ScriptedProvider([candidate])
            result = await QUBE(task, provider, evaluate, seed_candidate="hi", samples=1).run()
            self.assertEqual(result.best.candidate, candidate)
            self.assertEqual(seen, ["hi", candidate])
            self.assertIn(task, str(provider.calls))

    async def test_rejected_sample_budget_and_resets(self):
        async def evaluate(text):
            if text == "bad":
                raise ValueError("invalid candidate")
            if text == "infinite":
                return Evaluation(float("inf"), (1.0,))
            if text == "wrong dimension":
                return Evaluation(2, (1.0, 2.0))
            return Evaluation(len(text), (float(len(text)),))

        provider = ScriptedProvider(["bad", "infinite", "wrong dimension", "  ", "better seed"])
        agent = QUBE(
            "anything",
            provider,
            evaluate,
            seed_candidate="seed",
            samples=5,
            islands=4,
            reset_interval=2,
        )
        result = await agent.run()
        self.assertEqual((len(result.samples), result.accepted), (5, 1))
        self.assertTrue(all(sample.error for sample in result.samples[:4]))
        self.assertEqual(result.best.candidate, "better seed")
        self.assertEqual(len(result.resets), 4)
        self.assertEqual({event["sample"] for event in result.resets}, {2, 4})
        self.assertEqual(len(provider.calls), 5)

    async def test_signature_grouping_and_quality_selection(self):
        import math
        import random

        from qube.agent import Candidate, Cluster, Island

        island = Island()
        for text, signature, score in [
            ("a", (1.0, 2.0), 100),
            ("b", (1.0, 2.0), 100),
            ("c", (2.0, 1.0), 100),
            ("d", (3.0, 0.0), 1),
        ]:
            island.add(Candidate(text, signature, score))
        self.assertEqual(len(island.clusters), 3)
        first, second, third = island.clusters.values()
        self.assertEqual(len(first.candidates), 2)
        self.assertEqual(first.uiq(10, 2), math.inf)
        for cluster, quality in zip([first, second, third], [0, 5, 10]):
            cluster.visits, cluster.offspring_count, cluster.offspring_mean = 1, 1, quality
        parents, selected = island.select(random.Random(0), 10, 0, 1)
        self.assertEqual(selected, [third, second])
        self.assertEqual([parent.score for parent in parents], [1, 100])
        self.assertAlmostEqual(second.uiq(10, 2), 5 + 2 * math.sqrt(math.log(10)))
        cluster = Cluster([Candidate("a", (1.0,), 1), Candidate("a" * 100, (1.0,), 1)])
        self.assertEqual(cluster.sample(random.Random(0), 1).candidate, "a")

    async def test_reset_uses_offspring_quality(self):
        from qube.agent import Candidate

        async def evaluate(text):
            return Evaluation(1, (1.0,))

        agent = QUBE(
            "anything", ScriptedProvider([]), evaluate, seed_candidate="seed", islands=4, k=0
        )
        for index, island in enumerate(agent.islands):
            island.add(Candidate(str(index), (float(index),), 100 - index))
            cluster = next(iter(island.clusters.values()))
            cluster.visits, cluster.offspring_count, cluster.offspring_mean = 10, 1, index
        old = list(agent.islands)
        events = agent.reset(10)
        self.assertEqual({event["island"] for event in events}, {0, 1})
        self.assertIs(agent.islands[2], old[2])
        self.assertIsNot(agent.islands[0], old[0])
        self.assertTrue(all(cluster.visits == 0 for cluster in agent.islands[0].clusters.values()))

    async def test_failures_and_cancellation_propagate(self):
        import asyncio

        async def evaluate(text):
            return Evaluation(1, (1.0,))

        provider = ScriptedProvider([RuntimeError("provider offline")])
        agent = QUBE("anything", provider, evaluate, seed_candidate="seed")
        with self.assertRaisesRegex(RuntimeError, "provider offline"):
            await agent.run()
        self.assertEqual(agent.result.samples, [])

        async def cancelled(text):
            raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await QUBE("anything", ScriptedProvider([]), cancelled, seed_candidate="seed").run()

    async def test_temperature_error_surfaces_during_sampling(self):
        evaluated = []

        async def evaluate(text):
            evaluated.append(text)
            return Evaluation(1, (1.0,))

        provider = ScriptedProvider([])
        agent = QUBE("Task", provider, evaluate, seed_candidate="seed", temperature=0)
        with self.assertRaises(ZeroDivisionError):
            await agent.run()
        self.assertEqual(evaluated, ["seed"])
        self.assertEqual(provider.calls, [])

    async def test_finite_evaluations(self):
        async def evaluate(text):
            return Evaluation(float("nan"), (1.0,))

        provider = ScriptedProvider([])
        with self.assertRaises(ValueError):
            await QUBE("anything", provider, evaluate, seed_candidate="seed").run()
        for score, signature in [(1, (float("inf"),)), (float("nan"), (1,)), (1, ())]:
            with self.assertRaises(ValueError):
                Evaluation(score, signature)
        self.assertEqual(provider.calls, [])

    async def test_session_single_cluster_visits_and_rejection_quality(self):
        from slick import Session

        async def evaluate(text):
            if text == "bad":
                raise ValueError("rejected")
            return Evaluation(1, (1.0,))

        unused = ScriptedProvider([])
        provider = ScriptedProvider(["valid", "bad"])
        agent = QUBE(
            "Text-only task",
            unused,
            evaluate,
            seed_candidate="seed",
            samples=2,
            islands=1,
            reset_interval=0,
        )
        result = await agent.run(session=Session(provider=provider))
        cluster = next(iter(agent.islands[0].clusters.values()))
        self.assertEqual(
            (cluster.visits, cluster.offspring_count, cluster.offspring_mean), (2, 1, 1)
        )
        self.assertEqual(len(cluster.candidates), 2)
        self.assertEqual(result.best.candidate, "seed")
        self.assertEqual(unused.calls, [])
        rendered = await QUBE.generate.render(agent, result.best, result.best)
        self.assertIn(agent.task, rendered)

    async def test_native_evaluator_errors_propagate(self):
        async def evaluate(text):
            if text != "seed":
                raise AttributeError("evaluator bug")
            return Evaluation(1, (1.0,))

        agent = QUBE("anything", ScriptedProvider(["candidate"]), evaluate, seed_candidate="seed")
        with self.assertRaisesRegex(AttributeError, "evaluator bug"):
            await agent.run()
