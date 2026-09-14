"""Offline behavior checks for task-agnostic reflective evolution."""

import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts

from reevo import Config, ReEvo
from tests.providers import ScriptedProvider


class ReEvoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = patch.object(
            prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[1] / "reevo" / "prompts"
        )
        self.root.start()
        self.addCleanup(self.root.stop)

    async def test_unrelated_tasks_use_exact_candidate_text(self):
        for task, candidate in [
            ("Write a greeting", "hello, friend"),
            ("Design a menu", "rice; beans"),
        ]:
            seen = []

            async def evaluate(text):
                seen.append(text)
                return len(text)

            provider = ScriptedProvider([candidate])
            result = await ReEvo(task, provider, evaluate, config=Config(max_evaluations=1)).run()
            self.assertEqual(result.best.candidate, candidate)
            self.assertEqual(seen, [candidate])
            self.assertIn(task, str(provider.calls))

    async def test_reflection_pairs_elite_mutation_and_budget(self):
        for maximize, values in [(False, [9, 5, 3, 1]), (True, [1, 5, 7, 9])]:
            scores = dict(zip(["first", "second", "cross", "mutation"], values))

            async def evaluate(text):
                return scores[text]

            provider = ScriptedProvider(
                ["first", "second", "short advice", "cross", "long advice", "mutation"]
            )
            config = Config(
                initial_size=2,
                population_size=2,
                crossover_rate=0.5,
                mutation_rate=0.5,
                max_evaluations=4,
                maximize=maximize,
            )
            agent = ReEvo("Compose a message", provider, evaluate, config=config)
            result = await agent.run()
            self.assertEqual(result.best.candidate, "mutation")
            self.assertEqual([item.parents for item in result.individuals], [[], [], [0, 1], [2]])
            self.assertEqual(result.reflections[0]["short_term"], ["short advice"])
            self.assertEqual(result.reflections[0]["long_term"], "long advice")
            self.assertIn("cross", str(provider.calls[-1]))
            self.assertIn("long advice", str(provider.calls[-1]))
            self.assertEqual((len(provider.calls), len(result.individuals)), (6, 4))
            self.assertEqual(result.stop_reason, "budget")
            self.assertIn("Create a diverse initial candidate", provider.calls[0])
            self.assertIn("Combine the parents", provider.calls[3])
            self.assertIn("Mutate the elite", provider.calls[5])

    async def test_mutations_share_the_post_crossover_elite_snapshot(self):
        scores = dict(
            zip(["first", "second", "cross", "mutation one", "mutation two"], [9, 5, 3, 1, 2])
        )

        async def evaluate(text):
            return scores[text]

        provider = ScriptedProvider(
            ["first", "second", "advice", "cross", "memory", "mutation one", "mutation two"]
        )
        config = Config(
            initial_size=2,
            population_size=2,
            crossover_rate=0.5,
            mutation_rate=1,
            max_evaluations=5,
        )
        result = await ReEvo("Compose a message", provider, evaluate, config=config).run()
        self.assertEqual([item.parents for item in result.individuals[-2:]], [[2], [2]])
        self.assertEqual(result.best.candidate, "mutation one")
        self.assertNotIn("mutation one", provider.calls[-1])

    async def test_rejections_keep_budget_and_raw_text(self):
        calls = []

        async def evaluate(text):
            calls.append(text)
            if text == "bad":
                raise ValueError("rejected")
            return float("inf") if text == "infinite" else 2

        provider = ScriptedProvider(["  ", "bad", "infinite", "valid"])
        result = await ReEvo("anything", provider, evaluate, config=Config(max_evaluations=4)).run()
        self.assertEqual(len(result.individuals), 4)
        self.assertEqual(
            [item.candidate for item in result.individuals], ["  ", "bad", "infinite", "valid"]
        )
        self.assertTrue(all(item.error for item in result.individuals[:3]))
        self.assertEqual(calls, ["bad", "infinite", "valid"])
        self.assertEqual(result.best.candidate, "valid")

    async def test_reflection_disabled_and_mutation_only(self):
        async def evaluate(text):
            return len(text)

        provider = ScriptedProvider(["initial", "child"])
        result = await ReEvo(
            "anything",
            provider,
            evaluate,
            config=Config(
                initial_size=1,
                population_size=1,
                crossover_rate=0,
                mutation_rate=1,
                short_reflection=False,
                long_reflection=False,
                max_evaluations=2,
            ),
        ).run()
        self.assertEqual(result.best.candidate, "child")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(result.reflections[0]["short_term"], [])

    async def test_seed_and_parent_exhaustion(self):
        async def evaluate(text):
            return 1

        provider = ScriptedProvider([])
        result = await ReEvo(
            "anything", provider, evaluate, seed_candidate="seed", config=Config(max_evaluations=1)
        ).run()
        self.assertEqual(result.best.stage, "seed")
        self.assertEqual(provider.calls, [])
        provider = ScriptedProvider(["same", "quality"])
        result = await ReEvo(
            "anything", provider, evaluate, config=Config(initial_size=2, max_evaluations=3)
        ).run()
        self.assertEqual(result.stop_reason, "no_distinct_parents")
        self.assertEqual(len(provider.calls), 2)

    async def test_failures_and_cancellation_propagate(self):
        import asyncio

        async def evaluate(text):
            return 1

        provider = ScriptedProvider([RuntimeError("provider offline")])
        agent = ReEvo("anything", provider, evaluate)
        with self.assertRaisesRegex(RuntimeError, "provider offline"):
            await agent.run()
        self.assertEqual(agent.result.individuals, [])

        async def cancelled(text):
            raise asyncio.CancelledError

        agent = ReEvo("anything", ScriptedProvider(["text"]), cancelled)
        with self.assertRaises(asyncio.CancelledError):
            await agent.run()

    def test_configuration_is_stored_without_preflight_checks(self):
        config = Config(max_evaluations=0, crossover_rate=0, mutation_rate=0)
        self.assertEqual(config.max_evaluations, 0)
        self.assertEqual(config.crossover_rate, 0)
        self.assertEqual(config.mutation_rate, 0)

    async def test_native_evaluator_errors_propagate(self):
        async def evaluate(text):
            raise TypeError("evaluator bug")

        with self.assertRaisesRegex(TypeError, "evaluator bug"):
            await ReEvo("anything", ScriptedProvider(["candidate"]), evaluate).run()

    async def test_explicit_session_and_method_template_binding(self):
        import os
        import tempfile

        from slick import Session

        async def evaluate(text):
            return 1

        unused = ScriptedProvider([])
        provider = ScriptedProvider(["session candidate"])
        agent = ReEvo(
            "A task independent of Python", unused, evaluate, config=Config(max_evaluations=1)
        )
        before = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                rendered = await ReEvo.initial.render(agent)
                self.assertIn(agent.task, rendered)
                result = await agent.run(session=Session(provider=provider))
            finally:
                os.chdir(before)
        self.assertEqual(result.best.candidate, "session candidate")
        self.assertEqual(unused.calls, [])

    async def test_each_generation_operation_uses_its_own_prompt(self):
        from reevo import Individual

        async def evaluate(text):
            return 1

        agent = ReEvo("Write a greeting", ScriptedProvider([]), evaluate)
        worse = Individual(0, "worse text", "initial", 0, score=2)
        better = Individual(1, "better text", "initial", 0, score=1)
        initial = await ReEvo.initial.render(agent)
        crossover = await ReEvo.crossover.render(agent, worse, better, "compare advice")
        mutation = await ReEvo.mutate.render(agent, better, "carried advice")
        self.assertIn("Create a diverse initial candidate", initial)
        self.assertNotIn("None", initial)
        self.assertIn("Combine the parents", crossover)
        self.assertIn("worse text", crossover)
        self.assertIn("compare advice", crossover)
        self.assertIn("Mutate the elite", mutation)
        self.assertIn("better text", mutation)
        self.assertIn("carried advice", mutation)
