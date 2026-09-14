"""Offline checks for generic AEL selection and generation."""

import unittest
from pathlib import Path
from unittest.mock import patch

from slick import Session, prompts

from ael import AEL, Proposal
from tests.providers import ScriptedProvider


class AELTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = patch.object(
            prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[1] / "ael/prompts"
        )
        self.root.start()
        self.addCleanup(self.root.stop)

    async def test_tasks_budget_snapshot_and_elitism(self):
        for task in ("Write a welcoming invitation", "Plan warehouse deliveries"):
            provider = ScriptedProvider(
                Proposal(description="Idea", content=str(i)) for i in range(10)
            )

            async def evaluate(content):
                return -float(content)

            agent = AEL(task, provider, evaluate)
            population = await agent.run(
                population_size=2, generations=2, parents=2, offspring=1, crossover=1, mutation=1
            )
            self.assertEqual(len(provider.calls), 10)
            self.assertIn("Create a fresh strategy", provider.calls[0])
            self.assertIn("Combine useful ideas", provider.calls[2])
            self.assertIn("Revise the parent", provider.calls[3])
            self.assertEqual(population[0].fitness, -9)
            self.assertTrue(all(task in context for context in provider.calls))
            for attempt in agent.attempts:
                if attempt["operation"] == "crossover":
                    previous = agent.history[attempt["generation"] - 1]
                    self.assertTrue(set(attempt["parents"]) <= {item.id for item in previous})

    async def test_no_variation_does_not_generate_or_evaluate(self):
        provider = ScriptedProvider([Proposal(description="Idea", content="one")])
        evaluated = []

        async def evaluate(content):
            evaluated.append(content)
            return 1.0

        agent = AEL("Compose a title", provider, evaluate)
        population = await agent.run(
            population_size=1, parents=1, generations=3, crossover=0, mutation=0
        )
        self.assertEqual((len(provider.calls), len(evaluated)), (1, 1))
        self.assertEqual(population[0].id, 1)

    async def test_failed_candidates_are_bounded_and_keep_incumbent(self):
        async def evaluate(content):
            return float(content)

        provider = ScriptedProvider(["bad"] * 3)
        agent = AEL("Task", provider, evaluate)
        with self.assertRaisesRegex(RuntimeError, "initial"):
            await agent.run(population_size=1, parents=1, generations=0)
        self.assertEqual(len(provider.calls), 3)
        provider = ScriptedProvider(
            [Proposal(description="Idea", content=x) for x in ("2", "nan", "1")]
        )
        agent = AEL("Task", provider, evaluate, maximize=True)
        result = await agent.run(population_size=1, parents=1, generations=2, mutation=0)
        self.assertEqual(result[0].fitness, 2)
        self.assertIn("finite", agent.attempts[1]["error"])

    async def test_caller_owned_session_uses_its_provider(self):
        unused = ScriptedProvider([])
        provider = ScriptedProvider([Proposal(description="Idea", content="a")])

        async def evaluate(content):
            return 1.0

        agent = AEL("Task", unused, evaluate)
        session = Session(provider=provider)
        result = await agent.run(population_size=1, parents=1, generations=0, session=session)
        self.assertTrue(result)
        self.assertEqual(unused.calls, [])
        self.assertEqual(len(provider.calls), 1)

    async def test_provider_outage_aborts_with_attempt_record(self):
        from slick.providers import ProviderError

        async def evaluate(content):
            return 1.0

        for error in (ProviderError("offline"), TimeoutError("offline"), ValueError("offline")):
            with self.subTest(error=type(error).__name__):
                provider = ScriptedProvider([error, Proposal(description="Idea", content="valid")])
                agent = AEL("Task", provider, evaluate)
                with self.assertRaises(type(error)) as raised:
                    await agent.run(population_size=1, parents=1, generations=0)
                self.assertIs(raised.exception, error)
                self.assertEqual(len(provider.calls), 1)
                self.assertIn("offline", agent.attempts[0]["error"])

    async def test_evaluator_timeout_rejects_candidate_without_aborting(self):
        async def evaluate(content):
            if content == "slow":
                raise TimeoutError("evaluation timed out")
            return 1.0

        provider = ScriptedProvider(
            Proposal(description="Idea", content=text) for text in ("slow", "valid")
        )
        agent = AEL("Task", provider, evaluate)
        result = await agent.run(population_size=1, parents=1, generations=0)
        self.assertEqual(result[0].content, "valid")
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("evaluation timed out", agent.attempts[0]["error"])

    async def test_distinct_prompt_methods_route_to_local_templates(self):
        async def evaluate(content):
            return 1.0

        agent = AEL("Task", ScriptedProvider([]), evaluate)
        parent = Proposal(description="Idea", content="parent")
        for method, args, phrase in (
            (AEL.initialization, (), "Create a fresh strategy"),
            (AEL.crossover, ([parent],), "Combine useful ideas"),
            (AEL.mutation, ([parent],), "Revise the parent"),
        ):
            rendered = await method.render(agent, *args)
            self.assertIn(phrase, rendered)
            self.assertIn("Lower fitness is better", rendered)
