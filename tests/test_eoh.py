"""Offline checks for generic evolution of heuristics."""

import unittest
from pathlib import Path
from unittest.mock import patch

from slick import Session, prompts

from eoh import EoH, Proposal
from tests.providers import ScriptedProvider


class EoHTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "eoh/prompts"
        self.patch = patch.object(prompts, "TEMPLATE_ROOT", root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    async def test_tasks_operators_budgets_and_snapshot(self):
        for task in ("Write a welcoming invitation", "Plan warehouse deliveries"):
            provider = ScriptedProvider(
                Proposal(description="Idea", content=str(i)) for i in range(22)
            )

            async def evaluate(content):
                return float(content)

            agent = EoH(task, provider, evaluate)
            result = await agent.run(population_size=2, generations=2, parents=2)
            self.assertEqual(len(provider.calls), 22)
            for index, phrase in (
                (0, "without parent candidates"),
                (2, "as different as possible"),
                (4, "common idea"),
                (6, "reasoning or structure"),
                (8, "specific choices and settings"),
                (10, "redundant components"),
            ):
                self.assertIn(phrase, provider.calls[index])
            self.assertEqual([item.fitness for item in result], [21, 20])
            self.assertTrue(all(task in context for context in provider.calls))
            self.assertEqual(
                {row["operation"] for row in agent.attempts}, {"INIT", "E1", "E2", "M1", "M2", "M3"}
            )
            for row in agent.attempts[2:]:
                self.assertEqual(len(row["parents"]), 2 if row["operation"].startswith("E") else 1)
                self.assertTrue(
                    set(row["parents"])
                    <= {item.id for item in agent.history[row["generation"] - 1]}
                )

    async def test_failures_consume_attempts_and_preserve_incumbents(self):
        async def evaluate(content):
            return float(content)

        provider = ScriptedProvider(
            [
                Proposal(description="Idea", content="1"),
                "bad",
                Proposal(description="Idea", content="nan"),
                "bad",
                "bad",
                "bad",
            ]
        )
        agent = EoH("Task", provider, evaluate)
        result = await agent.run(population_size=1, parents=1, generations=1)
        self.assertEqual(result[0].id, 1)
        self.assertEqual(len(provider.calls), 6)
        self.assertTrue(all(row.get("error") for row in agent.attempts[1:]))

    async def test_initialization_is_bounded(self):
        async def evaluate(content):
            self.fail("Malformed candidates cannot be evaluated")

        provider = ScriptedProvider(["bad"] * 3)
        agent = EoH("Task", provider, evaluate)
        with self.assertRaisesRegex(RuntimeError, "initial"):
            await agent.run(population_size=1, parents=1)
        self.assertEqual(len(provider.calls), 3)

    async def test_caller_owned_session_uses_its_provider(self):
        unused = ScriptedProvider([])
        provider = ScriptedProvider([Proposal(description="Idea", content="a")])

        async def evaluate(content):
            return 1.0

        agent = EoH("Task", unused, evaluate)
        session = Session(provider=provider)
        result = await agent.run(population_size=1, parents=1, generations=0, session=session)
        self.assertTrue(result)
        self.assertEqual(unused.calls, [])
        self.assertEqual(len(provider.calls), 1)

    async def test_minimization_rank_weights_and_stable_ties(self):
        provider = ScriptedProvider(Proposal(description="Idea", content=str(i)) for i in range(4))
        weights_seen = []

        def choose(population, *, weights, k):
            weights_seen.append(weights.copy())
            return [0]

        async def evaluate(content):
            return float(content) if int(content) < 2 else 0.0

        agent = EoH("Task", provider, evaluate, maximize=False)
        with patch("eoh.agent.random.Random.choices", side_effect=choose):
            result = await agent.run(population_size=2, parents=2, generations=1, operators=("E1",))
        self.assertEqual(weights_seen, [[1 / 3, 1 / 4], [1 / 4]] * 2)
        self.assertEqual([item.id for item in result], [1, 3])
        self.assertEqual(agent.attempts[2]["parents"], (1, 2))

    async def test_provider_error_consumes_initialization_attempt(self):
        from slick.providers import ProviderError

        provider = ScriptedProvider(
            [ProviderError("offline"), Proposal(description="Idea", content="one")]
        )

        async def evaluate(content):
            return 1.0

        agent = EoH("Task", provider, evaluate)
        result = await agent.run(population_size=1, parents=1, generations=0)
        self.assertEqual(result[0].id, 2)
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("offline", agent.attempts[0]["error"])

    async def test_each_operator_has_its_own_prompt_method(self):
        async def evaluate(content):
            return 1.0

        agent = EoH("Task", ScriptedProvider([]), evaluate)
        for method, phrase in (
            (EoH.initialize, "without parent candidates"),
            (EoH.explore_diverse, "as different as possible"),
            (EoH.explore_shared, "common idea"),
            (EoH.modify_structure, "reasoning or structure"),
            (EoH.tune_settings, "specific choices and settings"),
            (EoH.simplify, "redundant components"),
        ):
            args = () if method is EoH.initialize else ([],)
            rendered = await method.render(agent, *args)
            self.assertIn(phrase, rendered)
            self.assertIn("Higher fitness is better", rendered)
