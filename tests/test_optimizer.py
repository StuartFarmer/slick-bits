"""Offline checks for a bounded proposal and feedback dialogue."""

import unittest
from pathlib import Path
from unittest.mock import patch

from slick import Session, prompts

from optimizer import Optimizer, Proposal
from tests.providers import ScriptedProvider


class OptimizerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patch = patch.object(
            prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[1] / "optimizer/prompts"
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    async def test_tasks_feedback_and_acceptance(self):
        for task in ("Write a welcoming invitation", "Plan warehouse deliveries"):
            provider = ScriptedProvider(
                Proposal(description="Idea", content=str(i)) for i in (1, 3, 2)
            )

            async def evaluate(content):
                return float(content)

            agent = Optimizer(task, provider, evaluate)
            result = await agent.run(feedback=("Make it clearer", "Try another approach"))
            self.assertEqual(result.fitness, 3)
            self.assertEqual(len(provider.calls), 3)
            self.assertTrue(all(task in context for context in provider.calls))
            self.assertIn("Make it clearer", provider.calls[1])
            self.assertIn("Try another approach", provider.calls[2])
            self.assertIn('"content": "3"', provider.calls[2])
            self.assertEqual([row["accepted"] for row in agent.history], [True, True, False])

    async def test_failed_revision_preserves_best_and_uses_budget(self):
        provider = ScriptedProvider(
            [
                Proposal(description="Idea", content="1"),
                "bad",
                Proposal(description="Idea", content="nan"),
            ]
        )

        async def evaluate(content):
            return float(content)

        agent = Optimizer("Task", provider, evaluate)
        result = await agent.run(feedback=("Revise", "Revise again"))
        self.assertEqual(result.fitness, 1)
        self.assertEqual(len(provider.calls), 3)
        self.assertTrue(all(row.get("error") for row in agent.history[1:]))

    async def test_invalid_initial_proposal_aborts(self):
        provider = ScriptedProvider(["bad"])

        async def evaluate(content):
            return 0.0

        agent = Optimizer("Task", provider, evaluate)
        with self.assertRaisesRegex(RuntimeError, "initial"):
            await agent.run()
        self.assertEqual(len(provider.calls), 1)

    async def test_caller_owned_session_uses_its_provider(self):
        unused = ScriptedProvider([])
        provider = ScriptedProvider(
            [Proposal(description="Idea", content="a"), Proposal(description="Idea", content="b")]
        )

        async def evaluate(content):
            return 1.0

        agent = Optimizer("Task", unused, evaluate)
        session = Session(provider=provider)
        result = await agent.run(feedback=("Revise",), session=session)
        self.assertTrue(result)
        self.assertEqual(unused.calls, [])
        self.assertEqual(len(provider.calls), 2)

    async def test_minimization_ties_retain_incumbent(self):
        provider = ScriptedProvider(Proposal(description="Idea", content=str(i)) for i in (3, 2, 1))

        async def evaluate(content):
            return max(2.0, float(content))

        agent = Optimizer("Task", provider, evaluate, maximize=False)
        result = await agent.run(feedback=("Improve", "Improve again"))
        self.assertEqual(result.content, "2")
        self.assertEqual([row["accepted"] for row in agent.history], [True, True, False])
