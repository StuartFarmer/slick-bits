"""Offline checks for generic evolution of heuristics."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import Session, prompts
from slick.providers import ProviderError

from eoh import CandidateRejected, EoH, Proposal
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
            return [population[0]] * k

        async def evaluate(content):
            return float(content) if int(content) < 2 else 0.0

        agent = EoH("Task", provider, evaluate, maximize=False)
        with patch("eoh.agent.random.Random.choices", side_effect=choose):
            result = await agent.run(population_size=2, parents=2, generations=1, operators=("E1",))
        self.assertEqual(weights_seen, [[1 / 3, 1 / 4]] * 2)
        self.assertEqual([item.id for item in result], [1, 3])
        self.assertEqual(agent.attempts[2]["parents"], (1, 1))

    async def test_content_is_preserved_and_raw_failures_are_logged(self):
        content = "\n# Keep this artifact intact\ndef heuristic(x):\n    return x\n\n"
        proposal = {"description": "Identity heuristic", "content": content}
        raw = json.dumps(proposal)
        provider = ScriptedProvider(["invalid JSON", raw])
        seen = []

        async def evaluate(candidate):
            seen.append(candidate)
            return 1.0

        agent = EoH("Implement heuristic(x)", provider, evaluate)
        result = await agent.run(population_size=1, generations=0)
        self.assertEqual(seen, [content])
        self.assertEqual(result[0].content, content)
        self.assertEqual([r["raw_response"] for r in agent.attempts], ["invalid JSON", raw])
        self.assertEqual(agent.evaluations, 1)

    async def test_session_recovers_after_malformed_proposal(self):
        session = Session(
            provider=ScriptedProvider(["bad", Proposal(description="Idea", content="x")])
        )

        async def evaluate(content):
            return 1.0

        agent = EoH("Task", ScriptedProvider([]), evaluate)
        result = await agent.run(population_size=1, generations=0, session=session)
        self.assertEqual(result[0].id, 2)
        self.assertEqual(agent.attempts[0]["raw_response"], "bad")
        self.assertEqual(len(session.history), 2)

    async def test_explicit_rejections_and_unexpected_evaluator_errors(self):
        async def evaluate(content):
            if content == "infeasible":
                raise CandidateRejected("wrong interface")
            if content == "bug":
                raise ValueError("broken evaluator configuration")
            return float(content)

        provider = ScriptedProvider(
            Proposal(description="Idea", content=c) for c in ("infeasible", "nan", "1", "bug")
        )
        agent = EoH("Task", provider, evaluate)
        with self.assertRaisesRegex(ValueError, "broken evaluator"):
            await agent.run(population_size=1, generations=1, operators=("M1",))
        self.assertEqual(agent.evaluations, 4)
        self.assertEqual(
            [r["status"] for r in agent.attempts], ["rejected", "rejected", "accepted", "error"]
        )
        self.assertEqual(agent.history[0][0].fitness, 1)

    async def test_small_population_supports_official_parent_sampling(self):
        provider = ScriptedProvider(Proposal(description="Idea", content=str(i)) for i in range(3))

        async def evaluate(content):
            return float(content)

        agent = EoH("Task", provider, evaluate)
        result = await agent.run(population_size=1, generations=1, operators=("E1", "E2"))
        self.assertEqual(result[0].fitness, 2)
        self.assertEqual([r["parents"] for r in agent.attempts[1:]], [(1,) * 5] * 2)

    async def test_generated_schema_rejections_and_evaluation_timeout(self):
        invalid = [
            {"description": " ", "content": "x"},
            {"description": "Idea", "content": " \n "},
            {"description": "Idea", "content": "x", "fitness": 999},
        ]
        provider = ScriptedProvider(
            [
                *(json.dumps(value) for value in invalid),
                Proposal(description="Idea", content="valid"),
                Proposal(description="Idea", content="timeout"),
            ]
        )

        async def evaluate(content):
            if content == "timeout":
                raise TimeoutError("worker deadline exceeded")
            self.assertEqual(content, "valid")
            return 1.0

        agent = EoH("Task", provider, evaluate)
        result = await agent.run(
            population_size=1, init_attempts=4, generations=1, operators=("M1",)
        )
        self.assertEqual(result[0].id, 4)
        self.assertEqual(agent.evaluations, 2)
        self.assertEqual(
            [r["status"] for r in agent.attempts], ["rejected"] * 3 + ["accepted", "rejected"]
        )

    async def test_session_provider_failure_aborts_for_caller_to_resume(self):
        good = Proposal(description="Idea", content="x")
        provider = ScriptedProvider([good, ProviderError("offline"), good])
        session = Session(provider=provider)

        async def evaluate(content):
            return 1.0

        agent = EoH("Task", ScriptedProvider([]), evaluate)
        with self.assertRaisesRegex(ProviderError, "offline"):
            await agent.run(population_size=1, generations=1, operators=("M1",), session=session)
        self.assertEqual(len(agent.attempts), 2)
        self.assertEqual(agent.evaluations, 1)
        self.assertEqual(agent.attempts[-1]["status"], "error")
        self.assertEqual(await session.arun(), good.model_dump_json())

    async def test_provider_error_consumes_initialization_attempt(self):
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
            self.assertIn("First", rendered)
            self.assertIn("implementation", rendered)
