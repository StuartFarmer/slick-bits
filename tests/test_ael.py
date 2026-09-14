"""Offline checks for generic AEL selection and generation."""

import json
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

    async def test_generated_code_is_preserved_and_blank_content_is_rejected(self):
        code = "\n\ndef solve(items):\n    return sorted(items)\n\n"
        provider = ScriptedProvider(
            [
                json.dumps({"description": "Sort", "content": " \n\t"}),
                json.dumps({"description": "Sort", "content": code}),
            ]
        )
        evaluated = []

        async def evaluate(content):
            evaluated.append(content)
            return 1.0

        agent = AEL("Design a reusable sorting algorithm", provider, evaluate)
        result = await agent.run(population_size=1, parents=1, generations=0)
        self.assertEqual(evaluated, [code])
        self.assertEqual(result[0].content, code)
        self.assertIn("error", agent.attempts[0])

    async def test_seed_algorithms_are_evaluated_then_missing_slots_are_generated(self):
        evaluated = []

        async def evaluate(content):
            evaluated.append(content)
            return float(content)

        provider = ScriptedProvider([Proposal(description="Generated", content="2")])
        agent = AEL("Minimize cost over evaluation instances", provider, evaluate)
        result = await agent.run(
            population_size=2,
            generations=0,
            initial=[
                Proposal(description="Invalid seed", content="nan"),
                Proposal(description="Seed", content="1"),
            ],
        )
        self.assertEqual(evaluated, ["nan", "1", "2"])
        self.assertEqual([item.fitness for item in result], [1, 2])
        self.assertEqual(
            [a["operation"] for a in agent.attempts], ["seed", "seed", "initialization"]
        )
        evaluated.clear()
        result = await agent.run(
            population_size=2,
            generations=0,
            init_attempts=0,
            initial=list(reversed(result)) + [Proposal(description="Unused", content="unused")],
        )
        self.assertEqual(evaluated, ["2", "1"])
        self.assertEqual([item.fitness for item in result], [1, 2])
        self.assertEqual(len(agent.attempts), 2)
        self.assertEqual(len(agent.history), 1)

    async def test_parent_prompts_expose_algorithm_without_measured_fitness(self):
        async def evaluate(content):
            return 98765.4321

        provider = ScriptedProvider(
            Proposal(description="Algorithm", content=str(i)) for i in range(3)
        )
        agent = AEL("Design an algorithm", provider, evaluate)
        await agent.run(population_size=1, parents=1, generations=1, mutation=1)
        self.assertNotIn("98765.4321", provider.calls[1])
        self.assertIn('"content": "0"', provider.calls[1])
        self.assertIn('"content": "1"', provider.calls[2])

    async def test_multiple_offspring_evaluate_only_final_mutations(self):
        evaluated = []

        async def evaluate(content):
            evaluated.append(content)
            return -float(content)

        provider = ScriptedProvider(
            Proposal(description="Algorithm", content=str(i)) for i in range(10)
        )
        agent = AEL("Task", provider, evaluate)
        result = await agent.run(
            population_size=2, parents=2, generations=1, offspring=2, mutation=1
        )
        self.assertEqual(evaluated, ["0", "1", "3", "5", "7", "9"])
        self.assertEqual([item.content for item in result], ["9", "7"])
        self.assertEqual(len(agent.history), 2)

    async def test_failed_mutation_keeps_incumbent_and_records_both_attempts(self):
        evaluated = []

        async def evaluate(content):
            evaluated.append(content)
            return float(content)

        provider = ScriptedProvider(
            [
                Proposal(description="Parent", content="2"),
                Proposal(description="Crossover", content="1"),
                "invalid JSON",
            ]
        )
        agent = AEL("Task", provider, evaluate)
        result = await agent.run(population_size=1, parents=1, generations=1, mutation=1)
        self.assertEqual(evaluated, ["2"])
        self.assertEqual(result[0].content, "2")
        self.assertEqual(agent.attempts[2]["parents"], (2,))
        self.assertIn("error", agent.attempts[2])

    async def test_session_parse_failure_leaves_recovery_to_session_owner(self):
        from pydantic import ValidationError

        async def evaluate(content):
            self.fail("Malformed generated content must not be evaluated")

        provider = ScriptedProvider(["invalid JSON"])
        session = Session(provider=provider)
        agent = AEL("Task", provider, evaluate)
        with self.assertRaises(ValidationError):
            await agent.run(population_size=1, parents=1, session=session)
        self.assertEqual(len(agent.attempts), 1)
        self.assertEqual(session.history[-1]["text"], "invalid JSON")
