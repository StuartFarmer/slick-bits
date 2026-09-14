"""Offline checks for LLM-GA's search, rejection feedback, and Slick boundaries."""

import random
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts
from slick.providers import ProviderError

from llm_ga import LLMGA, CandidateRejected, Proposal
from tests.providers import ScriptedProvider


def proposal(content):
    return Proposal(description=f"Idea {content}", content=str(content))


async def evaluate(content):
    if content == "broken":
        raise CandidateRejected("candidate could not execute")
    return float(content)


class LLMGATests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "llm_ga/prompts"
        root_patch = patch.object(prompts, "TEMPLATE_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    async def test_search_uses_all_operators_and_updated_batch_population(self):
        provider = ScriptedProvider(proposal(i) for i in range(20, 2, -1))
        agent = LLMGA("Optimize a route", provider, evaluate, template="route template")
        result = await agent.run(population_size=2, generations=2, seed=3)
        self.assertEqual([p.fitness for p in result], [3, 4])
        self.assertEqual(len(provider.calls), 18)
        self.assertEqual(agent.evaluations, 18)
        self.assertEqual([p[0].fitness for p in agent.history], [19, 11, 3])
        self.assertEqual(
            [r["operation"] for r in agent.attempts[2:10]],
            ["E1", "E1", "E2", "E2", "M1", "M1", "M2", "M2"],
        )
        rng = random.Random(3)
        population_ids = [2, 1]
        for start in range(2, 18, 2):
            for record in agent.attempts[start : start + 2]:
                count = 2 if record["operation"].startswith("E") else 1
                expected = rng.choices(population_ids, weights=[1 / 3, 1 / 4], k=count)
                self.assertEqual(record["parents"], tuple(expected))
            population_ids = [start + 2, start + 1]
        self.assertTrue(all("route template" in call for call in provider.calls))

    async def test_blacklists_are_operator_local_and_reject_equal_worst(self):
        provider = ScriptedProvider(
            map(proposal, [1, 3, 3, 8, 2, 2, "broken", 0, -1, -2, 9, -3, -4, -5, -6, -7, -8, -9])
        )
        agent = LLMGA("Choose a schedule", provider, evaluate)
        result = await agent.run(population_size=2, generations=2)
        self.assertEqual(result[0].fitness, -9)
        self.assertEqual(
            [row["proposal"]["content"] for row in agent.blacklists["E1"]], ["3", "8", "9"]
        )
        self.assertEqual(agent.blacklists["M1"][0]["proposal"]["content"], "broken")
        self.assertIn("Idea 3", provider.calls[3])
        self.assertIn("Idea 8", provider.calls[10])
        self.assertNotIn("Idea 8", provider.calls[4])
        self.assertIn("candidate could not execute", provider.calls[7])
        self.assertEqual(len(agent.history[1]), 2)

    async def test_parse_and_evaluation_failures_keep_raw_and_preserve_incumbent(self):
        responses = [proposal(1), "bad JSON", proposal("nan"), proposal("broken"), proposal(1)]
        agent = LLMGA("Any task", ScriptedProvider(responses), evaluate)
        result = await agent.run(population_size=1, generations=1)
        self.assertEqual(result[0].id, 1)
        self.assertEqual(agent.evaluations, 4)
        self.assertEqual(agent.attempts[1]["raw_response"], "bad JSON")
        self.assertEqual([len(rows) for rows in agent.blacklists.values()], [1, 1, 1, 1])
        self.assertTrue(all(row["status"] == "rejected" for row in agent.attempts[1:]))

    async def test_initialization_is_bounded_and_keeps_equal_fitness_candidates(self):
        agent = LLMGA("Task", ScriptedProvider(["bad"] * 3), evaluate)
        with self.assertRaisesRegex(RuntimeError, "initializ"):
            await agent.run(population_size=1, generations=0)
        self.assertEqual(len(agent.attempts), 3)
        self.assertEqual(agent.evaluations, 0)
        agent = LLMGA("Task", ScriptedProvider([proposal(1), proposal("1.0")]), evaluate)
        result = await agent.run(population_size=2, generations=0)
        self.assertEqual([p.id for p in result], [1, 2])

    async def test_unexpected_errors_and_transport_failures_abort(self):
        for failure in (ValueError("caller bug"), RuntimeError("worker unavailable")):

            async def fail(content):
                raise failure

            agent = LLMGA("Task", ScriptedProvider([proposal(1)]), fail)
            with self.assertRaises(type(failure)):
                await agent.run(population_size=1)
            self.assertIn(str(failure), agent.attempts[0]["error"])
        for failure in (ProviderError("offline"), TimeoutError("provider timeout")):
            agent = LLMGA("Task", ScriptedProvider([failure]), evaluate)
            with self.assertRaises(type(failure)):
                await agent.run(population_size=1)
            self.assertEqual(agent.evaluations, 0)

    async def test_evaluator_timeout_is_candidate_rejection(self):
        async def timed_evaluation(content):
            if content == "slow":
                raise TimeoutError("worker deadline")
            return float(content)

        agent = LLMGA(
            "Task", ScriptedProvider(map(proposal, [1, "slow", 1, 1, 1])), timed_evaluation
        )
        result = await agent.run(population_size=1, generations=1)
        self.assertEqual(result[0].id, 1)
        self.assertIn("worker deadline", agent.blacklists["E1"][0]["error"])

    async def test_rerun_resets_state_and_seed(self):
        provider = ScriptedProvider([proposal(i) for i in range(10, 0, -1)] * 2)
        agent = LLMGA("Task", provider, evaluate)
        first = await agent.run(population_size=2, generations=1, seed=4)
        previous = agent.attempts.copy()
        second = await agent.run(population_size=2, generations=1, seed=4)
        self.assertEqual(first, second)
        self.assertEqual(previous, agent.attempts)
        self.assertEqual(len(agent.history), 2)
        self.assertEqual(agent.evaluations, 10)

    async def test_generated_contract_preserves_content_and_rejects_blank(self):
        content = "def solve():\n    return 1\n"
        provider = ScriptedProvider([proposal(content), '{"description":"idea","content":"  "}'])
        agent = LLMGA("Task", provider, evaluate)
        self.assertEqual((await agent.initialize(provider=provider)).content, content)
        with self.assertRaises(ValueError):
            await agent.initialize(provider=provider)

    async def test_every_template_renders_with_explicit_owner_without_branches(self):
        agent = LLMGA(
            "A different task",
            ScriptedProvider([]),
            evaluate,
            template="SKELETON",
            information="INTERFACE",
            requirements="CONSTRAINTS",
        )
        for method in (
            LLMGA.initialize,
            LLMGA.explore_diverse,
            LLMGA.explore_shared,
            LLMGA.modify_structure,
            LLMGA.tune_settings,
        ):
            args = () if method is LLMGA.initialize else ([], [])
            rendered = await method.render(agent, *args)
            for text in (
                "A different task",
                "SKELETON",
                "INTERFACE",
                "CONSTRAINTS",
                "Lower fitness",
                '"description"',
                '"content"',
                "# Output Format",
            ):
                self.assertIn(text, rendered)
            if args:
                self.assertIn("Thinking", rendered)
                self.assertIn("blacklist", rendered)
        for path in self.root.glob("*.j2"):
            parsed = Environment().parse(path.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
