"""Offline algorithm and Slick boundary checks for EoH-S."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts
from slick.providers import ProviderError

from eoh_s import EoHS, EvaluationError, Individual, Proposal, complementary_select, cpi
from tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1] / "eoh_s/prompts"


class EoHSTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        root.start()
        self.addCleanup(root.stop)

    def test_cpi_selection_keeps_specialists_and_breaks_ties_by_average(self):
        pool = [
            Individual(id=i, description="idea", content=str(i), scores=scores)
            for i, scores in enumerate(((4, 4), (0, 10), (10, 0), (3, 5), (9, 9)))
        ]
        selected = complementary_select(pool, 3)
        self.assertEqual([p.id for p in selected], [0, 1, 2])
        self.assertEqual(cpi(selected), 0)
        self.assertEqual(cpi(selected[:2]), 2)
        self.assertEqual([p.id for p in complementary_select(pool, 5)], [0, 1, 2, 3, 4])
        self.assertEqual([p.id for p in pool], list(range(5)))

    async def test_search_budget_snapshot_and_problem_agnostic_content(self):
        for task in ("Design a routing heuristic", "Write instructions for document editing"):
            scores = ((4, 4), (0, 10), (10, 0), (0, 3), (3, 0), (8, 8), (1, 1))
            provider = ScriptedProvider(
                Proposal(description="idea", content=str(i)) for i in range(len(scores))
            )

            async def evaluate(content):
                return scores[int(content)]

            agent = EoHS(task, provider, evaluate)
            with patch(
                "eoh_s.agent.random.Random.random", side_effect=[0.1, 0.9, 0.5, 0.1, 0.9, 0.5]
            ):
                result = await agent.run(population_size=3, max_evaluations=7)
            self.assertEqual(agent.evaluations, 7)
            self.assertEqual(len(provider.calls), 7)
            self.assertEqual(len(agent.history), 3)
            self.assertEqual(cpi(result), 0)
            self.assertEqual(agent.attempts[3]["parents"], (2, 3))
            self.assertEqual([r["operation"] for r in agent.attempts[3:]], ["CS", "LS", "CS", "LS"])
            for record in agent.attempts[3:]:
                self.assertTrue(
                    set(record["parents"])
                    <= {item.id for item in agent.history[record["generation"] - 1]}
                )
            self.assertTrue(all(task in context for context in provider.calls))

    async def test_failures_are_bounded_logged_and_do_not_replace_incumbents(self):
        provider = ScriptedProvider(
            [
                "not json",
                Proposal(description="idea", content="a"),
                Proposal(description="idea", content="b"),
                Proposal(description="idea", content="nan"),
                Proposal(description="idea", content="short"),
                Proposal(description="idea", content="rejected"),
                Proposal(description="idea", content="timeout"),
                ProviderError("offline"),
            ]
        )

        async def evaluate(content):
            if content == "rejected":
                raise EvaluationError("invalid candidate")
            if content == "timeout":
                raise TimeoutError("evaluation deadline")
            return {"a": (1, 3), "b": (3, 1), "nan": (float("nan"), 0), "short": (0,)}[content]

        agent = EoHS("task", provider, evaluate)
        result = await agent.run(population_size=2, max_evaluations=20, max_attempts=8)
        self.assertEqual([p.content for p in result], ["a", "b"])
        self.assertEqual(agent.evaluations, 6)
        self.assertEqual(len(agent.attempts), 8)
        self.assertEqual(agent.attempts[0]["raw_response"], "not json")
        self.assertTrue(all("error" in agent.attempts[i] for i in (0, 3, 4, 5, 6, 7)))

    async def test_initialization_exhaustion_and_unexpected_errors(self):
        async def evaluate(content):
            raise RuntimeError("evaluator bug")

        agent = EoHS("task", ScriptedProvider(["bad"] * 6), evaluate)
        with self.assertRaisesRegex(RuntimeError, "initialize"):
            await agent.run(population_size=2, max_evaluations=10)
        self.assertEqual(len(agent.attempts), 6)
        agent = EoHS(
            "task", ScriptedProvider([Proposal(description="idea", content="x")]), evaluate
        )
        with self.assertRaisesRegex(RuntimeError, "evaluator bug"):
            await agent.run(population_size=2)
        self.assertIn("evaluator bug", agent.attempts[0]["error"])

    async def test_local_search_sorts_by_mean_before_rank_weighting(self):
        async def evaluate(content):
            return (1, 2)

        agent = EoHS("task", ScriptedProvider([]), evaluate)
        pool = [
            Individual(id=i, description="idea", content=str(i), scores=s)
            for i, s in enumerate(((2, 2), (0, 10), (3, 3)))
        ]
        import random

        agent.rng = random.Random(0)
        with patch.object(agent.rng, "choices", return_value=[pool[0]]) as choose:
            agent._select_local_parent(pool)
        args, kwargs = choose.call_args
        self.assertEqual([p.id for p in args[0]], [0, 2, 1])
        self.assertEqual(kwargs["weights"], [1 / 3, 1 / 4, 1 / 5])

    async def test_templates_and_generated_contract_from_another_directory(self):
        async def evaluate(content):
            return (1, 2)

        provider = ScriptedProvider(
            [
                '{"description":"idea","content":"   "}',
                '{"description":"idea","content":"  exact bytes\\n"}',
            ]
        )
        agent = EoHS("task and interface", provider, evaluate)
        parent = Individual(id=1, description="idea", content="parent", scores=(1, 2))
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for method, args in (
                    (EoHS.initialize, ()),
                    (EoHS.complementary_search, ((parent, parent),)),
                    (EoHS.local_search, (parent,)),
                ):
                    rendered = await method.render(agent, *args)
                    self.assertIn("task and interface", rendered)
                    self.assertIn('"required"', rendered)
                    self.assertEqual(rendered.count("# Output Format"), 1)
                with self.assertRaises(ValueError):
                    await agent.initialize(provider=provider)
                proposal = await agent.initialize(provider=provider)
                self.assertEqual(proposal.content, "  exact bytes\n")
            finally:
                os.chdir(original)
        for template in ROOT.glob("*.j2"):
            tree = Environment().parse(template.read_text())
            self.assertEqual(list(tree.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
