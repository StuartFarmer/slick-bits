"""Offline MEOH checks with measured objectives and scripted generation."""

import math
import random
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts
from slick.providers import ProviderError

from meoh import MEOH, CandidateRejected, Individual, Proposal, python_similarity
from meoh.agent import dominance_scores, dominates
from tests.providers import ScriptedProvider


class MEOHTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "meoh/prompts"
        patcher = patch.object(prompts, "TEMPLATE_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dominance_penalties_are_directional_and_strict(self):
        population = [
            Individual(id=1, description="a", content="a", objectives=(1, 3)),
            Individual(id=2, description="b", content="b", objectives=(2, 2)),
            Individual(id=3, description="c", content="c", objectives=(3, 4)),
            Individual(id=4, description="d", content="d", objectives=(1, 3)),
        ]
        pairs = []

        def similarity(a, b):
            pairs.append((a, b))
            return {("a", "c"): 0.8, ("b", "c"): 0.2, ("d", "c"): 0.5}[a, b]

        self.assertEqual(dominance_scores(population, (False, False), similarity), [0, 0, -1.5, 0])
        self.assertEqual(pairs, [("a", "c"), ("b", "c"), ("d", "c")])
        self.assertTrue(dominates((3, 1, 7), (2, 2, 6), (True, False, True)))
        self.assertFalse(dominates((1, 1), (1, 1), (False, False)))
        self.assertFalse(dominates((0, 2), (1, 1), (False, False)))
        for invalid in (float("nan"), -0.1, 1.1):
            with self.assertRaisesRegex(ValueError, "similarity"):
                dominance_scores(population, (False, False), lambda a, b: invalid)

    async def test_all_operators_archive_budget_and_growing_parent_pool(self):
        for task in ("Design a scheduling heuristic", "Write a concise invitation"):
            provider = ScriptedProvider(
                Proposal(description="Idea", content=str(i)) for i in range(8)
            )

            async def evaluate(content):
                return (float(content), -float(content))

            agent = MEOH(
                task, provider, evaluate, maximize=(False, False), similarity=lambda a, b: 1
            )
            result = await agent.run(population_size=2, generations=3, parents=1, seed=0)
            self.assertEqual(len(result), 8)
            self.assertEqual(len(agent.population), 2)
            self.assertEqual(len(provider.calls), 8)
            self.assertEqual(agent.evaluations, 8)
            self.assertEqual(
                [row["operation"] for row in agent.attempts],
                ["INIT", "INIT", "E1", "E2", "M1", "M2", "M3", "E1"],
            )
            # The second offspring may use the first before generation truncation.
            self.assertEqual(agent.attempts[3]["parents"], (3,))
            self.assertEqual([len(pop) for pop in agent.history], [2, 2, 2, 2])
            self.assertTrue(all(task in request for request in provider.calls))
            self.assertTrue(all('"description"' in request for request in provider.calls))
            self.assertEqual(
                [row["objectives"] for row in agent.attempts], [(i, -i) for i in range(8)]
            )

    async def test_softmax_selection_and_truncation(self):
        async def evaluate(content):
            return (0, 0)

        agent = MEOH(
            "Task",
            ScriptedProvider([]),
            evaluate,
            maximize=(False, False),
            similarity=lambda a, b: 1,
        )
        population = [
            Individual(id=1, description="a", content="a", objectives=(0, 0)),
            Individual(id=2, description="b", content="b", objectives=(1, 1)),
        ]
        agent.rng = random.Random(0)
        selected = agent._select_parents(population, 10000)
        frequency = sum(item.id == 1 for item in selected) / len(selected)
        self.assertAlmostEqual(frequency, 1 / (1 + math.exp(-1)), delta=0.015)
        self.assertEqual(agent._select_survivors(population[::-1], 1)[0].id, 1)

    async def test_rejections_keep_raw_output_and_consume_attempts(self):
        provider = ScriptedProvider(
            [
                "not json",
                Proposal(description="Valid", content="good"),
                Proposal(description="Invalid", content="nan"),
                Proposal(description="Invalid", content="wrong dimensions"),
                Proposal(description="Invalid", content="infeasible"),
                Proposal(description="Invalid", content="timeout"),
                '{"description":"Blank","content":"  "}',
                ProviderError("offline"),
            ]
        )

        async def evaluate(content):
            if content == "nan":
                return (float("nan"), 1)
            if content == "wrong dimensions":
                return (1,)
            if content == "infeasible":
                raise CandidateRejected("constraint violation")
            if content == "timeout":
                raise TimeoutError("deadline")
            return (1, 1)

        agent = MEOH("Task", provider, evaluate, maximize=(False, False), similarity=lambda a, b: 1)
        result = await agent.run(population_size=1, generations=6)
        self.assertEqual([item.content for item in result], ["good"])
        self.assertEqual(agent.evaluations, 5)
        self.assertEqual(len(agent.attempts), 8)
        self.assertEqual(agent.attempts[0]["raw_response"], "not json")
        self.assertEqual(sum(row["status"] == "rejected" for row in agent.attempts), 7)

    async def test_initialization_is_bounded_and_evaluator_bugs_propagate(self):
        async def evaluate(content):
            raise ValueError("caller bug")

        agent = MEOH("Task", ScriptedProvider(["bad"] * 3), evaluate, maximize=(False, False))
        with self.assertRaisesRegex(RuntimeError, "initialize"):
            await agent.run(population_size=1)
        self.assertEqual(len(agent.attempts), 3)
        self.assertEqual(agent.evaluations, 0)
        agent = MEOH(
            "Task",
            ScriptedProvider([Proposal(description="Idea", content="x")]),
            evaluate,
            maximize=(False, False),
        )
        with self.assertRaisesRegex(ValueError, "caller bug"):
            await agent.run(population_size=1)
        self.assertEqual(agent.attempts[0]["status"], "error")

    async def test_archive_removes_dominated_candidates_and_keeps_tied_artifacts(self):
        provider = ScriptedProvider(Proposal(description="Idea", content=c) for c in "abcd")

        async def evaluate(content):
            return {"a": (3, 3), "b": (2, 2), "c": (2, 2), "d": (1, 4)}[content]

        agent = MEOH("Task", provider, evaluate, maximize=(False, False), similarity=lambda a, b: 1)
        result = await agent.run(population_size=1, generations=3)
        self.assertEqual([item.content for item in result], ["b", "c", "d"])

    async def test_templates_render_with_explicit_owner(self):
        async def evaluate(content):
            return (1, 1)

        agent = MEOH("Task data", ScriptedProvider([]), evaluate, maximize=(True, False))
        parent = Individual(id=1, description="Idea", content="artifact", objectives=(1, 2))
        for method in (
            "initialize",
            "explore_diverse",
            "explore_shared",
            "modify_structure",
            "tune_settings",
            "simplify",
        ):
            args = () if method == "initialize" else ([parent],)
            rendered = await getattr(MEOH, method).render(agent, *args)
            self.assertIn("Task data", rendered)
            self.assertIn("maximize", rendered)
            self.assertIn("minimize", rendered)
            self.assertIn('"description"', rendered)
        for template in self.root.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])

    @unittest.skipUnless(find_spec("codebleu"), "install meoh/requirements.txt for AST integration")
    async def test_full_search_with_default_python_similarity(self):
        contents = [f"def heuristic(x):\n    return x + {i}" for i in range(8)]
        scores = {content: (i, i) for i, content in enumerate(contents)}

        async def evaluate(content):
            return scores[content]

        provider = ScriptedProvider(
            Proposal(description="Shift the input", content=content) for content in contents
        )
        agent = MEOH("Return a Python heuristic(x)", provider, evaluate, maximize=(False, False))
        result = await agent.run(population_size=2, generations=3, parents=2)
        self.assertEqual([item.content for item in result], contents[:1])
        self.assertEqual([item.content for item in agent.population], contents[:2])
        self.assertEqual(agent.evaluations, 8)
        self.assertTrue(all(row["status"] == "accepted" for row in agent.attempts))

    @unittest.skipUnless(find_spec("codebleu"), "install meoh/requirements.txt for AST integration")
    def test_python_similarity_uses_official_codebleu_direction(self):
        from codebleu.syntax_match import calc_syntax_match

        reference = "def f(x):\n    return x"
        candidate = "def f(x):\n    return x + 1"
        self.assertEqual(python_similarity(reference, reference), 1)
        self.assertEqual(python_similarity(reference, candidate), 0.2)
        self.assertEqual(
            python_similarity(reference, candidate),
            calc_syntax_match([reference], candidate, "python"),
        )


if __name__ == "__main__":
    unittest.main()
