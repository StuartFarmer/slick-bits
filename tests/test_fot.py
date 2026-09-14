"""Check FoT decisions through real Slick rendering and scripted generations."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from fot import Candidate, Evaluation, ForestOfThought
from tests.providers import ScriptedProvider


def candidate(content, answer=None):
    return {"content": content, "answer": answer}


def proposals(*items):
    return json.dumps({"candidates": [candidate(item) for item in items]})


class FoTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "fot/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_beam_correction_activation_and_retrieval(self):
        async def retrieve(task):
            return "Relevant prior example"

        async def evaluate(item):
            scores = {"weak": 0.2, "repaired": 0.9, "other": 0.6, "leaf": 0.8}
            return Evaluation(score=scores.get(item.content, 0.9), confidence=0.9)

        # Override confidence only for the candidate that needs correction.
        async def assess(item):
            value = await evaluate(item)
            return value.model_copy(update={"confidence": 0.2}) if item.content == "weak" else value

        provider = ScriptedProvider([
            proposals(),
            proposals("weak", "other"),
            json.dumps(candidate("repaired")),
            proposals("leaf"),
            json.dumps(candidate("finished", "plan")),
        ])
        agent = ForestOfThought("Plan a project", provider, assess, retrieve=retrieve)
        result = await agent.run(trees=2, depth=2, breadth=1)
        self.assertEqual(result.answer, "plan")
        self.assertEqual([tree.active for tree in result.trees], [False, True])
        self.assertEqual([c.content for c in agent.history[1]["frontier"]], ["repaired"])
        self.assertIn("Relevant prior example", provider.calls[1])
        self.assertIn("repaired", provider.calls[3])
        self.assertEqual(len(agent.corrections), 1)
        self.assertTrue(agent.corrections[0]["accepted"])

    async def test_rule_correction_verification_stops_before_siblings_and_trees(self):
        async def evaluate(item):
            return Evaluation(score=0.9, confidence=0.2 if item.answer == "bad" else 0.9)

        async def correct(item, feedback):
            return Candidate(content="checked artifact", answer="good")

        async def verify(item):
            return item.answer == "good"

        provider = ScriptedProvider([json.dumps({"candidates": [
            candidate("broken", "bad"), candidate("unneeded", "other")
        ]})])
        agent = ForestOfThought("Any task", provider, evaluate, correct=correct, verify=verify)
        result = await agent.run(trees=8)
        self.assertEqual(result.answer, "good")
        self.assertEqual(result.decision, "verified")
        self.assertEqual(len(result.trees), 1)
        self.assertEqual(result.calls, 1)

    async def test_majority_is_locked_against_planned_tree_count(self):
        responses = [json.dumps(candidate("solution " + a, a)) for a in ("A", "B", "A", "A")]
        agent = ForestOfThought("Task", ScriptedProvider(responses), self.evaluate)
        result = await agent.run(trees=5, depth=0)
        self.assertEqual(result.answer, "A")
        self.assertEqual(len(result.trees), 4)
        self.assertEqual(result.decision, "consensus")

    async def test_majority_vs_official_plurality_and_expert_choice(self):
        answers = ("A", "A", "B", "C")
        responses = [json.dumps(candidate("evidence " + a, a)) for a in answers]
        agent = ForestOfThought("Task", ScriptedProvider(responses + ['{"choice": 3}']), self.evaluate)
        result = await agent.run(trees=4, depth=0)
        self.assertEqual(result.answer, "B")
        self.assertEqual(result.decision, "expert")
        self.assertIn("evidence C", agent.calls[-1]["prompt"])
        agent = ForestOfThought("Task", ScriptedProvider(responses), self.evaluate)
        result = await agent.run(trees=4, depth=0, consensus="plurality")
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.calls, 4)

    async def test_rejection_empty_forest_and_answer_equivalence(self):
        async def invalid(item):
            return Evaluation(score=0.8, confidence=0.9, valid=False)

        agent = ForestOfThought("Task", ScriptedProvider([proposals("invalid")]), invalid)
        result = await agent.run(trees=1)
        self.assertIsNone(result.answer)
        self.assertEqual(result.decision, "no_solution")
        responses = [json.dumps(candidate("solution", a)) for a in ("YES", "yes")]
        agent = ForestOfThought("Task", ScriptedProvider(responses), self.evaluate, answer_key=str.casefold)
        result = await agent.run(trees=2, depth=0)
        self.assertEqual(result.answer, "YES")
        self.assertEqual(result.decision, "consensus")

    async def test_mctsr_resampling_refinement_and_selection(self):
        provider = ScriptedProvider([
            json.dumps(candidate("initial", "A")),
            json.dumps(candidate("better", "B")),
            json.dumps(candidate("best", "C")),
        ])

        async def evaluate(item):
            return Evaluation(score={"initial": 0.2, "better": 0.7, "best": 0.9}[item.content], confidence=0.9)

        agent = ForestOfThought("Design a procedure", provider, evaluate)
        result = await agent.run(trees=1, search="mctsr", rollouts=2, exploration=0)
        self.assertEqual(result.answer, "C")
        self.assertEqual(len(agent.nodes[0]), 3)
        self.assertEqual([node.parent for node in agent.nodes[0]], [None, 0, 1])
        self.assertEqual([len(node.rewards) for node in agent.nodes[0]], [2, 2, 1])
        self.assertIn("better", provider.calls[-1])

    async def test_budget_does_not_activate_incomplete_tree_or_guess_expert_answer(self):
        agent = ForestOfThought("Task", ScriptedProvider([proposals("partial")]), self.evaluate)
        result = await agent.run(trees=2, depth=2, max_calls=1)
        self.assertIsNone(result.answer)
        self.assertTrue(result.budget_exhausted)
        self.assertFalse(result.trees[0].active)
        provider = ScriptedProvider([json.dumps(candidate("one", "A")), json.dumps(candidate("two", "B"))])
        result = await ForestOfThought("Task", provider, self.evaluate).run(trees=2, depth=0, max_calls=2)
        self.assertIsNone(result.answer)
        self.assertEqual(result.decision, "budget_exhausted")
        self.assertTrue(all(tree.active for tree in result.trees))

    async def test_generated_errors_keep_raw_response_and_propagate(self):
        for response in ("not json", proposals(" "), proposals("A", "B", "C")):
            agent = ForestOfThought("Task", ScriptedProvider([response]), self.evaluate)
            with self.subTest(response=response), self.assertRaises(ValueError):
                await agent.run(trees=1, candidates=2)
            self.assertEqual(agent.calls[-1]["response"], response)
            self.assertIn("error", agent.calls[-1])
        agent = ForestOfThought("Task", ScriptedProvider([OSError("transport")]))
        with self.assertRaisesRegex(OSError, "transport"):
            await agent.run()
        self.assertIn("error", agent.calls[-1])

    async def test_all_prompt_operations_and_confidence_guard(self):
        evaluation = {"score": 0.4, "confidence": 0.2, "valid": True, "feedback": "Check assumptions"}
        worse = dict(evaluation, confidence=0.1)
        provider = ScriptedProvider([
            json.dumps(candidate("initial", "A")), json.dumps(evaluation),
            json.dumps(candidate("worse", "B")), json.dumps(worse),
        ])
        agent = ForestOfThought("Task", provider)
        result = await agent.run(trees=1, depth=0)
        self.assertEqual(result.answer, "A")
        self.assertFalse(agent.corrections[0]["accepted"])
        previous = Path.cwd()
        try:
            os.chdir(self.templates)
            for operation, args in (
                (ForestOfThought.propose, ({}, 2)),
                (ForestOfThought.initialize, ()),
                (ForestOfThought.assess, (Candidate(content="draft"),)),
                (ForestOfThought.correct_candidate, (Candidate(content="draft"), "feedback")),
                (ForestOfThought.refine, (Candidate(content="draft", answer="A"), "feedback")),
                (ForestOfThought.finish, ({},)),
                (ForestOfThought.choose, ([Candidate(content="draft", answer="A")],)),
            ):
                rendered = await operation.render(agent, *args)
                self.assertIn("Task", rendered)
                self.assertIn("JSON", rendered)
            for template in self.templates.glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)

    @staticmethod
    async def evaluate(item):
        return Evaluation(score=0.9, confidence=0.9)


if __name__ == "__main__":
    unittest.main()
