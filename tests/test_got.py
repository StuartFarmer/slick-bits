"""Check GoT graph decisions through the installed Slick prompt boundary."""

import os
import unittest
from graphlib import CycleError
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from got import CallBudgetExceeded, GraphOfThoughts, Operation, Thought, Validation
from tests.providers import ScriptedProvider


class GoTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "got/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_diamond_aggregation_keeps_incumbent_and_counts_unique_ancestors(self):
        async def evaluate(thought, graph):
            return {"left": 3, "right": 2, "merged": 1}[thought.content]

        # Deliberately not in topological order; a shared ancestor is counted once.
        plan = {
            "best": Operation("keep_best", ("scores", "merge_score")),
            "left": Operation("generate", instruction="Explore left"),
            "right": Operation("generate", instruction="Explore right"),
            "scores": Operation("score", ("left", "right")),
            "merge": Operation("aggregate", ("scores", "left")),
            "merge_score": Operation("score", ("merge",)),
        }
        agent = GraphOfThoughts("Any task", ScriptedProvider(["left", "right", "merged"]), evaluate)
        result = await agent.run("source", plan=plan)
        self.assertEqual(result.final[0].content, "left")
        merged = result.outputs["merge"][0]
        self.assertEqual(merged.parents, (0, 1, 2))
        self.assertEqual(result.volume(merged.id), 2)
        self.assertEqual(result.outputs["scores"][0].score, 3)
        self.assertIsNone(result.outputs["left"][0].score)
        self.assertEqual(result.calls, 3)

    async def test_default_plan_retains_best_across_aggregation_and_refinement(self):
        responses = ["A", "B", "C", "D", "E", "merge1", "merge2", "merge3", "bad1", "bad2"]
        scores = dict(zip(responses, [1, 5, 3, 2, 4, 0, 6, 2, -1, 0]))

        async def evaluate(thought, graph):
            return scores[thought.content]

        agent = GraphOfThoughts("Design anything", ScriptedProvider(responses), evaluate)
        result = await agent.run("requirements")
        self.assertEqual([t.content for t in result.final], ["merge2"])
        self.assertEqual([t.content for t in result.outputs["selected"]], ["B", "E", "C"])
        self.assertEqual(result.calls, 10)
        self.assertEqual(len(agent.assessments), 10)
        self.assertEqual(result.volume(result.final[0].id), 3)

    async def test_decompose_route_solve_and_merge_unrelated_text(self):
        plan = {
            "split": Operation("decompose", count=2, thought_kind="part"),
            "part1": Operation("select", ("split",), selector=lambda ts: ts[:1]),
            "part2": Operation("select", ("split",), selector=lambda ts: ts[1:]),
            "solve1": Operation("generate", ("part1",)),
            "solve2": Operation("generate", ("part2",)),
            "merge": Operation("aggregate", ("solve1", "solve2")),
        }
        agent = GraphOfThoughts(
            "Write a report",
            ScriptedProvider(['{"thoughts": ["topic1", "topic2"]}', "p1", "p2", "report"]),
        )
        result = await agent.run("brief", plan=plan)
        self.assertEqual(result.final[0].content, "report")
        self.assertEqual(result.volume(result.final[0].id), 4)
        self.assertEqual(result.outputs["split"][0].kind, "part")
        self.assertNotIn("topic2", agent.calls[1]["prompt"])

    async def test_validation_refines_only_invalid_and_checks_last_attempt(self):
        provider = ScriptedProvider(
            [
                "draft",
                '{"valid": false, "feedback": "missing evidence"}',
                "revision",
                '{"valid": false, "feedback": "still missing"}',
            ]
        )
        plan = {
            "draft": Operation("generate"),
            "repair": Operation("validate_and_improve", ("draft",), count=1),
            "valid": Operation("keep_valid", ("repair",)),
            "no_restart": Operation("generate", ("valid",)),
        }
        agent = GraphOfThoughts("Task", provider)
        result = await agent.run("input", plan=plan)
        self.assertEqual(result.final, ())
        self.assertFalse(result.outputs["repair"][0].valid)
        self.assertEqual(result.volume(result.outputs["repair"][0].id), 1)
        self.assertIn("missing evidence", agent.calls[2]["prompt"])
        self.assertEqual(result.calls, 4)

        async def validate(thought, graph):
            return Validation(valid=True, feedback="accepted")

        agent = GraphOfThoughts("Task", ScriptedProvider(["draft", "next"]), validate=validate)
        result = await agent.run("input", plan=plan)
        self.assertEqual(result.calls, 2)
        self.assertTrue(result.outputs["repair"][0].valid)

    async def test_score_sampling_minimization_stable_ties_and_branch_isolation(self):
        plan = {
            "draft": Operation("generate", count=2),
            "scores": Operation("score", ("draft",), count=2),
            "best": Operation("keep_best", ("scores",)),
        }
        provider = ScriptedProvider(
            ["A", "B", '{"score": 1}', '{"score": 3}', '{"score": 2}', '{"score": 2}']
        )
        result = await GraphOfThoughts("Task", provider, higher_is_better=False).run(
            "input", plan=plan
        )
        self.assertEqual(result.final[0].content, "A")
        self.assertEqual(result.final[0].score, 2)
        self.assertIsNone(result.outputs["draft"][0].score)

    async def test_failures_preserve_raw_records_and_respect_call_budget(self):
        cases = [
            (Operation("generate"), " "),
            (Operation("decompose", count=2), '{"thoughts": ["one"]}'),
            (Operation("decompose"), '{"thoughts": [" "]}'),
            (Operation("decompose"), "not json"),
            (Operation("score"), '{"score": NaN}'),
            (Operation("validate_and_improve"), '{"valid": "yes", "feedback": ""}'),
        ]
        for operation, response in cases:
            agent = GraphOfThoughts("Task", ScriptedProvider([response]))
            with self.subTest(response=response), self.assertRaises(ValueError):
                await agent.run("input", plan={"op": operation})
            self.assertEqual(agent.calls[0]["response"], response)
            self.assertIn("error", agent.calls[0])
            self.assertEqual(len(agent.calls), 1)

        agent = GraphOfThoughts("Task", ScriptedProvider(["first", "second"]))
        with self.assertRaises(CallBudgetExceeded):
            await agent.run("input", plan={"g": Operation("generate", count=2)}, max_calls=1)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(agent.thoughts[1].content, "first")
        result = await agent.run("new input", plan={"g": Operation("generate")})
        self.assertEqual(result.calls, 1)
        self.assertEqual(result.final[0].id, 1)

        async def evaluate(thought, graph):
            return float("inf")

        agent = GraphOfThoughts("Task", ScriptedProvider([]), evaluate)
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run("input", plan={"s": Operation("score")})
        self.assertIn("error", agent.assessments[0])
        agent = GraphOfThoughts("Task", ScriptedProvider([OSError("transport")]))
        with self.assertRaisesRegex(OSError, "transport"):
            await agent.run("input")
        self.assertIn("error", agent.calls[0])

    async def test_schedule_cycles_and_unscored_selection_fail_explicitly(self):
        agent = GraphOfThoughts("Task", ScriptedProvider([]))
        with self.assertRaises(CycleError):
            await agent.run("input", plan={"a": Operation("generate", ("a",))})
        self.assertEqual(agent.calls, [])
        with self.assertRaisesRegex(ValueError, "scored"):
            await agent.run("input", plan={"a": Operation("keep_best")})

    async def test_independent_roots_and_callbacks_preserve_branch_annotations(self):
        observed = []

        async def evaluate(thought, graph):
            observed.append((thought.score, graph[thought.id].score))
            return 9

        async def validate(thought, graph):
            observed.append((thought.score, graph[thought.id].score))
            return Validation(valid=True, feedback="accepted")

        agent = GraphOfThoughts("Task", ScriptedProvider([]), evaluate, validate=validate)
        result = await agent.run(
            "input",
            plan={
                "root_score": Operation("score"),
                "root_validate": Operation("validate_and_improve"),
            },
        )
        self.assertIsNone(result.outputs["root_validate"][0].score)
        self.assertEqual(observed, [(None, None), (None, None)])

        observed.clear()
        agent.provider = ScriptedProvider(["draft"])
        result = await agent.run(
            "input",
            plan={
                "draft": Operation("generate"),
                "score1": Operation("score", ("draft",)),
                "score2": Operation("score", ("draft",)),
            },
        )
        self.assertEqual(observed, [(None, None), (None, None)])
        self.assertIsNone(result.outputs["draft"][0].score)

    async def test_templates_render_from_other_directory_without_control_logic(self):
        agent = GraphOfThoughts("Task", ScriptedProvider([]))
        agent.input = "input"
        thought = Thought(0, "source")
        previous = Path.cwd()
        try:
            os.chdir(self.templates)
            for operation, args in (
                (GraphOfThoughts.generate, (thought, "instruction")),
                (GraphOfThoughts.decompose, (thought, 2, "instruction")),
                (GraphOfThoughts.aggregate, ((thought,), "instruction")),
                (GraphOfThoughts.improve, (thought, "instruction")),
                (GraphOfThoughts.score, (thought, "instruction")),
                (GraphOfThoughts.validate, (thought, "instruction")),
            ):
                rendered = await operation.render(agent, *args)
                self.assertIn("Task", rendered)
                self.assertIn("instruction", rendered)
            for template in self.templates.glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
