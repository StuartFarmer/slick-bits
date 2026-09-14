"""Deterministic RAP search and Slick boundary checks."""

import math
import os
import unittest
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

from jinja2 import Environment, nodes

from rap import RAP, Action, Score, State
from tests.providers import ScriptedProvider


class RAPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "rap/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_backtracks_with_lazy_world_model_and_suffix_returns(self):
        provider = ScriptedProvider(
            [
                Action(content="A"),
                Action(content="B"),
                Score(score=0.9),
                Score(score=0.8),
                State(content="after A"),
                Action(content="bad"),
                Action(content="bad"),
                Score(score=0.1),
                State(content="failed", terminal=True, answer="wrong"),
                State(content="after B"),
                Action(content="good"),
                Action(content="good"),
                Score(score=1),
                State(content="finished", terminal=True, answer="right"),
            ]
        )

        async def evaluate(transition):
            return {"A": 0.9, "B": 0.8, "bad": 0.1, "good": 1}[transition.action]

        agent = RAP("Find a route", provider, evaluate)
        result = await agent.run(
            State(content="start"),
            iterations=2,
            candidates=2,
            depth=2,
            confidence_samples=1,
            aggregate=True,
        )
        self.assertEqual(result.best.actions, ("B", "good"))
        self.assertAlmostEqual(result.best.score, 0.9)
        self.assertEqual(result.answer, "right")
        self.assertEqual(result.answer_weights, {"wrong": 1.0, "right": 1.8})
        a, b = agent.root.children
        self.assertEqual(a.returns, [0.5])
        self.assertEqual(b.returns, [0.9])
        self.assertEqual(b.children[0].returns, [1])
        self.assertEqual(agent.root.returns, [0.5, 0.9])
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(agent.evaluations), 4)
        self.assertEqual(len(provider.calls), 14)
        self.assertIn("after B", provider.calls[-1])
        self.assertNotIn("after A", provider.calls[-1])

    async def test_confidence_deduplication_and_terminal_cache(self):
        provider = ScriptedProvider(
            [
                Action(content="finish"),
                Action(content="finish"),
                Score(score=0.81),
                State(content="first", terminal=True, answer="yes"),
                State(content="second", terminal=True, answer="yes"),
                State(content="third", terminal=True, answer="no"),
            ]
        )
        agent = RAP("Task", provider, state_key=lambda state: state.answer)
        result = await agent.run(iterations=3, candidates=2, confidence_samples=3, aggregate=True)
        child = agent.root.children[0]
        self.assertEqual(child.state.content, "first")
        self.assertAlmostEqual(child.transition.confidence, 2 / 3)
        self.assertAlmostEqual(child.reward, math.sqrt(0.81 * 2 / 3))
        self.assertEqual(child.visits, 3)
        self.assertEqual(len(provider.calls), 6)
        self.assertEqual(result.answer_weights, {"yes": child.reward})
        self.assertEqual(provider.calls[3], provider.calls[4])

    async def test_depth_limit_materializes_last_action_without_extra_expansion(self):
        provider = ScriptedProvider(
            [
                Action(content="continue"),
                Score(score=1),
                State(content="partial"),
            ]
        )
        agent = RAP("Task", provider)
        result = await agent.run(
            iterations=2, candidates=1, depth=1, confidence_samples=1, aggregate=True
        )
        self.assertIsNone(result.best)
        self.assertIsNone(result.answer)
        self.assertEqual(result.partial.reason, "depth_limit")
        self.assertEqual(result.partial.states[-1].content, "partial")
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(agent.root.children[0].children, None)

    async def test_zero_budgets_and_terminal_root(self):
        for settings in ({"iterations": 0}, {"depth": 0}, {"candidates": 0}):
            agent = RAP("Task", ScriptedProvider([]))
            result = await agent.run(**settings)
            self.assertIsNone(result.best)
            self.assertEqual(agent.calls, [])
        agent = RAP("Task", ScriptedProvider([]))
        result = await agent.run(State(content="done", terminal=True, answer="yes"), aggregate=True)
        self.assertEqual(result.best.actions, ())
        self.assertEqual(result.answer, "yes")
        self.assertEqual(result.iterations, 0)

    async def test_generated_failure_and_evaluator_failure_remain_visible(self):
        agent = RAP("Task", ScriptedProvider(['{"content": " "}']))
        with self.assertRaises(ValueError):
            await agent.run(candidates=1)
        self.assertEqual(agent.calls[0]["response"], '{"content": " "}')
        self.assertIn("error", agent.calls[0])

        async def invalid(transition):
            return math.nan

        agent = RAP(
            "Task",
            ScriptedProvider(
                [
                    Action(content="act"),
                    Score(score=1),
                    State(content="new"),
                ]
            ),
            invalid,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run(candidates=1, confidence_samples=1)
        self.assertIn("error", agent.evaluations[0])
        self.assertEqual(agent.root.visits, 0)

    async def test_all_templates_render_from_another_directory(self):
        agent = RAP(
            "TASK TOKEN",
            ScriptedProvider([]),
            actions="ACTION TOKEN",
            world="WORLD TOKEN",
            criteria="CRITERIA TOKEN",
        )
        previous = Path.cwd()
        try:
            os.chdir("/tmp")
            texts = [
                await RAP.propose.render(agent, State(content="STATE TOKEN")),
                await RAP.assess.render(agent, State(content="STATE TOKEN"), "act"),
                await RAP.predict.render(agent, State(content="STATE TOKEN"), "act"),
            ]
        finally:
            os.chdir(previous)
        for text in texts:
            self.assertIn("TASK TOKEN", text)
            self.assertIn("STATE TOKEN", text)
            self.assertIn('"properties"', text)
            self.assertIn("JSON", text)
        for template in self.templates.glob("*.j2"):
            tree = Environment().parse(template.read_text())
            self.assertEqual(list(tree.find_all((nodes.If, nodes.CondExpr))), [])

    async def test_shared_edges_count_once_per_answer_and_backup_is_configurable(self):
        from rap import Node

        root = Node(state=State(content="root"))
        shared = Node(state=State(content="shared"), action="a", parent=root, reward=0.5)
        one = Node(
            state=State(content="one", terminal=True, answer="X"),
            action="b",
            parent=shared,
            reward=0.8,
        )
        two = Node(
            state=State(content="two", terminal=True, answer="X"),
            action="c",
            parent=shared,
            reward=0.6,
        )
        root.children, shared.children = [shared], [one, two]
        agent = RAP("Task", ScriptedProvider([]))
        agent.root = root
        self.assertEqual(agent._aggregate(), {"X": 1.9})
        agent.cum_reward, agent.calc_q = fmean, max
        agent._backpropagate([root, shared, one])
        agent._backpropagate([root, shared, two])
        self.assertEqual(shared.returns, [0.65, 0.55])
        self.assertEqual(agent._uct(shared), 0.65 + math.sqrt(math.log(2) / 2))
        agent.cum_reward, agent.calc_q = sum, fmean
        agent._backpropagate([root, shared, one])
        self.assertAlmostEqual(shared.returns[-1], 1.3)

    async def test_uct_uses_finite_unvisited_prior_and_reset_clears_tree(self):
        from rap import Node

        agent = RAP("Task", ScriptedProvider([]))
        root = Node(state=State(content="root"), returns=[0, 0, 0, 0])
        child = Node(parent=root, fast_reward=0.7)
        self.assertAlmostEqual(agent._uct(child), 0.7 + math.sqrt(math.log(4)))
        agent.calls.append({"old": True})
        await agent.run(iterations=0)
        self.assertEqual(agent.calls, [])
        self.assertEqual(agent.root.visits, 0)

    async def test_prediction_contract_provider_error_and_empty_answers(self):
        for response, error in [
            ('{"content":"state","terminal":false,"answer":"too early"}', ValueError),
            ("not json", ValueError),
            (RuntimeError("offline"), RuntimeError),
            (("ignored", [{"name": "tool"}]), ValueError),
        ]:
            agent = RAP(
                "Task",
                ScriptedProvider(
                    [
                        Action(content="act"),
                        Score(score=0.5),
                        response,
                    ]
                ),
            )
            with self.subTest(response=response), self.assertRaises(error):
                await agent.run(candidates=1, confidence_samples=1)
            self.assertIn("error", agent.calls[-1])
            self.assertEqual(agent.root.visits, 0)

        agent = RAP(
            "Task",
            ScriptedProvider(
                [
                    Action(content="act"),
                    Score(score=0.5),
                    State(content="dead end", terminal=True),
                ]
            ),
        )
        result = await agent.run(candidates=1, confidence_samples=1, aggregate=True)
        self.assertIsNone(result.best)
        self.assertEqual(result.partial.reason, "dead_end")
        self.assertIsNone(result.answer)
        self.assertEqual(result.answer_weights, {})

    async def test_high_reward_dead_end_does_not_replace_a_completed_answer(self):
        async def evaluate(transition):
            return 0.9 if transition.action == "blocked" else 0.8

        agent = RAP(
            "Task",
            ScriptedProvider(
                [
                    Action(content="blocked"),
                    Action(content="solve"),
                    Score(score=0.9),
                    Score(score=0.8),
                    State(content="cannot continue", terminal=True),
                    State(content="solved", terminal=True, answer="answer"),
                ]
            ),
            evaluate,
        )
        result = await agent.run(iterations=3, candidates=2, depth=1, confidence_samples=1)
        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.best.actions, ("solve",))
        self.assertEqual(result.partial.reason, "dead_end")
        self.assertGreater(result.partial.score, result.best.score)


if __name__ == "__main__":
    unittest.main()
