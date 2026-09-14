"""AMA checks use real Slick boundaries and, when installed, official MeTaL."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from jinja2 import Environment, nodes
from pydantic import ValidationError

from ama import AMA, Chain, Example, WeakSupervision, WSConfig
from ama.aggregation import majority_vote, select_dependency
from tests.providers import ScriptedProvider


class AMATests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "ama/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)

    async def test_open_answers_follow_distinct_questions_and_preserve_context(self):
        provider = ScriptedProvider(
            [
                "Q1?",
                "Yes",
                '{"answer":"blue"}',
                "Q2?",
                "red",
                '{"answer":"red"}',
                "Q3 ___",
                "blue",
                '{"answer":"blue"}',
            ]
        )
        agent = AMA("Find the colour", provider)
        result = await agent.run([Example("object", "blue paint")])
        self.assertEqual(result.predictions, ("blue",))
        self.assertEqual(result.votes, (("blue", "red", "blue"),))
        self.assertEqual(result.calls, 9)
        for i, question in enumerate(("Q1?", "Q2?", "Q3 ___")):
            self.assertIn(question, provider.calls[3 * i + 1])
            self.assertIn("blue paint", provider.calls[3 * i + 1])
            self.assertIn("object", provider.calls[3 * i + 1])
        self.assertEqual(result.traces[0].question, "Q1?")
        self.assertEqual(agent.generations, ["Q1?", "Yes", "Q2?", "red", "Q3 ___", "blue"])

    async def test_label_mapping_sees_original_input_and_can_abstain(self):
        provider = ScriptedProvider(
            [
                "Was it hard?",
                "No",
                '{"label":"supported"}',
                "Is it easy?",
                "Unknown",
                '{"label":null}',
            ]
        )
        agent = AMA(
            "Assess the claim",
            provider,
            labels=("supported", "refuted"),
            chains=(Chain(), Chain("wh")),
        )
        result = await agent.run(
            [Example("It was not hard", "It was easy")], aggregation="majority"
        )
        self.assertEqual(result.votes, (("supported", None),))
        self.assertEqual(result.predictions, ("supported",))
        self.assertIn("It was not hard", provider.calls[2])
        self.assertIn("Was it hard?", provider.calls[2])
        self.assertIn('"required": ["label"]', provider.calls[2])

    async def test_custom_mapping_and_identity_chain_need_no_extra_generation(self):
        async def map_answer(example, trace):
            return {"azure": "blue"}.get(trace.answer)

        agent = AMA(
            "Colour",
            ScriptedProvider(["azure", "unclear"]),
            chains=(Chain("identity"), Chain("identity")),
            map_answer=map_answer,
        )
        result = await agent.run([Example("What colour?")])
        self.assertEqual(result.predictions, ("blue",))
        self.assertEqual(result.calls, 2)

    async def test_failures_retain_calls_raw_text_and_completed_traces(self):
        agent = AMA("task", ScriptedProvider(["q", "a", '{"answer":"a"}', "   "]))
        with self.assertRaisesRegex(ValueError, "blank question"):
            await agent.run([Example("input")])
        self.assertEqual(agent.calls, 4)
        self.assertEqual(len(agent.traces), 1)
        self.assertEqual(agent.generations[-1], "   ")
        for invalid, error in [("not json", ValidationError), ('{"label":"invented"}', ValueError)]:
            agent = AMA("task", ScriptedProvider(["q", "a", invalid]), labels=("A", "B"))
            with self.assertRaises(error):
                await agent.run([Example("input")], aggregation="majority")
            self.assertEqual(agent.calls, 3)
            self.assertEqual(agent.traces[0].answer, "a")

    async def test_templates_render_from_another_directory_and_have_no_branches(self):
        agent = AMA("task", ScriptedProvider([]))
        previous = Path.cwd()
        os.chdir("/tmp")
        try:
            for method in (AMA.question_yes_no, AMA.question_wh, AMA.question_cloze):
                rendered = await method.render(agent, Example("marker"), Chain())
                self.assertIn("marker", rendered)
            self.assertIn("marker", await AMA.answer.render(agent, Example("x"), "marker", Chain()))
        finally:
            os.chdir(previous)
        for path in self.root.glob("*.j2"):
            self.assertEqual(
                list(Environment().parse(path.read_text()).find_all((nodes.If, nodes.CondExpr))), []
            )


class AggregationTests(unittest.TestCase):
    def test_majority_abstention_and_ties(self):
        self.assertEqual(majority_vote([None, "b", "a"]), "b")
        self.assertIsNone(majority_vote([None, None]))
        self.assertEqual(majority_vote(["a", "b", "b"]), "b")

    def test_dependency_policy_ignores_diagonal_and_dense_noise(self):
        self.assertEqual(
            select_dependency(np.array([[10, 0.2, 2], [0.2, 10, 0.1], [2, 0.1, 10]])), ((0, 2),)
        )
        self.assertEqual(select_dependency(np.full((3, 3), 2.0)), ())
        self.assertEqual(select_dependency(np.zeros((3, 3))), ())

    @unittest.skipUnless(
        importlib.util.find_spec("metal") and importlib.util.find_spec("cvxpy"),
        "install ama/requirements-ws.txt for official numerical integration",
    )
    def test_official_model_on_unlabeled_binary_and_multiclass_votes(self):
        rng = np.random.default_rng(42)
        for k in (2, 3):
            labels = tuple(str(i) for i in range(k))
            hidden = rng.integers(k, size=500)
            votes = np.column_stack(
                [
                    np.where(rng.random(500) < accuracy, hidden, rng.integers(k, size=500))
                    for accuracy in (0.9, 0.8, 0.7, 0.6)
                ]
            )
            strings = [[labels[c] for c in row] for row in votes]
            model = WeakSupervision(
                labels, WSConfig(epochs=400, learning_rate=0.001, dependencies=())
            )
            model.fit(strings)
            probabilities = model.predict_proba(strings)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1)
            self.assertGreater((probabilities.argmax(axis=1) == hidden).mean(), 0.75)
            np.testing.assert_allclose(model.model.p, np.ones(k) / k)
            np.testing.assert_allclose(model.predict_proba([[None] * 4]), [np.ones(k) / k])
            with self.assertRaises(ValueError):
                model.fit([[labels[0]] * 4] * 2)
            with self.assertRaisesRegex(RuntimeError, "fit the label model"):
                model.predict_proba(strings)
            with patch(
                "metal.label_model.LabelModel.train_model", side_effect=RuntimeError("failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    model.fit(strings)
            self.assertIsNone(model.model)

    @unittest.skipUnless(
        importlib.util.find_spec("metal") and importlib.util.find_spec("cvxpy"),
        "install ama/requirements-ws.txt for official numerical integration",
    )
    def test_recovers_correlated_pair_and_fits_official_dependency_model(self):
        rng = np.random.default_rng(42)
        hidden = rng.integers(2, size=1000)
        votes = np.column_stack(
            [np.where(rng.random(1000) < a, hidden, 1 - hidden) for a in (0.8, 0.75, 0.7, 0.7, 0.6)]
        )
        votes[:, 1] = np.where(rng.random(1000) < 0.85, votes[:, 0], votes[:, 1])
        strings = [[str(c) for c in row] for row in votes]
        model = WeakSupervision(
            ("0", "1"), WSConfig(epochs=100, dependency_epochs=500, dependency_learning_rate=1e-4)
        )
        model.fit(strings)
        self.assertEqual(model.dependencies, ((0, 1),))
        self.assertTrue(model.model.inv_form)
        self.assertGreater(model.model.d, votes.shape[1] * 2)
        probabilities = model.predict_proba(strings)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertGreater((probabilities.argmax(axis=1) == hidden).mean(), 0.65)


class BatchWSTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(importlib.util.find_spec("metal"), "install WS dependencies")
    async def test_target_and_extra_unlabeled_votes_fit_one_model(self):
        rng = np.random.default_rng(12)
        votes = rng.integers(1, 3, size=(40, 3))
        responses = [str(c) for row in votes for c in row]

        async def mapper(example, trace):
            return trace.answer

        agent = AMA(
            "task",
            ScriptedProvider(responses),
            labels=("1", "2"),
            chains=(Chain("identity"),) * 3,
            map_answer=mapper,
            ws_config=WSConfig(epochs=5, dependencies=()),
        )
        root = Path(__file__).resolve().parents[1] / "ama/prompts"
        with patch("slick.prompts.TEMPLATE_ROOT", root):
            result = await agent.run(
                [Example(str(i)) for i in range(10)],
                unlabeled=[Example(str(i)) for i in range(10, 40)],
            )
        self.assertEqual(result.calls, 120)
        self.assertEqual(len(result.predictions), 10)
        self.assertEqual(len(result.traces), 120)
        encoded = (votes[:, :, None] == np.array([1, 2])).reshape(40, 6)
        overlaps = encoded.astype(float).T @ encoded.astype(float) / 40
        np.testing.assert_allclose(agent.label_model.model.O.numpy(), overlaps, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
