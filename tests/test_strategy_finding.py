"""Exercise strategy discovery, category selection and the numerical combiner."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from jinja2 import Environment, nodes

from strategy_finding import (
    Candidate,
    CandidateRejected,
    Evaluation,
    StrategyFinder,
    TrainingData,
    fit_combiner,
)
from tests.providers import ScriptedProvider


class StrategyFindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT",
            Path(__file__).resolve().parents[1] / "strategy_finding/prompts",
        )
        root.start()
        self.addCleanup(root.stop)

    async def test_generation_selection_and_incremental_update(self):
        measured, prepared = [], []

        async def evaluate(candidate, context):
            measured.append((candidate.content, context))
            return Evaluation(0.3, "Measured on training observations only")

        async def prepare(candidates):
            prepared.append(candidates)
            n = len(candidates)
            return TrainingData(np.ones((8, n)), np.ones(8), np.ones((3, n)), np.ones(3))

        provider = ScriptedProvider(
            [
                '{"content":"use signals"}',
                '{"categories":["trend", "level"]}',
                '{"candidates":[{"name":"A", "content":"a"},'
                '{"name":"A duplicate", "content":"a"},{"name":"B", "content":"b"}]}',
                '{"candidates":[{"name":"C", "content":"c"}]}',
                '{"score":0.9,"reason":"stable"}',
                '{"score":0.5,"reason":"acceptable"}',
                '{"score":0.7,"reason":"stable"}',
                '{"score":0.7,"reason":"acceptable"}',
                '{"score":0.8,"reason":"stable"}',
                '{"score":0.8,"reason":"acceptable"}',
            ]
        )
        agent = StrategyFinder("Predict machine failures", provider, evaluate, prepare)
        result = await agent.run(["source"], context="current conditions", epochs=2)
        self.assertEqual([x.content for x in result.factory], ["a", "b", "c"])
        self.assertEqual([x.content for x in result.selected], ["a", "c"])
        self.assertEqual(prepared, [result.selected])
        self.assertEqual(len(measured), 3)
        self.assertEqual(result.generation_calls, 10)
        self.assertEqual(result.evaluations, 3)
        self.assertEqual(result.model.predict(np.ones((2, 2))).shape, (2,))
        for record in agent.generations:
            self.assertIsNotNone(record.response)

        agent.provider = ScriptedProvider(
            [
                '{"score":0.1,"reason":"changed"}',
                '{"score":0.1,"reason":"changed"}',
                '{"score":0.9,"reason":"changed"}',
                '{"score":0.9,"reason":"changed"}',
                '{"score":0.1,"reason":"changed"}',
                '{"score":0.1,"reason":"changed"}',
            ]
        )
        updated = await agent.run([], factory=result.factory, context="new", epochs=1)
        self.assertEqual([x.content for x in updated.selected], ["b"])
        self.assertEqual(updated.evaluations, 3)
        self.assertEqual(updated.generation_calls, 6)
        self.assertEqual(measured[-1][1], "new")

    async def test_strict_threshold_rejection_and_empty_selection(self):
        async def evaluate(candidate, context):
            if candidate.content == "bad":
                raise CandidateRejected("unsupported operator")
            return Evaluation(0.5, "historical evidence")

        async def prepare(candidates):
            self.fail("empty selection must not prepare data")

        provider = ScriptedProvider(
            [
                '{"score":0.5,"reason":"boundary"}',
                '{"score":0.5,"reason":"boundary"}',
            ]
        )
        agent = StrategyFinder("task", provider, evaluate, prepare)
        result = await agent.run(
            [],
            factory=(
                Candidate(category="one", name="A", content="a"),
                Candidate(category="two", name="Bad", content="bad"),
            ),
            threshold=0.5,
        )
        self.assertEqual(result.selected, ())
        self.assertIsNone(result.model)
        self.assertEqual(result.evaluations, 2)
        self.assertEqual(result.rejections[0].reason, "unsupported operator")

    async def test_invalid_generation_keeps_raw_response_and_no_retry(self):
        async def unused(*args):
            self.fail("must not reach evaluation")

        for raw in ["not json", '{"categories":["same","same"]}']:
            provider = ScriptedProvider(['{"content":"evidence"}', raw])
            agent = StrategyFinder("task", provider, unused, unused)
            with self.assertRaises(ValueError):
                await agent.run(["doc"])
            self.assertEqual(agent.generations[-1].response, raw)
            self.assertEqual(len(provider.calls), 2)

    async def test_nonfinite_measurement_and_unexpected_errors_propagate(self):
        candidate = Candidate(category="c", name="n", content="x")

        async def prepare(candidates):
            self.fail("must not train")

        async def evaluate(candidate, context):
            return Evaluation(float("nan"), "invalid measurement")

        agent = StrategyFinder("task", ScriptedProvider([]), evaluate, prepare)
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run([], factory=(candidate,))
        self.assertEqual(agent.evaluations, 1)

        async def broken(candidate, context):
            raise RuntimeError("worker unavailable")

        agent.evaluate = broken
        with self.assertRaisesRegex(RuntimeError, "worker unavailable"):
            await agent.run([], factory=(candidate,))

    async def test_ties_keep_first_and_risk_output_is_validated(self):
        async def evaluate(candidate, context):
            return Evaluation(1, "measured evidence")

        async def prepare(candidates):
            return TrainingData(np.ones((3, 1)), np.ones(3), np.ones((2, 1)), np.ones(2))

        candidates = tuple(Candidate(category="c", name=s, content=s) for s in ["a", "b"])
        response = '{"score":0.8,"reason":"supported"}'
        agent = StrategyFinder("task", ScriptedProvider([response] * 4), evaluate, prepare)
        result = await agent.run([], factory=candidates, epochs=0)
        self.assertEqual(result.selected, candidates[:1])

        invalid = '{"score":1.5,"reason":"unsupported"}'
        agent.provider = ScriptedProvider([response, invalid])
        with self.assertRaises(ValueError):
            await agent.run([], factory=candidates)
        self.assertEqual(agent.generations[-1].response, invalid)
        self.assertEqual(agent.evaluations, 1)

    async def test_irrelevant_documents_and_provider_failure_accounting(self):
        async def unused(*args):
            self.fail("must not evaluate or train")

        agent = StrategyFinder("task", ScriptedProvider(['{"content":""}']), unused, unused)
        result = await agent.run(["irrelevant"])
        self.assertEqual(result.factory, ())
        self.assertEqual(result.generation_calls, 1)
        agent.provider = ScriptedProvider([OSError("connection lost")])
        with self.assertRaisesRegex(OSError, "connection lost"):
            await agent.run(["doc"])
        self.assertEqual(len(agent.generations), 1)
        self.assertIn("connection lost", agent.generations[0].error)

    async def test_templates_render_from_other_directory_without_control_flow(self):
        async def unused(*args):
            self.fail("rendering must not execute a provider")

        agent = StrategyFinder("machine reliability", ScriptedProvider([]), unused, unused)
        candidate = Candidate(category="trend", name="A", content="a")
        evaluation = Evaluation(0.7, "historical evidence")
        operations = [
            (StrategyFinder.filter_document, ("document",)),
            (StrategyFinder.categorize, ("evidence",)),
            (StrategyFinder.generate, ("evidence", "trend", [])),
            (StrategyFinder.confidence, (candidate, evaluation, "conditions")),
            (StrategyFinder.risk, (candidate, evaluation, "conditions")),
        ]
        root = Path(__file__).resolve().parents[1] / "strategy_finding/prompts"
        previous = Path.cwd()
        try:
            os.chdir("/tmp")
            for method, args in operations:
                rendered = await method.render(agent, *args)
                self.assertIn("machine reliability", rendered)
                self.assertIn('"properties"', rendered)
        finally:
            os.chdir(previous)
        for template in root.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


class CombinerTests(unittest.TestCase):
    def test_learning_reproducibility_and_exact_local_weights(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(160, 2))
        y = 2 * x[:, 0] - x[:, 1] + 0.4
        data = TrainingData(x[:120], y[:120], x[120:], y[120:])
        model = fit_combiner(data, epochs=600, learning_rate=0.03, regularization=0, seed=4)
        prediction = model.predict(x[120:])
        self.assertLess(np.mean((prediction - y[120:]) ** 2), 0.08)
        weights, intercepts = model.local_weights(x[120:])
        np.testing.assert_allclose((weights * x[120:]).sum(axis=1) + intercepts, prediction)
        self.assertEqual(model.w1.shape, (2, 10))
        self.assertAlmostEqual(model.validation_loss, np.mean((prediction - y[120:]) ** 2))
        again = fit_combiner(data, epochs=600, learning_rate=0.03, regularization=0, seed=4)
        np.testing.assert_array_equal(model.w1, again.w1)

    def test_validation_does_not_fit_scaling_and_constant_columns_work(self):
        x = np.column_stack((np.arange(8), np.ones(8)))
        data = TrainingData(x, np.arange(8), np.array([[1000, 1]]), np.array([5]))
        model = fit_combiner(data, epochs=0)
        np.testing.assert_array_equal(model.minimum, [0, 1])
        np.testing.assert_array_equal(model.scale, [7, 1])
        self.assertTrue(np.isfinite(model.predict(data.x_validation)).all())


if __name__ == "__main__":
    unittest.main()
