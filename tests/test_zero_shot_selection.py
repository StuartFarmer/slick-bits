"""Check independent synthesis and the paper's numerical selection rules offline."""

import asyncio
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from jinja2 import Environment, nodes

from tests.providers import ScriptedProvider
from zero_shot_selection import CandidateRejected, Operator, ZeroShotSelection
from zero_shot_selection.selectors import gpt_selection, kimi_selection

SOURCE = "def custom_selection(population, k=100, status={}):\n    return population[:k]\n"


class SynthesisTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "zero_shot_selection/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_independent_calls_reject_then_validate_all_before_evaluation(self):
        events = []
        second = SOURCE.replace("population[:k]", "[population[0]] * k")
        provider = ScriptedProvider(["not JSON", Operator(source=SOURCE), Operator(source=second)])

        async def validate(source):
            events.append(("validate", source))

        async def evaluate(source):
            events.append(("evaluate", source))
            return {"train/task-a/seed-0": 0.8, "test/task-a/seed-0": -0.2}

        agent = ZeroShotSelection(
            "Select schedules with fewer missed deadlines.", provider, validate, evaluate
        )
        result = await agent.run(count=2, max_attempts=3)
        self.assertEqual([item.source for item in result], [SOURCE, second])
        self.assertEqual([phase for phase, _ in events], ["validate"] * 2 + ["evaluate"] * 2)
        self.assertEqual(len(set(provider.calls)), 1)
        self.assertNotIn(second, provider.calls[-1])
        self.assertEqual(agent.attempts[0]["response"], "not JSON")
        self.assertEqual(agent.attempts[0]["phase"], "generate")
        self.assertIn("error", agent.attempts[0])
        self.assertEqual(result[0].metrics["test/task-a/seed-0"], -0.2)

    async def test_bad_source_and_functional_failures_consume_budget(self):
        checked = []

        async def validate(source):
            checked.append(source)
            raise CandidateRejected("wrong number of parents")

        bad_sources = [
            "def custom_selection(:",
            "def other(population): pass",
            "def custom_selection(population, k, status, extra): pass",
            SOURCE,
        ]
        provider = ScriptedProvider([Operator(source=source) for source in bad_sources])
        agent = ZeroShotSelection("Any population", provider, validate)
        with self.assertRaisesRegex(RuntimeError, "0 of 1"):
            await agent.run(count=1, max_attempts=4)
        self.assertEqual(checked, [SOURCE])
        self.assertEqual(len(agent.attempts), 4)
        self.assertTrue(all("error" in item for item in agent.attempts))

    async def test_timeout_cancels_validator_and_keeps_partial_results(self):
        cancelled = []
        count = 0

        async def validate(source):
            nonlocal count
            count += 1
            if count == 2:
                try:
                    await asyncio.sleep(1)
                finally:
                    cancelled.append(True)

        provider = ScriptedProvider([Operator(source=SOURCE)] * 2)

        async def evaluate(source):
            self.fail("partial batches must never reach evaluation")

        agent = ZeroShotSelection(
            "Any population", provider, validate, evaluate, validation_timeout=0.01
        )
        with self.assertRaisesRegex(RuntimeError, "1 of 2"):
            await agent.run(count=2, max_attempts=2)
        self.assertEqual(cancelled, [True])
        self.assertEqual(len(agent.operators), 1)
        self.assertEqual(agent.attempts[-1]["phase"], "validate")

    async def test_generated_module_is_never_executed_by_the_agent(self):
        source = 'raise RuntimeError("must only execute in the external worker")\n' + SOURCE
        seen = []

        async def validate(code):
            # Deliberately no execution: this test exercises the agent boundary only.
            seen.append(code)

        agent = ZeroShotSelection("Task", ScriptedProvider([Operator(source=source)]), validate)
        result = await agent.run(count=1)
        self.assertEqual(seen, [source])
        self.assertEqual(result[0].source, source)
        self.assertIsNone(result[0].metrics)

    async def test_infrastructure_errors_propagate_without_resynthesis(self):
        async def validate(source):
            raise RuntimeError("worker offline")

        provider = ScriptedProvider([Operator(source=SOURCE)] * 2)
        agent = ZeroShotSelection("Task", provider, validate)
        with self.assertRaisesRegex(RuntimeError, "worker offline"):
            await agent.run(count=1)
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("worker offline", agent.attempts[0]["error"])

    async def test_nonfinite_metrics_and_evaluator_failures_do_not_regenerate(self):
        async def validate(source):
            pass

        for failure in (float("nan"), RuntimeError("benchmark failed")):

            async def evaluate(source):
                if isinstance(failure, Exception):
                    raise failure
                return {"score": failure}

            provider = ScriptedProvider([Operator(source=SOURCE)])
            agent = ZeroShotSelection("Task", provider, validate, evaluate)
            with self.assertRaises((ValueError, RuntimeError)):
                await agent.run(count=1)
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(len(agent.operators), 1)
            self.assertEqual(agent.attempts[0]["phase"], "evaluate")
            self.assertIn("error", agent.attempts[0])

    async def test_provider_errors_and_tool_responses_are_recorded(self):
        async def validate(source):
            pass

        agent = ZeroShotSelection("Task", ScriptedProvider([OSError("network")]), validate)
        with self.assertRaises(OSError):
            await agent.run(count=1)
        self.assertIn("network", agent.attempts[0]["error"])
        agent = ZeroShotSelection("Task", ScriptedProvider([("raw", ["tool"])]), validate)
        with self.assertRaises(RuntimeError):
            await agent.run(count=1, max_attempts=1)
        self.assertEqual(agent.attempts[0]["response"], "raw")

    async def test_template_binding_schema_and_launch_directories(self):
        async def validate(source):
            pass

        agent = ZeroShotSelection("Rank designs {{ literal }}", ScriptedProvider([]), validate)
        previous = Path.cwd()
        try:
            for directory in (self.templates.parent, self.templates.parent.parent):
                os.chdir(directory)
                rendered = await ZeroShotSelection.synthesize.render(agent)
                self.assertIn("Rank designs {{ literal }}", rendered)
                self.assertIn('"source"', rendered)
                self.assertEqual(rendered.count("# Output Format"), 1)
                self.assertNotIn("symbolic regression", rendered)
            parsed = Environment().parse((self.templates / "synthesize.j2").read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)


class SelectorTests(unittest.TestCase):
    def test_kimi_intermediate_blend_when_novelty_and_fitness_disagree(self):
        errors = np.array([[1.0, 0.0], [0.0, 2.0], [0.7, 0.8]])
        novelty = np.array([13 / 30, -13 / 15, -43 / 300])
        for stage, width in ((0.0, 2), (0.35, 4), (1.0, 8)):
            a = 19 / 21 * stage
            fitness = np.array([-0.5 * (1 + a), -2 * (1 - a), -0.565 + 0.075 * a])
            scores = stage * (fitness - fitness.mean()) / (fitness.std() + 1e-12)
            scores += (1 - stage) * (novelty - novelty.mean()) / (novelty.std() + 1e-12)
            scores += 0.1 * np.array([0.0, 1.0, 0.5])
            draws = np.random.default_rng(13).integers(3, size=(40, width))
            expected = draws[np.arange(40), scores[draws].argmax(axis=1)]
            actual = kimi_selection(
                errors,
                [20.0, 1.0],
                [3.0, 1.0, 2.0],
                40,
                stage,
                rng=np.random.default_rng(13),
            )
            np.testing.assert_array_equal(actual, expected)

    def test_kimi_stage_endpoints_and_curriculum_against_hand_scores(self):
        # The supplied matrix is already normalized; no hidden rescaling is allowed.
        errors = np.array([[0.0, 0.5], [1.0, 0.0], [0.5, 1.0]])
        sizes = np.array([3.0, 1.0, 2.0])
        variances = np.array([1.0, 3.0])
        novelty = np.array([0.75, 0.5, 0.25])
        weighted_fitness = np.array([-0.1875, -0.25, -0.8125])
        for stage, values, width in ((0.0, novelty, 2), (1.0, weighted_fitness, 8)):
            scores = (values - values.mean()) / (values.std() + 1e-12)
            scores += 0.1 * np.array([0.0, 1.0, 0.5])
            draws = np.random.default_rng(13).integers(3, size=(20, width))
            expected = draws[np.arange(20), scores[draws].argmax(axis=1)]
            actual = kimi_selection(
                errors, variances, sizes, 20, stage, rng=np.random.default_rng(13)
            )
            np.testing.assert_array_equal(actual, expected)

    def test_gpt_elite_order_and_crowding_probabilities(self):
        # All rows are unit vectors; two duplicate behaviors and one opposite.
        behavior = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        losses = np.array([0.0, 1.0, 2.0])
        sizes = np.array([1.0, 2.0, 3.0])
        # At t=0: fitness weight 1.2, diversity .8, parsimony .15.
        score = np.array([1.35, 0.675, 0.8])
        mass = np.exp((score - score.max()) / 0.9)
        mass *= 0.35 + 0.65 / np.array([3.0, 3.0, 2.0])
        mass[0] *= 0.5
        expected = np.r_[0, np.random.default_rng(4).choice(3, size=4, p=mass / mass.sum())]
        actual = gpt_selection(behavior, losses, sizes, sizes, 5, 0.0, rng=np.random.default_rng(4))
        np.testing.assert_array_equal(actual, expected)

    def test_gpt_does_not_normalize_behavior_and_exposes_elite_downweight(self):
        behavior = np.array([[2.0, 0.0], [0.4, 0.0], [-0.1, 0.0]])
        score = np.array([1.35, 0.675 + 0.8 * 68 / 183, 0.8])
        for downweight in (0.25, 0.5, 1.0):
            mass = np.exp((score - score.max()) / 0.9)
            mass *= 0.35 + 0.65 / np.array([3.0, 2.0, 1.0])
            mass[0] *= downweight
            rng = Mock(wraps=np.random.default_rng(4))
            indices = gpt_selection(
                behavior,
                [0, 1, 2],
                [1, 2, 3],
                [1, 2, 3],
                5,
                0,
                rng=rng,
                elite_downweight=downweight,
            )
            self.assertEqual(indices[0], 0)
            np.testing.assert_allclose(rng.choice.call_args.kwargs["p"], mass / mass.sum())

    def test_degenerate_populations_cardinality_and_no_mutation(self):
        for n in (1, 4):
            matrix = np.zeros((n, 3))
            values = np.ones(n)
            before = matrix.copy()
            for stage in (0.0, 0.5, 1.0):
                for k in (0, 1, 9):
                    kimi = kimi_selection(
                        matrix, np.zeros(3), values, k, stage, rng=np.random.default_rng(2)
                    )
                    gpt = gpt_selection(
                        matrix, values, values, values, k, stage, rng=np.random.default_rng(2)
                    )
                    for selected in (kimi, gpt):
                        self.assertEqual(selected.shape, (k,))
                        self.assertTrue(np.all((selected >= 0) & (selected < n)))
            np.testing.assert_array_equal(matrix, before)
        self.assertEqual(kimi_selection([], [], [], 0, 0).size, 0)
        self.assertEqual(gpt_selection([], [], [], [], 0, 0).size, 0)


if __name__ == "__main__":
    unittest.main()
