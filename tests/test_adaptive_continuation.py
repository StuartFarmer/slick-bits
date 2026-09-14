"""Exercise numeric control through real Slick parsing and external solver steps."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick.providers.base import ProviderError

from adaptive_continuation import AdaptiveContinuation, Gate, Measurement, Parameter
from adaptive_continuation.simp import SETTINGS, TUNABLES, fallback, grayness, valid_snapshot
from tests.providers import ScriptedProvider


def action(**parameters):
    return json.dumps({"parameters": parameters, "restart": True, "note": "recover"})


class ContinuationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "adaptive_continuation/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)

    async def test_control_clamps_gates_restarts_and_finishes_from_best(self):
        seen = []

        async def step(state, parameters):
            seen.append((state.copy(), parameters.copy()))
            state.append(len(seen))
            scores = [5, 3, 9, 4, 8, 2, 1]
            return Measurement(state, scores[len(seen) - 1], {"uncertainty": 0.8})

        initial = Measurement([], 10, {"uncertainty": 0.8})
        agent = AdaptiveContinuation(
            "Minimize arbitrary loss",
            ScriptedProvider([action(rate=99, radius=99)]),
            step,
            parameters={
                "rate": Parameter(0.01, 1, 0.1),
                "radius": Parameter(1, 4, 2, monotone="decrease"),
            },
            gates=(Gate("uncertainty", "rate", 0.2, 0.25),),
        )
        result = await agent.run(
            initial,
            iterations=5,
            settings={"call_every": 3},
            tail={"rate": 0.8, "radius": 1.5},
            tail_iterations=2,
        )
        self.assertEqual(initial.state, [])
        self.assertEqual(seen[3], ([1, 2], {"rate": 0.25, "radius": 2}))
        self.assertEqual(seen[5], ([1, 2], {"rate": 0.8, "radius": 1.5}))
        self.assertEqual(result.best.iteration, 2)
        self.assertEqual(result.best.measurement.state, [1, 2])
        self.assertEqual(result.final.objective, 1)
        self.assertEqual((result.main_evaluations, result.tail_evaluations), (5, 2))
        self.assertTrue(result.calls[0]["applied"]["restart"])

    async def test_gate_is_rechecked_on_restored_state(self):
        seen = []

        async def step(state, parameters):
            seen.append(parameters["rate"])
            return Measurement(state + 1, -state, {"risk": 0})

        agent = AdaptiveContinuation(
            "maximize",
            ScriptedProvider([action(rate=0.9)]),
            step,
            parameters={"rate": Parameter(0, 1, 0.1)},
            maximize=True,
            gates=(Gate("risk", "rate", 0.5, 0.2),),
        )
        result = await agent.run(
            Measurement(0, 10, {"risk": 1}), iterations=5, settings={"call_every": 3}
        )
        self.assertEqual(result.best.iteration, 0)
        self.assertEqual(seen[3], 0.2)
        self.assertEqual(len(result.calls), 1)

    async def test_invalid_responses_fall_back_but_solver_errors_propagate(self):
        async def step(state, parameters):
            return Measurement(state + 1, 5 - state, {})

        invalid = ["not json", action(wrong=1), action(rate=float("nan")), ProviderError("offline")]
        agent = AdaptiveContinuation(
            "task",
            ScriptedProvider(invalid),
            step,
            parameters={"rate": Parameter(0, 1, 0.1)},
            schedule=lambda i, n, settings: {"rate": 0.3},
        )
        result = await agent.run(Measurement(0, 10, {}), iterations=5, settings={"call_every": 1})
        self.assertEqual(result.fallbacks, 4)
        self.assertEqual(result.calls[0]["response"], "not json")
        self.assertTrue(all("error" in call for call in result.calls))
        self.assertFalse(result.primary_eligible)

        async def broken(state, parameters):
            raise ValueError("solver broke")

        agent.evaluate = broken
        with self.assertRaisesRegex(ValueError, "solver broke"):
            await agent.run(Measurement(0, 10, {}), iterations=1)
        self.assertEqual(agent.main_evaluations, 1)

    async def test_gate_changes_between_calls_and_missing_snapshot_blocks_restart(self):
        seen = []

        async def step(state, parameters):
            seen.append(parameters["rate"])
            return Measurement(state + 1, 10 - state, {"risk": float(state >= 2)}, False)

        agent = AdaptiveContinuation(
            "task",
            ScriptedProvider([action(rate=0.9)]),
            step,
            parameters={"rate": Parameter(0, 1, 0.1)},
            gates=(Gate("risk", "rate", 0.5, 0.2),),
        )
        result = await agent.run(
            Measurement(0, 10, {"risk": 0}, False), iterations=4, settings={"call_every": 2}
        )
        self.assertEqual(seen, [0.1, 0.1, 0.9, 0.2])
        self.assertFalse(result.calls[0]["applied"]["restart"])
        self.assertEqual(result.calls[0]["observation"]["stagnation"], 2)

    async def test_strict_generated_numbers_and_meta_failure_retention(self):
        async def step(state, parameters):
            return Measurement(state, 1, {})

        for bad in (True, "0.2", float("inf")):
            agent = AdaptiveContinuation(
                "task",
                ScriptedProvider([action(rate=bad)]),
                step,
                parameters={"rate": Parameter(0, 1, 0.1)},
            )
            result = await agent.run(
                Measurement(0, 1, {}), iterations=2, settings={"call_every": 1}
            )
            self.assertEqual(result.fallbacks, 1)
            self.assertEqual(result.history[-1]["parameters"], {"rate": 0.1})

        async def comparison(settings):
            return {"objective": 1}

        agent.provider = ScriptedProvider(['{"updates":{"unknown":3},"note":"bad"}'])
        result = await agent.run_meta(comparison, rounds=2, settings=SETTINGS, tunables=TUNABLES)
        self.assertEqual(result["settings"], SETTINGS)
        self.assertIn("error", result["reflections"][0])

        async def invalid(state, parameters):
            return Measurement(state, float("nan"), {})

        agent.evaluate = invalid
        with self.assertRaisesRegex(ValueError, "measured objective"):
            await agent.run(Measurement(0, 1, {}), iterations=1)

    async def test_fixed_has_no_tail_and_invalid_best_uses_initial(self):
        seen = []

        async def step(state, parameters):
            seen.append(state)
            return Measurement(state + 1, 5, {}, feasible=False)

        agent = AdaptiveContinuation(
            "task", ScriptedProvider([]), step, parameters={"rate": Parameter(0, 1, 0.1)}
        )
        initial = Measurement(0, 10, {}, feasible=False)
        result = await agent.run(
            initial, iterations=2, mode="fixed", tail={"rate": 0.5}, tail_iterations=2
        )
        self.assertEqual((result.tail_evaluations, len(result.calls)), (0, 0))
        result = await agent.run(initial, iterations=2, tail={"rate": 0.5}, tail_iterations=1)
        self.assertIsNone(result.best)
        self.assertEqual(seen[-1], 0)
        self.assertEqual(result.tail_origin, "initial")

    async def test_meta_clamps_updates_and_next_run_observes_them(self):
        provider = ScriptedProvider(
            ['{"updates":{"call_every":99,"grayness_gate":0.9},"note":"less overhead"}']
        )
        seen = []

        async def comparison(settings):
            seen.append(settings.copy())
            return {"llm": {"objective": 3}, "fixed": {"objective": 4}}

        agent = AdaptiveContinuation("task", provider, None, parameters={})
        result = await agent.run_meta(comparison, rounds=2, settings=SETTINGS, tunables=TUNABLES)
        self.assertEqual(seen[1]["call_every"], 15)
        self.assertEqual(seen[1]["grayness_gate"], 0.35)
        self.assertEqual(SETTINGS["call_every"], 5)
        self.assertEqual(len(result["comparisons"]), 2)
        self.assertIn('"fixed"', provider.calls[0])

    async def test_simp_preset_and_external_templates(self):
        self.assertEqual(grayness([0, 1]), 0)
        self.assertEqual(grayness([0.5, 0.5]), 1)
        settings = SETTINGS.copy()
        values = fallback(75, 100, settings)
        self.assertEqual(set(values), {"penal", "beta", "rmin", "move"})
        self.assertFalse(valid_snapshot(Measurement(None, 1, {"grayness": 0.25}), values))
        self.assertTrue(valid_snapshot(Measurement(None, 1, {"grayness": 0.1}), values))
        agent = AdaptiveContinuation("marker", ScriptedProvider([]), None, parameters={})
        cwd = Path.cwd()
        try:
            os.chdir("/tmp")
            rendered = await AdaptiveContinuation.control.render(agent, {"iteration": 1})
            meta = await AdaptiveContinuation.reflect.render(agent, [], SETTINGS, TUNABLES)
        finally:
            os.chdir(cwd)
        for text in (rendered, meta):
            self.assertIn("marker", text)
            self.assertIn("# Output Format", text)
        for path in self.root.glob("*.j2"):
            tree = Environment().parse(path.read_text())
            self.assertFalse(list(tree.find_all((nodes.If, nodes.CondExpr))))


if __name__ == "__main__":
    unittest.main()
