"""OPRO score-history ordering, frozen proposal batches and deduplication."""

import math
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from opro import OPRO
from tests.providers import ScriptedProvider


class OPROTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "opro/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_scored_trajectory_order_snapshot_and_dedup(self):
        provider = ScriptedProvider(
            [
                "<TEXT>winning</TEXT>",
                "<TEXT>middle</TEXT>",
                "<TEXT>winning</TEXT>",
                "<TEXT>new</TEXT>",
            ]
        )
        scores = {"low": -1, "middle": 1, "high": 3, "winning": 4, "new": 2}
        evaluated = []

        async def score(text):
            evaluated.append(text)
            return scores[text]

        result = await OPRO("Summarize legal prose.", provider, score).run(
            ["low", "high", "middle"],
            iterations=2,
            proposals_per_step=2,
            history_size=2,
            exemplars=["A supplied example"],
        )
        self.assertEqual(result["best"]["prompt"], "winning")
        self.assertEqual(evaluated, ["low", "high", "middle", "winning", "new"])
        self.assertEqual(result["duplicates"], 2)
        self.assertEqual(result["evaluations"], 5)
        self.assertEqual(result["optimizer_calls"], 4)
        for context in provider.calls[:2]:
            self.assertLess(context.index("\nmiddle\n"), context.index("\nhigh\n"))
            self.assertNotIn("winning", context)
            self.assertNotIn("\nlow\n", context)
            self.assertIn("A supplied example", context)
        for context in provider.calls[2:]:
            self.assertLess(context.index("\nhigh\n"), context.index("\nwinning\n"))

    async def test_bad_generation_is_counted_before_evaluation(self):
        async def score(text):
            return 1

        for response in ("missing", "<TEXT> </TEXT>", "<TEXT>a</TEXT><TEXT>b</TEXT>"):
            agent = OPRO("Task", ScriptedProvider([response]), score)
            with self.assertRaises(ValueError) as error:
                await agent.run(["seed"], iterations=1, proposals_per_step=1)
            self.assertIn(repr(response), str(error.exception))
            self.assertEqual(agent.evaluations, 1)
            self.assertEqual(agent.optimizer_calls, 1)
            self.assertEqual(agent.raw_responses, [response])

    async def test_minimization_sampling_and_stagnation(self):
        provider = ScriptedProvider(
            [f"<TEXT>{value}</TEXT>" for value in ("1", "4", "1", "3", "5", "1")]
        )

        async def loss(text):
            return float(text) ** 2

        examples = [f"Example {i}" for i in range(8)]
        result = await OPRO("Return a number.", provider, loss, maximize=False).run(
            ["9", "2", "6"],
            iterations=10,
            proposals_per_step=2,
            history_size=2,
            exemplars=examples,
            seed=42,
            patience=2,
        )
        self.assertEqual(result["best"]["prompt"], "1")
        self.assertEqual([item["score"] for item in result["history"]], [1, 1, 1])
        self.assertEqual(result["stop_reason"], "patience")
        self.assertEqual(result["optimizer_calls"], 6)
        self.assertEqual(result["evaluations"], 7)
        self.assertEqual(result["duplicates"], 2)
        self.assertEqual(len(result["raw_responses"]), 6)
        self.assertLess(provider.calls[0].index("\n6\n"), provider.calls[0].index("\n2\n"))
        self.assertNotIn("\n9\n", provider.calls[0])
        self.assertIn("lower scores are better", provider.calls[0])
        rng = random.Random(42)
        for step in range(3):
            expected = set(rng.sample(examples, 3))
            self.assertEqual(provider.calls[step * 2], provider.calls[step * 2 + 1])
            shown = {line for line in provider.calls[step * 2].splitlines() if line in examples}
            self.assertEqual(shown, expected)

    async def test_empty_seed_ties_and_zero_iterations(self):
        async def score(text):
            return 0

        provider = ScriptedProvider([])
        result = await OPRO("Optimize a prompt.", provider, score).run(
            ["", "second", ""], iterations=0
        )
        self.assertEqual(result["best"]["prompt"], "")
        self.assertEqual(result["stop_reason"], "iterations")
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["history"], [])
        self.assertEqual(provider.calls, [])

    async def test_improvement_resets_patience_and_examples_can_be_disabled(self):
        async def score(text):
            return float(text)

        provider = ScriptedProvider([f"<TEXT>{n}</TEXT>" for n in (0, 2, 1, 2)])
        result = await OPRO("Maximize the number.", provider, score).run(
            ["1"],
            iterations=10,
            proposals_per_step=1,
            exemplars=["Hidden example"],
            exemplars_per_step=0,
            patience=2,
        )
        self.assertEqual([p["score"] for p in result["history"]], [1, 2, 2, 2])
        self.assertEqual(result["stop_reason"], "patience")
        self.assertTrue(all("Hidden example" not in context for context in provider.calls))

    async def test_render_without_provider_and_evaluator_failure(self):
        async def fail(text):
            if text == "candidate":
                raise RuntimeError("evaluation failed")
            return 1

        agent = OPRO(
            "Optimize JSON configurations.", ScriptedProvider(["<TEXT>candidate</TEXT>"]), fail
        )
        rendered = await OPRO.propose.render(agent, [], [])
        self.assertIn("candidate solution", rendered)
        self.assertNotIn("Generate an instruction", rendered)
        self.assertEqual(agent.provider.calls, [])
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            await agent.run(["seed"], iterations=1, proposals_per_step=1)
        self.assertEqual(agent.evaluations, 2)
        self.assertEqual(len(agent.archive), 1)
        self.assertEqual(agent.raw_responses, ["<TEXT>candidate</TEXT>"])

    async def test_measurement_and_transport_failures(self):
        async def invalid(text):
            return math.inf

        with self.assertRaisesRegex(ValueError, "finite"):
            await OPRO("Task", ScriptedProvider([]), invalid).run(["seed"], iterations=0)

        async def score(text):
            return 1

        provider = ScriptedProvider([RuntimeError("transport")])
        with self.assertRaisesRegex(RuntimeError, "transport"):
            await OPRO("Task", provider, score).run(["seed"], iterations=1)
        self.assertEqual(len(provider.calls), 1)
