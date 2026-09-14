"""MoP clustering, complement proposals, expert assignment and query routing."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mop import Example, MoP
from tests.providers import ScriptedProvider


class MoPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "mop/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_regions_choose_different_instructions_and_route_inputs(self):
        async def embed(inputs):
            return np.array(
                [[1 if x.startswith("A") else -1, 0.1 if x.endswith("2") else 0] for x in inputs],
                dtype=float,
            )

        async def evaluate(instruction, demos, examples):
            return 1 if not demos else float(instruction.endswith(demos[0].input[0]))

        provider = ScriptedProvider(["instruction A", "instruction B", "answer A", "answer B"])
        agent = MoP("Task", provider, evaluate, embed)
        examples = [Example(x, "target " + x) for x in ("A1", "A2", "B1", "B2")]
        result = await agent.run(
            examples,
            examples,
            max_experts=2,
            candidates_per_region=1,
            pool_size=2,
            regional_examples=2,
        )
        self.assertEqual(len(result["experts"]), 2)
        self.assertEqual(
            {e.instruction for e in result["experts"]}, {"instruction A", "instruction B"}
        )
        self.assertEqual(result["evaluations"], 6)
        self.assertEqual(result["optimizer_calls"], 2)
        self.assertNotEqual(await agent.route("A query"), await agent.route("B query"))
        self.assertEqual(await agent.predict("A query"), "answer A")
        self.assertIn("instruction A", provider.calls[2])
        self.assertEqual(await agent.predict("B query"), "answer B")
        self.assertIn("instruction B", provider.calls[3])
        # Each complement contains one cluster's examples, never all four.
        for context in provider.calls[:2]:
            self.assertNotEqual("Input: A1" in context, "Input: B1" in context)

    async def test_zero_inertia_collapses_identical_embeddings(self):
        async def embed(inputs):
            return np.ones((len(inputs), 2))

        async def evaluate(instruction, demos, examples):
            return -1

        result = await MoP("Task", ScriptedProvider(["instruction"]), evaluate, embed).run(
            [Example("a", "A"), Example("b", "B")],
            [Example("v", "V")],
            max_experts=2,
            candidates_per_region=1,
        )
        self.assertEqual(len(result["experts"]), 1)
        self.assertEqual(result["experts"][0].score, -1)

    async def test_bad_generation_is_not_evaluated(self):
        async def embed(inputs):
            return np.ones((len(inputs), 1))

        agent = MoP("Task", ScriptedProvider([" "]), None, embed)
        with self.assertRaisesRegex(ValueError, "empty"):
            await agent.run([Example("a", "A")], [Example("v", "V")], max_experts=1)
        self.assertEqual(agent.evaluations, 0)
