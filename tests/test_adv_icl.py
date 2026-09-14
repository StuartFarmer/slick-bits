"""Check alternating minimax directions and instruction/example coordinates."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from adv_icl import AdvICL, DiscriminatorExample, DiscriminatorPrompt, Example, GeneratorPrompt
from tests.providers import ScriptedProvider


class AdvICLTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "adv_icl/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_alternating_instructions_and_demonstrations(self):
        async def generate(prompt, input):
            return "better" if prompt.instruction == "G2" else "fake"

        async def evaluate(prompt, input, output):
            if output == "real":
                return math.log(0.9 if prompt.instruction == "D2" else 0.5)
            return math.log(0.8 if output == "better" else 0.2)

        provider = ScriptedProvider(
            [
                '{"text":"D2"}',
                '{"input":"edited","output":"real","label":"real"}',
                '{"text":"G2"}',
                '{"input":"edited","output":"real"}',
            ]
        )
        agent = AdvICL("Task", provider, generate, evaluate)
        generator = GeneratorPrompt("G", (Example(input="demo", output="real"),))
        discriminator = DiscriminatorPrompt(
            "D", (DiscriminatorExample(input="demo", output="real", label="real"),)
        )
        result = await agent.run(
            generator,
            discriminator,
            [Example(input="x", output="real")],
            rounds=1,
            batch_size=1,
            candidates=1,
        )
        self.assertEqual(result["generator"].instruction, "G2")
        self.assertEqual(result["discriminator"].instruction, "D2")
        self.assertEqual(
            result["generator"].examples, generator.examples
        )  # Equal loss is rejected.
        self.assertEqual(
            [(h["player"], h["position"]) for h in result["history"]],
            [("discriminator", None), ("discriminator", 0), ("generator", None), ("generator", 0)],
        )
        self.assertGreater(result["history"][0]["after"], result["history"][0]["before"])
        self.assertLess(result["history"][2]["after"], result["history"][2]["before"])
        self.assertTrue(all("Task" in call for call in provider.calls))

    async def test_best_of_r_uses_frozen_coordinate(self):
        async def generate(prompt, input):
            return "fake"

        async def evaluate(prompt, input, output):
            return (
                math.log({"D": 0.5, "D1": 0.6, "D2": 0.8}[prompt.instruction])
                if output == "real"
                else math.log(0.2)
            )

        provider = ScriptedProvider(
            ['{"text":"D1"}', '{"text":"D2"}', '{"text":"G1"}', '{"text":"G2"}']
        )
        agent = AdvICL("Task", provider, generate, evaluate)
        result = await agent.run(
            GeneratorPrompt("G", ()),
            DiscriminatorPrompt("D", ()),
            [Example(input="x", output="real")],
            rounds=1,
            batch_size=1,
            candidates=2,
        )
        self.assertEqual(result["discriminator"].instruction, "D2")
        self.assertIn("Instruction: D\n", provider.calls[1])

    async def test_invalid_probability_and_generated_label(self):
        async def generate(prompt, input):
            return "fake"

        async def evaluate(prompt, input, output):
            return 0.0

        agent = AdvICL("Task", ScriptedProvider([]), generate, evaluate)
        with self.assertRaises(ValueError):
            await agent.run(
                GeneratorPrompt("G", ()),
                DiscriminatorPrompt("D", ()),
                [Example(input="x", output="real")],
                rounds=1,
                batch_size=1,
            )
        with self.assertRaises(ValueError):
            await agent.discriminator_example(
                DiscriminatorExample(input="x", output="a", label="real"),
                provider=ScriptedProvider(['{"input":"x","output":"a","label":"maybe"}']),
            )
