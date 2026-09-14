"""Check MOPO's three layers, token edits, and objective-specific survivors."""

import random
import unittest
from pathlib import Path
from unittest.mock import patch

from mopo import MOPO, Individual
from mopo.agent import GeneratedText
from tests.providers import ScriptedProvider


async def fill(prefix, suffix):
    return "new"


class MOPOTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "mopo/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_three_layers_and_word_variation(self):
        async def evaluate(text):
            return (len(text), -len(text))

        # Two operator rewrites, two combinations, four sentence paraphrases.
        provider = ScriptedProvider(
            [
                GeneratedText(text=s)
                for s in (
                    "new combine",
                    "new paraphrase",
                    "combined <class>",
                    "mixed <class>",
                    "first <class>",
                    "second <class>",
                    "third <class>",
                    "fourth <class>",
                )
            ]
        )
        agent = MOPO("Task", provider, evaluate, ["length", "brevity"], fill, ["<class>"])
        result = await agent.run(
            ["write <class>", "say <class>"],
            ["combine"],
            ["paraphrase"],
            generations=1,
            population_size=2,
            specialists=1,
            operator_size=1,
        )
        self.assertEqual(len(provider.calls), 8)
        self.assertEqual(result["evaluations"], 14)
        self.assertTrue(result["pareto"])
        self.assertIn("new combine", provider.calls[3])
        self.assertIn("new paraphrase", provider.calls[5])
        self.assertTrue(all("<class>" in p.prompt for p in result["population"]))

    async def test_word_edits_and_required_token_rejection(self):
        agent = MOPO("Task", ScriptedProvider([]), None, ["a", "b"], fill, ["<class>"])
        agent.rng = random.Random(0)
        variants = await agent._word_variants("write <class>")
        self.assertEqual(sorted(len(text.split()) for text in variants), [1, 2, 3])
        self.assertTrue(all("<class>" in text for text in variants))
        with self.assertRaises(ValueError):
            await agent.combine("a", "b", "combine", provider=ScriptedProvider(['{"text":"lost"}']))

    def test_specialists_and_operator_credit(self):
        agent = MOPO("Task", ScriptedProvider([]), None, ["a", "b"], fill)
        agent.rng = random.Random(0)
        population = [
            Individual("left", (1, 0), ("combine", "good")),
            Individual("right", (0, 1)),
            Individual("middle", (0.6, 0.6)),
        ]
        selected = agent._survive(population, 1, 1)
        self.assertEqual({p.prompt for p in selected}, {"left", "right", "middle"})
        self.assertEqual(agent._credit(["bad", "good"], selected, "combine", 1), ["good"])
