"""Check exhaustive component edits, signed UCB credit, and batch boundaries."""

import unittest
from pathlib import Path
from unittest.mock import patch

from sprig import SPRIG, Individual
from tests.providers import ScriptedProvider


class SPRIGTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "sprig/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_additions_and_signed_delete_credit(self):
        batches = []

        async def evaluate(prompts, generation):
            batches.append((prompts, generation))
            return [text.count("help") for text in prompts]

        agent = SPRIG("Task", ScriptedProvider([]), evaluate, ["help", "harm"])
        result = await agent.run(rounds=2, beam_size=1, paraphrases=0)
        self.assertEqual(result["best"].components, ("help", "help"))
        self.assertTrue(all(value == 1 for value in result["credit"]["help"]))
        self.assertEqual([i for _, i in batches], [-1, 0, 1])
        self.assertEqual(agent._choose_components(1, 0), ["help"])

    async def test_swap_and_paraphrase_keep_component_boundaries(self):
        provider = ScriptedProvider(['{"texts":["A"]}', '{"texts":["B"]}'])
        agent = SPRIG("Task", provider, None, ["a", "b"])
        agent.credit, agent.origins = {"a": [], "b": []}, {"a": "a", "b": "b"}
        edits = await agent._expand([Individual(("a", "b"), 0)], 0, 1, 1)
        self.assertIn(("b", "a"), [e[0] for e in edits])
        self.assertIn(("A", "b"), [e[0] for e in edits])
        self.assertEqual(agent.origins["A"], "a")
        self.assertIn("Task", provider.calls[0])

    async def test_bad_batch_and_generated_content_propagate(self):
        async def evaluate(prompts, generation):
            return [float("nan")]

        agent = SPRIG("Task", ScriptedProvider([]), evaluate, ["x"])
        with self.assertRaises(ValueError):
            await agent.run()
        with self.assertRaises(ValueError):
            await agent.rephrase("x", 1, provider=ScriptedProvider(['{"texts":[" "]}']))
