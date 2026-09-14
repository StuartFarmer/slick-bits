"""EASE ordered representations, exact transport filtering and UCB updates."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ease import EASE, Configuration
from tests.providers import ScriptedProvider


def surrogate(theta, contexts):
    return contexts @ theta, contexts


class EASETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "ease/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_transport_cost_represents_coverage(self):
        agent = EASE("Task", ScriptedProvider([]), None, None, surrogate)
        agent.example_features = np.eye(2)
        agent.validation_features = np.eye(2)
        self.assertAlmostEqual(agent.transport_cost([0, 1]), 0)
        self.assertAlmostEqual(agent.transport_cost([1, 0]), 0)
        self.assertAlmostEqual(agent.transport_cost([0, 0]), 0.5)

    async def test_ordered_joint_configurations_training_and_batch_ucb(self):
        seen = []

        async def hidden(texts):
            seen.extend(texts)
            return np.array(
                [[1 + text.find("EX_A") / 100, 1 + text.find("EX_B") / 100] for text in texts]
            )

        async def evaluate(instruction, examples):
            return float(examples[0] == "EX_A") + float(instruction == "better")

        agent = EASE("Task", ScriptedProvider(["better"]), evaluate, hidden, surrogate)
        result = await agent.run(
            ["seed"],
            ["EX_A", "EX_B", "EX_C"],
            ["EX_A", "EX_B"],
            np.zeros(2),
            additional_instructions=1,
            shots=2,
            initial_samples=4,
            iterations=2,
            proposal_sets=4,
            retained_sets=2,
            domain_batches=2,
            training_steps=2,
            seed=0,
        )
        self.assertEqual(result["training_runs"], 3)
        self.assertEqual(len(result["acquisitions"]), 4)
        self.assertEqual(result["evaluations"] + result["cache_hits"], 6)
        self.assertEqual(result["optimizer_calls"], 1)
        self.assertTrue(np.all(result["u"] >= 0.1))
        self.assertTrue(all(len(set(r["configuration"].indices)) == 2 for r in result["records"]))
        vectors = await agent._represent(
            [Configuration("seed", (0, 1)), Configuration("seed", (1, 0))]
        )
        self.assertFalse(np.allclose(vectors[0], vectors[1]))
        self.assertTrue(any("better" in text for text in seen))

    async def test_blank_instruction_and_nonfinite_score_fail(self):
        async def hidden(texts):
            return np.ones((len(texts), 2))

        async def evaluate(instruction, examples):
            return np.nan

        agent = EASE("Task", ScriptedProvider([]), evaluate, hidden, surrogate)
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run(["seed"], ["EX_A"], ["EX_B"], np.zeros(2), shots=1, initial_samples=2)
        self.assertEqual(agent.evaluations, 1)
        with self.assertRaisesRegex(ValueError, "empty"):
            await agent.induce([], provider=ScriptedProvider([" "]))

    async def test_single_initial_observation_has_finite_normalization(self):
        async def hidden(texts):
            return np.ones((len(texts), 2))

        async def evaluate(instruction, examples):
            return 1

        result = await EASE("Task", ScriptedProvider([]), evaluate, hidden, surrogate).run(
            ["seed"],
            ["EX_A", "EX_B"],
            ["EX_A"],
            np.zeros(2),
            shots=1,
            initial_samples=1,
            iterations=1,
            proposal_sets=2,
            retained_sets=1,
            training_steps=1,
        )
        self.assertTrue(np.isfinite(result["theta"]).all())
        self.assertTrue(np.isfinite(result["u"]).all())
