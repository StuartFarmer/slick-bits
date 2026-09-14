import unittest
from pathlib import Path

import numpy as np
from scipy.special import log_softmax, softmax
from slick import prompts

from tests.providers import ScriptedProvider
from tuna import TUNA, Example, TeacherResponse, ranking_loss


class TUNATests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.old_root = prompts.TEMPLATE_ROOT
        prompts.TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "tuna" / "prompts"

    def tearDown(self):
        prompts.TEMPLATE_ROOT = self.old_root

    def test_rank_gap_objective_gradient(self):
        scores = np.array([-0.8, -0.9, -0.7])
        _, gradient, dr = ranking_loss(scores, -1.0, 0.3, 0.7)
        for i in range(3):
            plus, minus = scores.copy(), scores.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            expected = (
                ranking_loss(plus, -1.0, 0.3, 0.7)[0] - ranking_loss(minus, -1.0, 0.3, 0.7)[0]
            ) / 2e-6
            self.assertAlmostEqual(gradient[i], expected)
        self.assertEqual(dr, -0.7)

    async def test_contextual_generation_observes_first_stage_update(self):
        teacher_count, generated_parameters = 0, []

        async def teacher(query, seed):
            nonlocal teacher_count
            teacher_count += 1
            return TeacherResponse(
                "good" if teacher_count % 2 else "bad",
                np.array([-0.1 if teacher_count % 2 else -2.0]),
            )

        async def generate(parameters, query, temperature, seed):
            generated_parameters.append(parameters.copy())
            return "good" if len(generated_parameters) % 2 else "bad"

        async def forward(parameters, query, response):
            index = 0 if response == "good" else 1
            gradient = -softmax(parameters)
            gradient[index] += 1
            return np.array([log_softmax(parameters)[index]]), gradient[None, :]

        agent = TUNA("task", ScriptedProvider(['{"order": [0, 1]}']), teacher, generate, forward)
        result = await agent.run(
            np.zeros(2),
            [Example("q", "good")],
            candidates=2,
            probabilistic_rate=0.1,
            contextual_rate=0.1,
        )
        self.assertGreater(generated_parameters[0][0], generated_parameters[0][1])
        self.assertEqual(result["probabilistic_rankings"][0][1], ["good", "bad"])
        self.assertEqual(result["ranking_calls"], 1)

    async def test_rejects_duplicate_generated_ranking(self):
        agent = TUNA("task", ScriptedProvider(['{"order": [0, 0]}']), None, None, None)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            await agent.rank("q", ["a", "b"], provider=agent.provider)
