import random
import unittest
from pathlib import Path
from unittest.mock import patch

from erm import ERM, Exemplar, Exemplars, Feedbacks, Memory
from tests.providers import ScriptedProvider


class ERMTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "erm/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_verified_factory_feedback_rewards_and_memory_forgetting(self):
        async def failures(text):
            return [{"question": "q", "answer": "a", "output": "wrong"}]

        async def verify(exemplar, source):
            return exemplar.rationale == "valid solution"

        async def evaluate(text, examples):
            return {"seed": 0, "better": 2, "weak": 1, "memory-weak": 0}[text] + bool(examples)

        good = Exemplar(question="q", answer="a", rationale="valid solution")
        bad = Exemplar(question="q", answer="wrong", rationale="invalid")
        provider = ScriptedProvider(
            [
                Exemplars(exemplars=[bad, good]),
                Feedbacks(feedbacks=["useful"]),
                "better",
                Exemplars(exemplars=[good]),
                Feedbacks(feedbacks=["unhelpful"]),
                "weak",
                "memory-weak",
            ]
        )
        agent = ERM("Task", provider, evaluate, failures, verify, lambda a, b: float(a == b), ["q"])
        result = await agent.run(
            "seed", iterations=2, beam_size=1, beta=1, threshold=0.5, replacement=0
        )
        self.assertEqual(result["best"].prompt, "better")
        self.assertEqual(result["evaluations"], 8)
        self.assertEqual(result["optimizer_calls"], 7)
        self.assertEqual(result["feedback_memory"], [])
        self.assertEqual([row["stored"] for row in result["factory_history"]], [False, True, False])
        self.assertEqual([row["stored"] for row in result["feedback_history"]], [True, False])
        self.assertIn("useful", provider.calls[-1])
        self.assertIn("valid solution", provider.calls[4])
        self.assertEqual(agent.retrieve("q")[0], good)

    async def test_inference_priority_similarity_and_exemplar_forgetting(self):
        async def evaluate(text, examples):
            return 1

        async def failures(text):
            return []

        async def verify(exemplar, source):
            return True

        agent = ERM(
            "Task",
            ScriptedProvider([]),
            evaluate,
            failures,
            verify,
            lambda query, text: {"relevant": 1, "irrelevant": 0.01}[text],
            [],
        )
        agent.rng, agent.feedback_memory = random.Random(0), []
        agent.exemplar_memory = [
            Memory(Exemplar(question="irrelevant", answer="a", rationale="r"), 1),
            Memory(Exemplar(question="relevant", answer="b", rationale="r"), 0.5),
        ]
        self.assertEqual(agent.retrieve("query", count=1)[0].question, "relevant")
        agent._reward(agent.exemplar_memory[:1], False, 1, 0.2)
        self.assertEqual(len(agent.exemplar_memory), 1)

    async def test_unknown_and_wrong_reasoning_exemplars_never_enter_memory(self):
        async def evaluate(text, examples):
            return 1

        async def failures(text):
            return [{"question": "q", "answer": "a"}]

        async def verify(exemplar, source):
            return False

        provider = ScriptedProvider(
            [
                Exemplars(
                    exemplars=[
                        Exemplar(question="unknown", answer="a", rationale="r"),
                        Exemplar(question="q", answer="a", rationale="bad reasoning"),
                    ]
                ),
                Feedbacks(feedbacks=[]),
            ]
        )
        result = await ERM("Task", provider, evaluate, failures, verify, lambda a, b: 0, ["q"]).run(
            "seed", iterations=1
        )
        self.assertEqual(result["exemplar_memory"], [])
        self.assertTrue(all(not row["verified"] for row in result["factory_history"]))

    async def test_parent_without_training_failures_stays_in_beam(self):
        async def evaluate(text, examples):
            return {"seed": 0, "strong": 3, "weak": 1}[text]

        async def failures(text):
            return [] if text == "strong" else [{"question": "q", "answer": "a"}]

        async def verify(exemplar, source):
            return True

        provider = ScriptedProvider(
            [
                Exemplars(exemplars=[]),
                Feedbacks(feedbacks=["f1", "f2"]),
                "strong",
                "weak",
                Exemplars(exemplars=[]),
                Feedbacks(feedbacks=[]),
            ]
        )
        result = await ERM(
            "Task", provider, evaluate, failures, verify, lambda a, b: float(a == b), []
        ).run("seed", iterations=2, beam_size=2, memory_period=3)
        self.assertEqual([item.prompt for item in result["population"]], ["strong", "weak"])
        self.assertEqual(result["evaluations"], 8)


if __name__ == "__main__":
    unittest.main()
