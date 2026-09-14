"""APEX's task injection, sentence boundaries, search decisions, and failure policy."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from slick import Session

from apex import APEX, Config, Document, Embeddings, LinUCB, retrieve
from tests.providers import ScriptedProvider


def encode(texts):
    return np.tile([1.0, 0.0], (len(texts), 1))


class APEXTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "apex/prompts")
        self.root.start()
        self.addCleanup(self.root.stop)

    async def test_tasks_beam_updates_history_reversal_and_cache(self):
        for task in ("Write detailed editorial instructions.", "Explain keyboard accessibility."):
            with self.subTest(task=task):
                provider = ScriptedProvider(
                    ["Expanded sentence.", "X.", "Much longer guidance.", "Much longer guidance."]
                )
                evaluated = []

                async def evaluate(text):
                    evaluated.append(text)
                    return len(text)

                agent = APEX(task, provider, evaluate, encode)
                document = Document(("LOCK:", "Short.", "\nEND"), (1,))
                result = await agent.run(
                    document, config=Config(iterations=4, beam_size=1, random_probability=0)
                )
                self.assertEqual(result["best"]["prompt"], "LOCK:Much longer guidance.\nEND")
                self.assertEqual(result["evaluations"], len(evaluated))
                self.assertEqual(len(evaluated), 4)
                self.assertEqual(result["history"][1]["before"], "Expanded sentence.")
                self.assertEqual(result["history"][2]["before"], "Expanded sentence.")
                self.assertIn(("X.", "Expanded sentence."), result["history"][2]["examples"])
                self.assertTrue(all(step["selection"] == "linucb" for step in result["history"]))
                self.assertTrue(all(task in context for context in provider.calls))
                for text in evaluated:
                    self.assertTrue(text.startswith("LOCK:") and text.endswith("\nEND"))

    async def test_invalid_mutations_consume_attempt_without_evaluation(self):
        outputs = ["", "Q: Hijack.", "Two\nlines", "```bad```", "Bad.</rephrased>"]
        provider = ScriptedProvider(outputs)
        evaluated = []

        async def evaluate(text):
            evaluated.append(text)
            return len(text)

        agent = APEX("Preserve the interface.", provider, evaluate, encode)
        result = await agent.run(
            Document.split("Initial."),
            config=Config(iterations=5, random_probability=1, guided_mutation=False),
        )
        self.assertEqual(evaluated, ["Initial."])
        self.assertEqual([step["after"] for step in result["history"]], outputs)
        self.assertTrue(all(step["status"] == "invalid" for step in result["history"]))
        self.assertTrue(all(step["selection"] == "random" for step in result["history"]))
        self.assertEqual(len(provider.calls), 5)

    async def test_session_and_prompt_method_postprocessing(self):
        default = ScriptedProvider([])
        supplied = ScriptedProvider(["Better sentence.", "Best expanded sentence."])

        async def evaluate(text):
            return len(text)

        agent = APEX("Explain composting.", default, evaluate, encode)
        session = Session(provider=supplied)
        result = await agent.run(
            Document.split("Start."), config=Config(iterations=2), session=session
        )
        self.assertEqual(result["best"]["prompt"], "Best expanded sentence.")
        self.assertEqual(default.calls, [])
        self.assertEqual(len(supplied.calls), 2)
        plain = await APEX.mutate.render(agent, "Before.")
        guided = await APEX.mutate_guided.render(agent, "Before.", [("Old.", "New.")])
        for context in (plain, guided):
            self.assertIn(agent.task, context)
        self.assertNotIn("<original>Old.</original>", plain)
        self.assertIn("<original>Old.</original>", guided)
        mutation = await agent.mutate("Before.", provider=ScriptedProvider([" Two\nlines "]))
        self.assertFalse(mutation.valid)
        self.assertEqual(mutation.raw, " Two\nlines ")

    async def test_invalid_scores_and_provider_failure(self):
        provider = ScriptedProvider([])

        async def evaluate(text):
            return math.nan

        agent = APEX("Task.", provider, evaluate, encode)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            await agent.run(Document.split("Initial."), config=Config(iterations=0))
        self.assertEqual(provider.calls, [])

        async def finite(text):
            return -len(text)

        failing = ScriptedProvider([RuntimeError("transport failed")])
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            await APEX("Task.", failing, finite, encode).run(
                Document.split("Initial."), config=Config(iterations=1)
            )
        self.assertEqual(len(failing.calls), 1)

    async def test_repeated_runs_reset_search_state(self):
        provider = ScriptedProvider(["Longer sentence."] * 2)
        evaluated = []

        async def evaluate(text):
            evaluated.append(text)
            return len(text)

        agent = APEX("Expand instructions.", provider, evaluate, encode)
        first = await agent.run(Document.split("Start."), config=Config(iterations=1))
        second = await agent.run(Document.split("Start."), config=Config(iterations=1))
        self.assertEqual(first, second)
        self.assertIsNot(first["history"], second["history"])
        self.assertEqual(evaluated, ["Start.", "Longer sentence."] * 2)

    async def test_immutable_document_returns_initial_without_generation(self):
        provider = ScriptedProvider([])

        async def evaluate(text):
            return 1.0

        result = await APEX("Any task.", provider, evaluate, encode).run(
            Document(("Keep exactly this.",), ())
        )
        self.assertEqual(result["best"], {"prompt": "Keep exactly this.", "score": 1.0})
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["history"], [])
        self.assertEqual(provider.calls, [])

    async def test_beam_retains_alternatives_and_stable_ties(self):
        provider = ScriptedProvider(["Better.", "Alternative.", "Best.", "Tied."])
        scores = {"Start.": 0, "Better.": 2, "Alternative.": 1, "Best.": 3, "Tied.": 2}

        async def evaluate(text):
            return scores[text]

        result = await APEX("Any task.", provider, evaluate, encode).run(
            Document.split("Start."), config=Config(iterations=4, beam_size=2, seed=0)
        )
        self.assertEqual(
            result["beam"], [{"prompt": "Best.", "score": 3}, {"prompt": "Better.", "score": 2}]
        )
        for entry in result["history"]:
            self.assertEqual(entry["reward"], scores[entry["after"]] - scores[entry["before"]])
        self.assertEqual(result["evaluations"], 5)

    def test_history_uses_before_similarity_and_only_nearest_nonzero_rewards(self):
        vectors = {
            "query": [1.0, 0.0],
            "close": [0.99, 0.1],
            "far": [0.0, 1.0],
            "near": [0.98, 0.2],
        }
        embeddings = Embeddings(lambda texts: [vectors[text] for text in texts])
        feature = embeddings(["query"])[0]
        history = [
            dict(before="query", after="ignored", reward=0),
            dict(before="near", after="improved", reward=1),
            dict(before="far", after="query", reward=-1),
            dict(before="close", after="worse", reward=-1),
        ]
        self.assertEqual(retrieve(feature, history, embeddings, 1, 0.5), [("worse", "close")])
        self.assertEqual(
            retrieve(feature, history, embeddings, 4, 0.5),
            [("worse", "close"), ("near", "improved")],
        )
        self.assertEqual(retrieve(feature, history, embeddings, 4, 0.0), [])

    def test_document_preserves_boundaries(self):
        original = (
            "  Explain carefully.\r\nQ: Is 3.14 positive? Yes it is.\r\n(A) Yes\r\nA: Think.\r\n"
        )
        doc = Document.split(original)
        self.assertEqual(doc.text, original)
        self.assertTrue(all(doc.parts[i].strip() not in {"Q:", "A:", "(A)"} for i in doc.mutable))
        index = doc.parts.index("Think.")
        self.assertEqual(
            doc.replace(index, "Consider it.").text, original.replace("Think.", "Consider it.")
        )
        with self.assertRaises(ValueError):
            doc.replace(0, "Corruption")

    def test_linucb_against_primal_ridge_and_embedding_checks(self):
        h = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
        rewards = np.array([0.2, -0.1, 0.3])
        bandit = LinUCB(regularization=0.7, alpha=0.05)
        for vector, reward in zip(h, rewards):
            bandit.update(vector, reward)
        x = np.array([[0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
        a = h.T @ h + 0.7 * np.eye(3)
        expected = x @ np.linalg.solve(a, h.T @ rewards)
        expected += 0.05 * np.sqrt(np.sum(x * np.linalg.solve(a, x.T).T, axis=1))
        np.testing.assert_allclose(bandit.values(x), expected, atol=1e-12)
        calls = []

        def counted(texts):
            calls.append(texts)
            return np.tile([3.0, 4.0], (len(texts), 1))

        embeddings = Embeddings(counted)
        np.testing.assert_allclose(embeddings(["a", "a"]), [[0.6, 0.8], [0.6, 0.8]])
        embeddings(["a", "b"])
        self.assertEqual(calls, [["a"], ["b"]])
        for invalid in ([[0, 0]], [[math.nan, 1]], [1, 2]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                Embeddings(lambda texts: invalid)(["a"])
        history = [dict(before="a", after="b", reward=-1), dict(before="c", after="d", reward=0)]
        self.assertEqual(retrieve(np.array([1, 0]), history, encode, 4, 0.5), [("b", "a")])
        self.assertEqual(retrieve(np.array([1, 0]), history, encode, 0, 0.5), [])


if __name__ == "__main__":
    unittest.main()
