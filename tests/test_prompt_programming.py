"""Check prompt composition and counterfactual selection through real Slick calls."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_programming import PromptProgrammer
from tests.providers import ScriptedProvider


class PromptProgrammingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "prompt_programming/prompts"
        patcher = patch("slick.prompts.TEMPLATE_ROOT", root)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_multipart_injection_preserves_bytes_and_evaluates_only_answer(self):
        provider = ScriptedProvider([" first", " second", "  artifact\n"])
        evaluated = []

        async def evaluate(answer):
            evaluated.append(answer)
            return 0.75

        agent = PromptProgrammer("arbitrary task", provider, evaluate)
        result = await agent.run(fragments=("Plan:", "\nCheck:", "\nOutput:"))
        self.assertEqual(result.answer, "  artifact\n")
        self.assertEqual(result.fills, (" first", " second"))
        self.assertEqual(result.score, 0.75)
        self.assertEqual(evaluated, ["  artifact\n"])
        self.assertEqual(
            provider.calls,
            [
                "arbitrary task\nPlan:",
                "arbitrary task\nPlan: first\nCheck:",
                "arbitrary task\nPlan: first\nCheck: second\nOutput:",
            ],
        )
        self.assertEqual(result.calls, 4)
        self.assertEqual(result.transcript, provider.calls[-1] + result.answer)

    async def test_counterfactual_uses_global_maximum_then_discards_tail(self):
        provider = ScriptedProvider([" a b c d", "answer"])
        seen = []
        scores = iter([-math.inf, -2.0, -4.0, -0.5, -0.5])

        async def score_suffix(prefix, suffix):
            seen.append((prefix, suffix))
            return next(scores)

        agent = PromptProgrammer(
            "task",
            provider,
            score_suffix=score_suffix,
            token_boundaries=lambda text: (0, 2, 4, 6, 8),
        )
        result = await agent.run(fragments=("Seed:", "\nVerdict:"))
        self.assertEqual(result.fills, (" a b c",))
        self.assertEqual(provider.calls[-1], "task\nSeed: a b c\nVerdict:")
        self.assertEqual(seen[-1], ("task\nSeed: a b c d", "\nVerdict:"))
        self.assertEqual(result.calls, 7)

    async def test_trailing_space_reaches_generation_and_scoring_unchanged(self):
        provider = ScriptedProvider(["a b", "done"])

        async def score_suffix(prefix, suffix):
            self.assertEqual(prefix, " task\nSeed: a ")
            self.assertEqual(suffix, "Verdict: ")
            return -1.0

        agent = PromptProgrammer(
            " task",
            provider,
            score_suffix=score_suffix,
            token_boundaries=lambda text: (2,),
        )
        await agent.run(fragments=("Seed: ", "Verdict: "))
        self.assertEqual(provider.calls, [" task\nSeed: ", " task\nSeed: a Verdict: "])

    async def test_simple_colon_zero_and_few_shot_and_other_operations(self):
        provider = ScriptedProvider(["hello", "goodbye", "answer", "explanation"])
        agent = PromptProgrammer("bonjour", provider)
        await agent.run(mode="demonstration", input_label="French", output_label="English")
        self.assertEqual(provider.calls[-1], "French: bonjour\nEnglish:")
        await agent.run(
            mode="demonstration",
            examples=(("au revoir", "goodbye"),),
            input_label="French",
            output_label="English",
        )
        self.assertEqual(
            provider.calls[-1], "French: au revoir\nEnglish: goodbye\n\nFrench: bonjour\nEnglish:"
        )
        await agent.run(mode="direct")
        self.assertEqual(provider.calls[-1], "bonjour\nAnswer:")
        await agent.run(mode="proxy", proxy="a patient teacher")
        self.assertIn("a patient teacher", provider.calls[-1])
        self.assertIn("bonjour", provider.calls[-1])
        self.assertEqual(len(agent.calls), 1)

    async def test_default_serialization_and_failure_records(self):
        agent = PromptProgrammer("question", ScriptedProvider([" analysis", "answer"]))
        await agent.run()
        self.assertEqual(
            agent.calls[0]["prompt"],
            "question\nLet's solve this problem by splitting it into steps.",
        )
        self.assertEqual(
            agent.calls[1]["prompt"],
            "question\nLet's solve this problem by splitting it into steps."
            " analysis\nThus, the correct answer is",
        )
        agent = PromptProgrammer("q", ScriptedProvider([" \n"]))
        with self.assertRaisesRegex(ValueError, "blank"):
            await agent.run(mode="direct")
        self.assertEqual(agent.calls[0]["response"], " \n")
        self.assertIn("error", agent.calls[0])
        agent = PromptProgrammer("q", ScriptedProvider([OSError("transport")]))
        with self.assertRaises(OSError):
            await agent.run()
        self.assertEqual(len(agent.calls), 1)
        self.assertIn("transport", agent.calls[0]["error"])

    async def test_budget_and_invalid_measured_scores_stop_without_retry(self):
        agent = PromptProgrammer("q", ScriptedProvider([" fill", "answer"]))
        with self.assertRaisesRegex(RuntimeError, "call budget"):
            await agent.run(max_calls=1)
        self.assertEqual(len(agent.calls), 1)
        for bad in (math.nan, math.inf, 0.5, -math.inf):

            async def score_suffix(prefix, suffix):
                return bad

            agent = PromptProgrammer(
                "q",
                ScriptedProvider([" fill"]),
                score_suffix=score_suffix,
                token_boundaries=lambda text: (0, len(text)),
            )
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await agent.run()
            self.assertNotIn("answer", [call["operation"] for call in agent.calls])

    async def test_scoring_and_evaluation_share_budget_and_propagate_errors(self):
        async def score_suffix(prefix, suffix):
            return -1.0

        async def evaluate(answer):
            return math.nan

        agent = PromptProgrammer(
            "q",
            ScriptedProvider([" fill", "answer"]),
            token_boundaries=lambda text: (0, len(text)),
            score_suffix=score_suffix,
        )
        with self.assertRaisesRegex(RuntimeError, "call budget"):
            await agent.run(max_calls=2)
        self.assertEqual([call["operation"] for call in agent.calls], ["fill", "score_suffix"])
        agent = PromptProgrammer("q", ScriptedProvider(["answer"]), evaluate)
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run(mode="direct")
        self.assertIn("error", agent.calls[-1])
        self.assertEqual(len(agent.calls), 2)

    async def test_empty_prefix_and_full_prefix_are_candidate_boundaries(self):
        for scores, expected in (([-1.0, -2.0], ""), ([-2.0, -1.0], " fill")):
            values = iter(scores)

            async def score_suffix(prefix, suffix):
                return next(values)

            agent = PromptProgrammer(
                "q",
                ScriptedProvider([" fill", "answer"]),
                token_boundaries=lambda text: (0, len(text)),
                score_suffix=score_suffix,
            )
            result = await agent.run(fragments=("Seed:", "\nVerdict:"))
            self.assertEqual(result.fills, (expected,))
            self.assertEqual(agent.calls[-1]["prompt"], "q\nSeed:" + expected + "\nVerdict:")


if __name__ == "__main__":
    unittest.main()
