"""EvoPROMPT's generic task context, GA/DE rules, and checked generation."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import Session

from evoprompt import EvoPrompt
from tests.providers import ScriptedProvider


async def length_score(text):
    return len(text)


class EvoPromptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "evoprompt/prompts"
        )
        self.root.start()
        self.addCleanup(self.root.stop)

    async def test_ga_global_elitism_and_parent_snapshot_for_unrelated_tasks(self):
        for task in (
            "Give detailed software installation instructions.",
            "Teach pruning fruit trees.",
        ):
            with self.subTest(task=task):
                provider = ScriptedProvider(
                    [
                        response
                        for p in ("x", "strongest", "middle")
                        for response in ("<prompt>crossed</prompt>", f"<prompt>{p}</prompt>")
                    ]
                )
                agent = EvoPrompt(task, provider, length_score)
                result = await agent.run(["aa", "bbb", "cccc"], algorithm="ga", iterations=1)
                self.assertEqual(
                    [p["prompt"] for p in result["population"]], ["strongest", "middle", "cccc"]
                )
                self.assertEqual(result["evaluations"], 6)
                self.assertEqual(result["optimizer_calls"], 6)
                for context in provider.calls:
                    self.assertIn(task, context)
                    self.assertNotIn("strongest", context)
                    self.assertNotIn("middle", context)
                for context in provider.calls[1::2]:
                    self.assertIn("Instruction: crossed", context)

    async def test_de_distinct_donors_snapshot_strict_target_replacement(self):
        provider = ScriptedProvider(
            [
                response
                for p in ("longest", "x", "dddd")
                for response in (
                    "donor differences",
                    "mutated fragments",
                    "<prompt>combined</prompt>",
                    f"<prompt>{p}</prompt>",
                )
            ]
        )
        result = await EvoPrompt("Improve instructions.", provider, length_score).run(
            ["aa", "bbb", "cccc"], algorithm="de", iterations=1, seed=7
        )
        self.assertEqual([p["prompt"] for p in result["population"]], ["longest", "bbb", "cccc"])
        self.assertEqual(result["optimizer_calls"], 12)
        self.assertEqual(result["evaluations"], 6)
        for target, context in zip(("aa", "bbb", "cccc"), provider.calls[::4]):
            donors = [
                line.split(": ", 1)[1]
                for line in context.splitlines()
                if line.startswith(("Prompt 1:", "Prompt 2:"))
            ]
            self.assertEqual(len(set(donors + [target])), 3)
            self.assertNotIn("longest", context)
        for context in provider.calls[1::4]:
            self.assertIn("donor differences", context)
        for context in provider.calls[2::4]:
            self.assertIn("mutated fragments", context)
            self.assertIn("Prompt 3: cccc", context)
        for target, context in zip(("aa", "bbb", "cccc"), provider.calls[3::4]):
            self.assertIn("Prompt 1: combined", context)
            self.assertIn(f"Prompt 2: {target}", context)

    async def test_roulette_cache_and_tie_incumbents(self):
        async def score(text):
            return int("accessible" in text)

        provider = ScriptedProvider(["<prompt>accessible</prompt>"] * 12)
        result = await EvoPrompt("Make an accessible interface.", provider, score).run(
            ["plain", "neutral", "accessible"], algorithm="ga", iterations=2
        )
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(result["cache_hits"], 6)
        for context in provider.calls[::2]:
            self.assertIn("Prompt 1: accessible", context)
            self.assertIn("Prompt 2: accessible", context)

        async def zero(text):
            return 0

        contexts = []
        for _ in range(2):
            provider = ScriptedProvider(["<prompt>new</prompt>"] * 8)
            result = await EvoPrompt("Task.", provider, zero).run(
                ["old", "kept"], algorithm="ga", iterations=2, seed=9, cache=False
            )
            self.assertEqual([p["prompt"] for p in result["population"]], ["old", "kept"])
            self.assertEqual(result["evaluations"], 6)
            self.assertEqual(result["cache_hits"], 0)
            contexts.append(provider.calls)
        self.assertEqual(*contexts)

    async def test_variations_session_and_checked_methods(self):
        default = ScriptedProvider([])
        supplied = ScriptedProvider(["<prompt>bb</prompt>", "<prompt>ccc</prompt>"])
        agent = EvoPrompt("Summarize scientific findings.", default, length_score)
        result = await agent.run(
            ["a"], population_size=3, iterations=0, session=Session(provider=supplied)
        )
        self.assertEqual(result["best"]["prompt"], "ccc")
        self.assertEqual(result["optimizer_calls"], 2)
        self.assertEqual(default.calls, [])
        for context in supplied.calls:
            self.assertIn(agent.task, context)
            self.assertIn("Instruction: a", context)
        self.assertEqual(
            await agent.variation(
                "a", provider=ScriptedProvider(["Explanation\n<prompt> hello\nworld </prompt>"])
            ),
            "hello\nworld",
        )
        context = await EvoPrompt.crossover.render(agent, "first", "second")
        self.assertIn(agent.task, context)
        self.assertIn("Prompt 2: second", context)

    async def test_rejections_happen_before_scoring_offspring(self):
        for bad in (
            "unmarked",
            "<prompt> </prompt>",
            "<prompt>x</prompt><prompt>y</prompt>",
            "<prompt><prompt>x</prompt>",
        ):
            with self.subTest(bad=bad):
                provider = ScriptedProvider([bad])
                evaluated = []

                async def evaluate(text):
                    evaluated.append(text)
                    return len(text)

                agent = EvoPrompt("Task.", provider, evaluate)
                with self.assertRaises(ValueError) as caught:
                    await agent.run(["a", "bb"], algorithm="ga", iterations=1)
                self.assertIn(repr(bad), str(caught.exception))
                self.assertEqual(evaluated, ["a", "bb"])
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual(agent.optimizer_calls, 1)
                self.assertEqual(agent.evaluations, 2)

    async def test_fitness_and_transport(self):
        provider = ScriptedProvider([])
        for value in (-1, math.nan, math.inf):

            async def invalid(text):
                return value

            with self.subTest(value=value), self.assertRaises(ValueError):
                await EvoPrompt("Task.", provider, invalid).run(["a", "b", "c"], iterations=0)
        self.assertEqual(provider.calls, [])
        failing = ScriptedProvider([RuntimeError("transport failed")])
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            await EvoPrompt("Task.", failing, length_score).run(
                ["a", "b"], algorithm="ga", iterations=1
            )
        self.assertEqual(len(failing.calls), 1)

    async def test_repeated_runs_reset_search_state(self):
        provider = ScriptedProvider(["<prompt>longer</prompt>"] * 8)
        evaluated = []

        async def evaluate(text):
            evaluated.append(text)
            return len(text)

        agent = EvoPrompt("Expand instructions.", provider, evaluate)
        first = await agent.run(["a", "bb"], algorithm="ga", iterations=1)
        second = await agent.run(["a", "bb"], algorithm="ga", iterations=1)
        self.assertEqual(first, second)
        self.assertEqual(evaluated, ["a", "bb", "longer"] * 2)

    async def test_population_size_selects_best_seeds(self):
        result = await EvoPrompt("Task.", ScriptedProvider([]), length_score).run(
            ["a", "bb", "ccc", "dddd"], population_size=3, iterations=0
        )
        self.assertEqual(len(result["population"]), 3)
        self.assertEqual([p["prompt"] for p in result["population"]], ["dddd", "ccc", "bb"])
        self.assertEqual(result["evaluations"], 4)

    async def test_failed_operation_and_evaluation_attempts_remain_inspectable(self):
        for algorithm, responses, calls in (
            ("ga", ["<prompt>crossed</prompt>", "bad mutation"], 2),
            ("de", ["difference", "  "], 2),
            ("de", ["difference", "changed", "bad combination"], 3),
            ("de", ["difference", "changed", "<prompt>combined</prompt>", "bad trial"], 4),
        ):
            with self.subTest(algorithm=algorithm, calls=calls):
                agent = EvoPrompt("Task.", ScriptedProvider(responses), length_score)
                with self.assertRaises(ValueError) as caught:
                    await agent.run(["a", "bb", "ccc"], algorithm=algorithm, iterations=1)
                self.assertIn(repr(responses[-1]), str(caught.exception))
                self.assertEqual(agent.optimizer_calls, calls)
                self.assertEqual(agent.evaluations, 3)

        async def failing_score(text):
            raise RuntimeError("evaluation failed")

        agent = EvoPrompt("Task.", ScriptedProvider([]), failing_score)
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            await agent.run(["a", "bb"], algorithm="ga")
        self.assertEqual(agent.evaluations, 1)
        self.assertEqual(agent.optimizer_calls, 0)


if __name__ == "__main__":
    unittest.main()
