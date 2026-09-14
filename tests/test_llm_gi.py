"""Offline checks for Gin-style LLM mutation and best-first local search."""

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts

from llm_gi import CandidateRejected, Evaluation, GeneticImprovement, Target
from tests.providers import ScriptedProvider


def fenced(content, label="text"):
    return f"```{label}\n{content}\n```"


def validate_number(content):
    try:
        float(content)
    except ValueError as exc:
        raise CandidateRejected("not a number") from exc


async def evaluate_number(content):
    return Evaluation(float(content))


class GeneticImprovementTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "llm_gi/prompts"
        root_patch = patch.object(prompts, "TEMPLATE_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def agent(self, responses=(), **kwargs):
        return GeneticImprovement(
            "Find a smaller number", ScriptedProvider(responses), evaluate_number, **kwargs
        )

    async def test_local_budget_strict_acceptance_and_incumbent_parent(self):
        agent = self.agent(map(fenced, [8, 9, "08", 7]))
        best = await agent.run("10", budget=5)
        self.assertEqual((best.content, best.fitness), ("7\n", 7))
        self.assertEqual(agent.baseline.fitness, 10)
        self.assertEqual(len(agent.attempts), 5)
        self.assertEqual(agent.evaluations, 5)
        self.assertEqual(len(agent.provider.calls), 4)
        self.assertEqual([r["parent"] for r in agent.attempts[1:]], ["10", "8\n", "8\n", "8\n"])
        self.assertEqual([r["accepted"] for r in agent.attempts], [True, True, False, False, True])

    async def test_random_sampling_uses_original_and_all_attempt_slots(self):
        agent = self.agent(map(fenced, [8, 9, 8]))
        best = await agent.run("10", mode="random", budget=3)
        self.assertEqual(best.fitness, 8)
        self.assertIsNone(agent.baseline)
        self.assertEqual(agent.evaluations, 3)  # Duplicates are measured again.
        self.assertEqual([r["parent"] for r in agent.attempts], ["10"] * 3)
        self.assertEqual([r["unique"] for r in agent.attempts], [True, True, False])

    async def test_first_valid_labeled_block_only_is_evaluated(self):
        response = "\n".join([fenced(1, "python"), fenced("bad"), fenced(8), fenced(2)])
        agent = self.agent([response], validate=validate_number)
        best = await agent.run("10", budget=2)
        self.assertEqual(best.fitness, 8)
        self.assertEqual(agent.evaluations, 2)
        self.assertEqual(agent.attempts[-1]["raw_response"], response)
        self.assertEqual(len(agent.attempts[-1]["invalid_suggestions"]), 1)

    async def test_failed_first_valid_suggestion_does_not_try_the_next(self):
        async def evaluate(content):
            return Evaluation(float(content), passed=float(content) != 8)

        agent = GeneticImprovement(
            "Minimize", ScriptedProvider([fenced(8) + "\n" + fenced(2)]), evaluate
        )
        best = await agent.run("10", budget=2)
        self.assertEqual(best.fitness, 10)
        self.assertEqual(agent.evaluations, 2)
        self.assertEqual(agent.attempts[-1]["status"], "test_failed")

    async def test_invalid_compile_test_timeout_and_nonfinite_never_win(self):
        async def evaluate(content):
            content = content.strip()
            if content == "timeout":
                raise TimeoutError("deadline")
            return {
                "10": Evaluation(10),
                "compile": Evaluation(None, compiled=False, passed=False),
                "tests": Evaluation(0, passed=False),
                "nan": Evaluation(math.nan),
                "8": Evaluation(8),
            }[content]

        provider = ScriptedProvider(
            ["unfenced", *map(fenced, ["compile", "tests", "timeout", "nan", 8])]
        )
        agent = GeneticImprovement("Minimize", provider, evaluate)
        best = await agent.run("10", budget=7)
        self.assertEqual(best.fitness, 8)
        self.assertEqual(agent.evaluations, 6)
        self.assertEqual(
            [r["status"] for r in agent.attempts],
            [
                "baseline",
                "invalid",
                "compile_failed",
                "test_failed",
                "timeout",
                "nonfinite",
                "passed",
            ],
        )
        self.assertEqual(agent.attempts[1]["raw_response"], "unfenced")

    async def test_target_replacement_preserves_surrounding_content(self):
        async def evaluate(content):
            self.assertTrue(content.startswith("prefix[") and content.endswith("]suffix"))
            return Evaluation(float(content[7:-7]))

        agent = GeneticImprovement(
            "Minimize",
            ScriptedProvider([fenced(8)]),
            evaluate,
            targets=lambda content: [Target(7, len(content) - 7)],
        )
        best = await agent.run("prefix[10]suffix", budget=2)
        self.assertEqual(best.content, "prefix[8\n]suffix")
        self.assertNotIn("prefix[", agent.provider.calls[0])

    async def test_classic_callback_and_canonical_noops(self):
        def mutate(content, rng):
            return str(float(content) - rng.choice([1, 2]))

        agent = self.agent(classic={"STATEMENT": mutate}, fingerprint=lambda s: str(float(s)))
        best = await agent.run("10", operator="STATEMENT", budget=4, seed=3)
        self.assertLess(best.fitness, 10)
        self.assertEqual(agent.provider.calls, [])
        self.assertEqual(agent.evaluations, 4)
        first = best
        self.assertEqual(await agent.run("10", operator="STATEMENT", budget=4, seed=3), first)

        agent = self.agent([fenced("10.0")], fingerprint=lambda s: str(float(s)))
        best = await agent.run("10", budget=2)
        self.assertEqual(best.content, "10")
        self.assertEqual(agent.attempts[-1]["status"], "no_op")
        self.assertEqual(agent.evaluations, 1)

    async def test_provider_and_unexpected_evaluator_errors_propagate(self):
        agent = self.agent([RuntimeError("offline")])
        with self.assertRaisesRegex(RuntimeError, "offline"):
            await agent.run("10", budget=2)
        self.assertEqual(agent.attempts[-1]["status"], "error")

        async def broken(content):
            raise RuntimeError("evaluator bug")

        agent = GeneticImprovement("Minimize", ScriptedProvider([fenced(8)]), broken)
        with self.assertRaisesRegex(RuntimeError, "evaluator bug"):
            await agent.run("10", mode="random", budget=1)
        self.assertEqual(agent.attempts[-1]["status"], "error")
        self.assertEqual(agent.evaluations, 1)

    async def test_failed_baseline_aborts_and_empty_targets_consume_budget(self):
        async def failed(content):
            return Evaluation(None, passed=False)

        agent = GeneticImprovement("Minimize", ScriptedProvider([]), failed)
        with self.assertRaisesRegex(CandidateRejected, "baseline"):
            await agent.run("10", budget=2)
        self.assertEqual(len(agent.attempts), 1)
        agent = self.agent(targets=lambda content: [])
        best = await agent.run("10", budget=3)
        self.assertEqual(best.fitness, 10)
        self.assertEqual(agent.evaluations, 1)
        self.assertEqual(len(agent.attempts), 3)
        self.assertEqual(agent.provider.calls, [])

    async def test_every_template_renders_from_another_directory(self):
        agent = self.agent(
            context="project context",
            language="python",
            requirements="keep the interface",
            example_before="old",
            example_after="new",
        )
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for name in ("simple", "medium", "detailed"):
                    rendered = await getattr(GeneticImprovement, name).render(agent, "source")
                    self.assertIn("5", rendered)
                    self.assertIn("source", rendered)
                    if name != "simple":
                        self.assertIn("keep the interface", rendered)
                        self.assertIn("project context", rendered)
                        self.assertIn("python", rendered)
                    if name == "detailed":
                        self.assertIn("old", rendered)
                        self.assertIn("new", rendered)
                    parsed = Environment().parse((self.root / f"{name}.j2").read_text())
                    self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
            finally:
                os.chdir(previous)

    async def test_prompt_dispatch_and_default_budgets(self):
        for operator in ("SIMPLE", "MEDIUM", "DETAILED"):
            agent = self.agent([fenced(8)])
            best = await agent.run("10", operator=operator, budget=2)
            self.assertEqual(best.fitness, 8)
            self.assertEqual(len(agent.provider.calls), 1)
        for mode, count in (("local", 100), ("random", 1000)):
            agent = self.agent(targets=lambda content: [])
            await agent.run("10", mode=mode)
            self.assertEqual(len(agent.attempts), count)
            self.assertEqual(agent.evaluations, int(mode == "local"))


if __name__ == "__main__":
    unittest.main()
