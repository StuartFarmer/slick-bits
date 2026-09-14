"""Check journal GI decisions without models or execution of generated content."""

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts

from llm_gi_2025 import CandidateRejected, Evaluation, GeneticImprovement, Target
from tests.providers import ScriptedProvider


def fenced(text, label="text"):
    return f"```{label}\n{text}```"


async def evaluate_number(content, method):
    return Evaluation(float(content))


class JournalGITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "llm_gi_2025/prompts"
        root_patch = patch.object(prompts, "TEMPLATE_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def agent(self, responses=(), **kwargs):
        return GeneticImprovement(
            "Find a smaller number", ScriptedProvider(responses), evaluate_number, **kwargs
        )

    async def test_artifact_neighborhood_strict_acceptance_and_budget(self):
        agent = self.agent(map(fenced, [8, 9, "08", 7]))
        best = (await agent.run("10", budget=5, neighborhood="artifact"))["artifact"]
        self.assertEqual((best.content, best.fitness), ("7", 7))
        self.assertEqual(len(best.patch), 2)
        self.assertEqual(agent.evaluations, 5)
        self.assertEqual(agent.optimizer_calls, 4)
        self.assertEqual([r["accepted"] for r in agent.history], [True, True, False, False, True])
        self.assertEqual([r["parent"] for r in agent.history[1:]], ["10", "8", "8", "8"])

    async def test_paper_removes_any_edit_and_replays_remaining_patch(self):
        def targets(content):
            split = content.index("|")
            return {"pair": {"left": Target(0, split), "right": Target(split + 1, len(content))}}

        scores = {"aa|bb": 10, "long|bb": 8, "long|x": 6, "aa|x": 4}

        async def evaluate(content, method):
            return Evaluation(scores[content])

        agent = GeneticImprovement(
            "Reduce cost", ScriptedProvider(map(fenced, ["long", "x"])), evaluate, targets=targets
        )
        # First add left, then add right, then remove the earlier left edit.
        with (
            patch("random.Random.choice", side_effect=["left", "right"]),
            patch("random.Random.random", side_effect=[0.2, 0.8]),
            patch("random.Random.randrange", return_value=0),
        ):
            best = (await agent.run("aa|bb", budget=4))["pair"]
        self.assertEqual(best.content, "aa|x")
        self.assertEqual(len(best.patch), 1)
        self.assertEqual(agent.optimizer_calls, 2)
        self.assertEqual(agent.history[-1]["operation"], "remove")

    async def test_random_uses_original_and_remeasures_duplicates_and_noops(self):
        agent = self.agent(map(fenced, [8, 8, 10]))
        best = (await agent.run("10", mode="random", budget=3))["artifact"]
        self.assertEqual(best.fitness, 8)
        self.assertEqual(agent.baselines, {})
        self.assertEqual(agent.evaluations, 3)
        self.assertEqual([r["parent"] for r in agent.history], ["10"] * 3)
        self.assertEqual([r["unique"] for r in agent.history], [True, False, True])

    async def test_first_parseable_fence_evaluated_once_even_if_tests_fail(self):
        def validate(content):
            if content == "bad":
                raise CandidateRejected("invalid syntax")

        async def evaluate(content, method):
            return Evaluation(float(content), passed=content != "8")

        response = "\n".join([fenced(1, "python"), fenced("bad"), fenced(8, ""), fenced(2)])
        agent = GeneticImprovement(
            "Reduce", ScriptedProvider([response]), evaluate, validate=validate
        )
        best = (await agent.run("10", budget=2))["artifact"]
        self.assertEqual(best.fitness, 10)
        self.assertEqual(agent.evaluations, 2)
        self.assertEqual(agent.history[-1]["raw_response"], response)
        self.assertEqual(agent.history[-1]["invalid_suggestions"], ["invalid syntax"])
        self.assertEqual(agent.history[-1]["status"], "test_failed")

    async def test_invalid_outcomes_consume_slots_and_raw_responses_survive(self):
        async def evaluate(content, method):
            if content == "timeout":
                raise TimeoutError("deadline")
            return {
                "10": Evaluation(10),
                "compile": Evaluation(compiled=False),
                "tests": Evaluation(0, passed=False),
                "nan": Evaluation(math.nan),
            }[content]

        agent = GeneticImprovement(
            "Reduce",
            ScriptedProvider(["unfenced", *map(fenced, ["compile", "tests", "timeout", "nan"])]),
            evaluate,
        )
        await agent.run("10", budget=6)
        self.assertEqual(
            [r["status"] for r in agent.history],
            ["baseline", "invalid", "compile_failed", "test_failed", "timeout", "nonfinite"],
        )
        self.assertEqual(agent.evaluations, 5)
        self.assertEqual(agent.history[1]["raw_response"], "unfenced")

    async def test_independent_hot_methods_and_uniform_random_method_selection(self):
        def targets(content):
            return {"a": {"block": Target(0, 2)}, "b": {"block": Target(3, len(content))}}

        async def evaluate(content, method):
            return Evaluation(float(content.split("|")[method == "b"]))

        agent = GeneticImprovement(
            "Reduce", ScriptedProvider(map(fenced, [8, 7])), evaluate, targets=targets
        )
        best = await agent.run("10|20", budget=2)
        self.assertEqual(best["a"].content, "8|20")
        self.assertEqual(best["b"].content, "10|7")
        self.assertEqual(agent.evaluations, 4)
        agent = GeneticImprovement(
            "Reduce", ScriptedProvider([fenced(6)]), evaluate, targets=targets
        )
        with patch("random.Random.choice", side_effect=["b", "block"]):
            best = await agent.run("10|20", mode="random", budget=1)
        self.assertIsNone(best["a"])
        self.assertEqual(best["b"].content, "10|6")

    async def test_provider_errors_propagate_and_baseline_failure_aborts(self):
        agent = self.agent([RuntimeError("offline")])
        with self.assertRaisesRegex(RuntimeError, "offline"):
            await agent.run("10", budget=2)
        self.assertEqual(agent.history[-1]["status"], "error")

        async def failed(content, method):
            return Evaluation(passed=False)

        agent = GeneticImprovement("Reduce", ScriptedProvider([]), failed)
        with self.assertRaisesRegex(CandidateRejected, "baseline"):
            await agent.run("10")
        self.assertEqual(agent.optimizer_calls, 0)

    async def test_templates_and_dispatch_from_another_directory(self):
        examples = {
            "SMALL_CHANGES": ["original", "copy", "delete", "replace", "swap"],
            "STRUCTURAL_CHANGES": ["original", "while", "stream", "indices"],
        }
        agent = self.agent([fenced(8)] * 3, examples=examples)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for name, style in [
                    ("basic", "BASIC"),
                    ("small_changes", "SMALL_CHANGES"),
                    ("structural_changes", "STRUCTURAL_CHANGES"),
                ]:
                    args = [] if style == "BASIC" else [examples[style]]
                    rendered = await getattr(GeneticImprovement, name).render(
                        agent, "source", *args
                    )
                    self.assertIn("5", rendered)
                    self.assertIn("source", rendered)
                    parsed = Environment().parse((self.root / f"{name}.j2").read_text())
                    self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
                    result = await agent.run("10", budget=2, prompt_style=style)
                    self.assertEqual(result["artifact"].fitness, 8)
            finally:
                os.chdir(previous)

    async def test_missing_target_on_replay_consumes_slot_without_evaluation(self):
        def targets(content):
            result = {"whole": Target(0, len(content))}
            if content.startswith("a"):
                result["suffix"] = Target(1, len(content))
            return {"method": result}

        async def evaluate(content, method):
            return Evaluation({"z": 10, "abc": 8, "ax": 6}[content])

        agent = GeneticImprovement(
            "Reduce", ScriptedProvider(map(fenced, ["abc", "x"])), evaluate, targets=targets
        )
        with (
            patch("random.Random.choice", side_effect=["whole", "suffix"]),
            patch("random.Random.random", side_effect=[0.2, 0.8]),
            patch("random.Random.randrange", return_value=0),
        ):
            result = await agent.run("z", budget=4)
        self.assertEqual(result["method"].content, "ax")
        self.assertEqual(agent.evaluations, 3)
        self.assertEqual(agent.history[-1]["status"], "invalid")
        self.assertIn("target disappeared", agent.history[-1]["error"])

    async def test_default_budgets_and_classic_mutation_need_no_provider(self):
        for mode, count in [("local", 100), ("random", 1000)]:
            agent = self.agent(targets=lambda content: {"artifact": {}})
            await agent.run("10", mode=mode)
            self.assertEqual(len(agent.history), count)
            self.assertEqual(agent.evaluations, int(mode == "local"))
            self.assertEqual(agent.optimizer_calls, 0)
        agent = self.agent(classic=lambda fragment, rng: str(float(fragment) - 1))
        result = await agent.run("10", budget=3, prompt_style="STATEMENT", neighborhood="artifact")
        self.assertEqual(result["artifact"].fitness, 8)
        self.assertEqual(agent.optimizer_calls, 0)
        self.assertEqual(agent.evaluations, 3)

    async def test_unexpected_validator_and_evaluator_errors_propagate(self):
        def broken(content):
            raise RuntimeError("parser bug")

        agent = self.agent([fenced(8)], validate=broken)
        with self.assertRaisesRegex(RuntimeError, "parser bug"):
            await agent.run("10", budget=2)
        self.assertEqual(agent.history[-1]["raw_response"], fenced(8))

        async def broken_evaluator(content, method):
            raise RuntimeError("evaluator bug")

        agent = GeneticImprovement("Reduce", ScriptedProvider([fenced(8)]), broken_evaluator)
        with self.assertRaisesRegex(RuntimeError, "evaluator bug"):
            await agent.run("10", mode="random", budget=1)
        self.assertEqual(agent.history[-1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
