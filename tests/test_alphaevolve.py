"""Deterministic checks of AlphaEvolve's edit, evaluation, and search boundaries."""

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts
from slick.providers import ProviderError

from alphaevolve import AlphaEvolve, Config, Evaluation, EvaluationStage, InvalidCandidate
from alphaevolve.edits import apply_diff, check_rewrite
from tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1] / "alphaevolve/prompts"
SOURCE = "fixed\n# EVOLVE-BLOCK-START\nvalue = 0\n# EVOLVE-BLOCK-END\nfixed end\n"


def diff(old, new):
    return f"<<<<<<< SEARCH\n{old}\n=====\n{new}\n>>>>>>> REPLACE\n"


class EditTests(unittest.TestCase):
    def test_exact_sequential_edits_and_immutable_skeleton(self):
        changed = apply_diff(SOURCE, diff("value = 0", "value = 1") + diff("1", "2"))
        self.assertEqual(changed, SOURCE.replace("0", "2"))
        for response in (
            diff("fixed", "changed"),
            diff("absent", "new"),
            diff("value = 0", "value = 0"),
            diff("value = 0", "x") + "<<<<<<< SEARCH\nbroken",
        ):
            with self.subTest(response=response), self.assertRaises(InvalidCandidate):
                apply_diff(SOURCE, response)
        with self.assertRaises(InvalidCandidate):
            apply_diff("same same", diff("same", "different"))
        with self.assertRaises(InvalidCandidate):
            apply_diff("aaa", diff("aa", "b"))

    def test_rewrites_preserve_multiple_blocks_and_support_other_languages(self):
        source = "head\n// EVOLVE-BLOCK-START\na\n// EVOLVE-BLOCK-END\nmid\n// EVOLVE-BLOCK-START\nb\n// EVOLVE-BLOCK-END\ntail"
        child = source.replace("\na\n", "\na new\n").replace("\nb\n", "\nb new\n")
        self.assertEqual(check_rewrite(source, child), child)
        with self.assertRaises(InvalidCandidate):
            check_rewrite(source, child.replace("mid", "other"))
        self.assertEqual(check_rewrite("whole file", "new file"), "new file")
        for child in ("", "whole file", "# EVOLVE-BLOCK-START\nx"):
            with self.assertRaises(InvalidCandidate):
                check_rewrite("whole file", child)


class AlphaEvolveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_feedback_evolution_and_rejection_budget(self):
        provider = ScriptedProvider(
            [diff("value = 0", "value = 1"), "bad diff", diff("fixed", "x")]
        )

        async def evaluate(content):
            return Evaluation({"quality": float("value = 1" in content)}, feedback="measured")

        agent = AlphaEvolve("arbitrary task", provider, evaluate, config=Config(islands=1))
        best = await agent.run(SOURCE, attempts=3, concurrency=1)
        self.assertEqual(best.metrics["quality"], 1)
        self.assertEqual(agent.generation_calls, 3)
        self.assertEqual(agent.evaluations, 2)
        self.assertEqual(len(agent.attempts), 3)
        self.assertEqual(agent.attempts[1]["raw"], "bad diff")
        self.assertTrue(all(row.get("error") for row in agent.attempts[1:]))
        self.assertIn("measured", provider.calls[1])
        self.assertIn("value = 1", provider.calls[1])

    async def test_cascade_prunes_and_nonfinite_metrics_cannot_enter_archive(self):
        expensive = []

        async def cheap(content):
            return Evaluation({"valid": float(content != "invalid")}, feedback="cheap")

        async def final(content):
            expensive.append(content)
            return Evaluation({"quality": float("nan") if content == "nan" else 1})

        agent = AlphaEvolve(
            "task",
            ScriptedProvider(["invalid", "nan", "good"]),
            final,
            stages=(EvaluationStage(cheap, {"valid": 1}),),
            config=Config(islands=1, mode="rewrite"),
        )
        result = await agent.run("seed", attempts=3, concurrency=1)
        self.assertEqual(result.content, "seed")
        self.assertEqual(expensive, ["seed", "nan", "good"])
        self.assertEqual(agent.evaluations, 7)
        self.assertEqual(len(agent.programs), 2)
        self.assertIn("cheap", result.feedback)

    async def test_multiobjective_cells_and_island_reset(self):
        values = {
            "seed": (0, 0, (0,)),
            "fast": (5, 1, (0,)),
            "small": (1, 5, (0,)),
            "novel": (0, 0, (1,)),
        }

        async def evaluate(content):
            speed, size, cell = values[content]
            return Evaluation({"speed": speed, "size": size}, cell=cell)

        agent = AlphaEvolve(
            "task",
            ScriptedProvider(["fast", "small", "novel"]),
            evaluate,
            config=Config(islands=1, mode="rewrite"),
        )
        await agent.run("seed", attempts=3, concurrency=1)
        self.assertEqual(agent.best_by_metric["speed"].content, "fast")
        self.assertEqual(agent.best_by_metric["size"].content, "small")
        self.assertEqual({p.content for p in agent.islands[0].values()}, {"fast", "small", "novel"})
        agent.islands.append(dict(agent.islands[0]))
        agent.reset_islands()
        self.assertEqual(len(agent.events), 1)
        self.assertEqual(agent.best_by_metric["speed"].content, "fast")

    async def test_async_workers_use_bounded_attempts_without_generation_barrier(self):
        gate = asyncio.Event()
        started = asyncio.Event()

        async def evaluate(content):
            if content == "slow":
                started.set()
                await gate.wait()
            if content == "fast":
                await started.wait()
            if content == "next":
                gate.set()
            return Evaluation({"score": 1})

        provider = ScriptedProvider(["slow", "fast", "next"])
        agent = AlphaEvolve("task", provider, evaluate, config=Config(mode="rewrite"))
        await asyncio.wait_for(agent.run("seed", attempts=3, concurrency=2), 2)
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(agent.evaluations, 4)
        self.assertEqual(len(agent.programs), 4)

    async def test_meta_prompts_are_evaluated_through_offspring(self):
        provider = ScriptedProvider(["one", "Try a distinct representation", "two"])

        async def evaluate(content):
            return Evaluation({"score": {"seed": 0, "one": 1, "two": 2}[content]})

        agent = AlphaEvolve(
            "task",
            provider,
            evaluate,
            config=Config(islands=1, mode="rewrite", meta_interval=2),
        )
        await agent.run("seed", attempts=2, concurrency=1)
        self.assertEqual(agent.generation_calls, 2)
        self.assertEqual(agent.meta_calls, 1)
        self.assertIn("Try a distinct representation", provider.calls[-1])
        self.assertEqual(agent.prompt_ideas[-1].uses, 1)
        self.assertGreater(agent.prompt_ideas[-1].reward, 0)

    async def test_unexpected_evaluator_errors_propagate_and_cancel_siblings(self):
        cancelled = asyncio.Event()
        started = asyncio.Event()

        async def evaluate(content):
            if content == "slow":
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            if content == "broken":
                await started.wait()
                raise RuntimeError("evaluator bug")
            return Evaluation({"score": 0})

        agent = AlphaEvolve(
            "task", ScriptedProvider(["slow", "broken"]), evaluate, config=Config(mode="rewrite")
        )
        with self.assertRaisesRegex(RuntimeError, "evaluator bug"):
            await agent.run("seed", attempts=2, concurrency=2)
        self.assertTrue(cancelled.is_set())

    async def test_rejected_evaluation_diagnostics_reach_next_prompt(self):
        async def evaluate(content):
            if content == "bad":
                return Evaluation(
                    {"score": 0}, feedback="specific diagnostic: wrong shape", error="invalid"
                )
            return Evaluation({"score": 1})

        provider = ScriptedProvider(["bad", "good"])
        agent = AlphaEvolve("task", provider, evaluate, config=Config(mode="rewrite"))
        await agent.run("seed", attempts=2, concurrency=1)
        self.assertIn("specific diagnostic: wrong shape", provider.calls[1])

    async def test_provider_errors_and_model_weights_have_no_hidden_retries(self):
        unused = ScriptedProvider([])
        selected = ScriptedProvider([ProviderError("offline"), TimeoutError("late"), "good"])

        async def evaluate(content):
            return Evaluation({"score": float(content == "good")})

        agent = AlphaEvolve(
            "task",
            unused,
            evaluate,
            ensemble=((unused, 0), (selected, 1)),
            config=Config(mode="rewrite", islands=2, reset_interval=1),
        )
        best = await agent.run("seed", attempts=3, concurrency=1)
        self.assertEqual(best.content, "good")
        self.assertEqual(unused.calls, [])
        self.assertEqual(agent.generation_calls, 3)
        self.assertEqual(agent.evaluations, 2)
        self.assertEqual(len(agent.events), 3)
        self.assertTrue(all(row["model"] == 1 for row in agent.attempts))
        self.assertEqual(
            [row["status"] for row in agent.attempts], ["rejected", "rejected", "evaluated"]
        )

    async def test_failed_meta_call_falls_back_and_preserves_raw_response(self):
        async def evaluate(content):
            return Evaluation({"score": float(content == "good")})

        agent = AlphaEvolve(
            "task",
            ScriptedProvider([" ", "good"]),
            evaluate,
            prompt_variants=(("chosen instruction", 1), ("excluded instruction", 0)),
            config=Config(mode="rewrite", meta_interval=1),
        )
        result = await agent.run("seed", attempts=1)
        self.assertEqual(result.content, "good")
        self.assertEqual(agent.meta_calls, 1)
        self.assertEqual(agent.attempts[0]["meta_raw"], " ")
        self.assertIn("blank", agent.attempts[0]["meta_error"])
        self.assertIn("chosen instruction", agent.attempts[0]["guidance"])
        self.assertNotIn("excluded instruction", agent.attempts[0]["guidance"])

    async def test_evaluation_timeout_and_changed_objectives_are_rejected(self):
        async def evaluate(content):
            if content == "slow":
                await asyncio.Event().wait()
            return Evaluation({"other" if content == "different" else "score": 1})

        agent = AlphaEvolve(
            "task",
            ScriptedProvider(["slow", "different"]),
            evaluate,
            config=Config(mode="rewrite", evaluation_timeout=0.02),
        )
        best = await agent.run("seed", attempts=2, concurrency=1)
        self.assertEqual(best.content, "seed")
        self.assertIn("TimeoutError", agent.attempts[0]["error"])
        self.assertIn("objective names", agent.attempts[1]["error"])
        self.assertEqual(agent.evaluations, 3)

    async def test_seed_failure_propagates_without_model_calls(self):
        async def evaluate(content):
            return Evaluation(error="seed is invalid")

        provider = ScriptedProvider([])
        agent = AlphaEvolve("task", provider, evaluate)
        with self.assertRaisesRegex(InvalidCandidate, "seed is invalid"):
            await agent.run("seed", attempts=0)
        self.assertEqual(provider.calls, [])

    async def test_every_prompt_renders_with_explicit_owner_and_no_conditionals(self):
        async def evaluate(content):
            return Evaluation({"score": 1})

        agent = AlphaEvolve("arbitrary problem", ScriptedProvider([]), evaluate)
        parent = await agent.run("seed", attempts=0)
        for method in (AlphaEvolve.mutate, AlphaEvolve.rewrite):
            rendered = await method.render(agent, parent, [], "guidance", [])
            self.assertIn("arbitrary problem", rendered)
            self.assertIn("seed", rendered)
            self.assertIn("guidance", rendered)
        rendered = await AlphaEvolve.evolve_prompt.render(agent, parent, [], [])
        self.assertIn("arbitrary problem", rendered)
        for path in ROOT.glob("*.j2"):
            parsed = Environment().parse(path.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
