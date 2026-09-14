"""Exercise GEPA's search decisions without a model service or benchmark."""

import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from gepa import GEPA, Candidate, Evaluation
from gepa.agent import pareto_weights
from tests.providers import ScriptedProvider


class GEPATests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "gepa/prompts"
        self.root = patch("slick.prompts.TEMPLATE_ROOT", root)
        self.root.start()
        self.addCleanup(self.root.stop)

    def test_pareto_keeps_specialists_and_counts_instance_wins(self):
        self.assertEqual(pareto_weights([(1, 0, 0), (1, 1, 0), (0, 0, 1)]), {1: 2, 2: 1})
        # Official implementation removes redundant win coverage, even when
        # no single score vector strictly dominates another.
        self.assertEqual(pareto_weights([(1, 1, 0), (1, 0, 1), (0, 1, 1)]), {1: 2, 2: 2})

    def test_pareto_preserves_official_tie_removal_order(self):
        scores = [
            [1, 0, 2, 2, 2, 3, 3],
            [2, 2, 2, 2, 0, 2, 2],
            [0, 0, 0, 2, 1, 1, 2],
            [1, 3, 0, 0, 3, 3, 2],
            [0, 3, 2, 2, 1, 2, 0],
            [3, 2, 0, 2, 1, 1, 2],
            [1, 0, 3, 0, 0, 0, 1],
            [2, 1, 2, 1, 0, 2, 0],
            [2, 0, 3, 2, 0, 2, 2],
            [1, 0, 1, 3, 1, 2, 1],
            [3, 1, 3, 0, 0, 1, 1],
            [0, 3, 1, 1, 3, 2, 2],
        ]
        self.assertEqual(pareto_weights(scores), {5: 1, 3: 3, 8: 1, 9: 1, 0: 2})

    async def test_mutation_round_robin_budget_and_held_out_separation(self):
        evaluated = []

        async def evaluate(prompts, item):
            evaluated.append((dict(prompts), item))
            score = sum(value == "improved" for value in prompts.values()) / 2
            return Evaluation(
                score,
                output="response",
                feedback=f"feedback:{item}",
                trace="tool output",
                module_feedback={"a": "fix module a"},
            )

        provider = ScriptedProvider(["```\nimproved\n```"] * 2)
        agent = GEPA("Any system with two prompts", provider, evaluate)
        result = await agent.run(
            {"a": "seed", "b": "seed"}, ["train"], ["held-out"], budget=7, minibatch_size=1
        )
        self.assertEqual(result["best"].prompts, {"a": "improved", "b": "improved"})
        self.assertEqual([c.parents for c in result["population"]], [(), (0,), (1,)])
        self.assertEqual([h["module"] for h in result["history"]], ["a", "b"])
        self.assertEqual(result["rollouts"], 7)
        self.assertEqual(result["reflection_calls"], 2)
        self.assertEqual(len(evaluated), 7)
        self.assertEqual(
            [item for _, item in evaluated],
            ["held-out", "train", "train", "held-out", "train", "train", "held-out"],
        )
        self.assertIn("fix module a", provider.calls[0])
        self.assertIn("tool output", provider.calls[0])
        self.assertNotIn("held-out", "".join(provider.calls))

    async def test_rejections_keep_incumbent_and_record_raw_responses(self):
        async def evaluate(prompts, item):
            return Evaluation(0.5)

        provider = ScriptedProvider(["bad response", "```\n   \n```", "```\nsame score\n```"])
        agent = GEPA("Task", provider, evaluate)
        result = await agent.run({"a": "seed"}, [0], [1], budget=7, minibatch_size=1)
        self.assertEqual(result["best"].prompts, {"a": "seed"})
        self.assertEqual(
            [h["status"] for h in result["history"]], ["invalid", "invalid", "not_improved"]
        )
        self.assertEqual(
            result["raw_responses"], ["bad response", "```\n   \n```", "```\nsame score\n```"]
        )
        self.assertEqual(result["rollouts"], 5)

    async def test_budget_reserves_full_validation_before_reflection(self):
        async def evaluate(prompts, item):
            return Evaluation(0)

        for budget in (2, 3, 4, 5):
            agent = GEPA("Task", ScriptedProvider([]), evaluate)
            result = await agent.run({"a": "seed"}, [0], [1, 2], budget=budget, minibatch_size=1)
            self.assertEqual(result["rollouts"], 2)
            self.assertEqual(result["reflection_calls"], 0)
        agent = GEPA("Task", ScriptedProvider([]), evaluate)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            await agent.run({"a": "seed"}, [0], [1, 2], budget=1)
        self.assertEqual(agent.rollouts, 0)

    async def test_errors_propagate_with_attempt_counts(self):
        for value in (math.nan, math.inf):

            async def evaluate(prompts, item):
                return Evaluation(value)

            agent = GEPA("Task", ScriptedProvider([]), evaluate)
            with self.assertRaisesRegex(ValueError, "finite"):
                await agent.run({"a": "seed"}, [0], [1], budget=4, minibatch_size=1)
            self.assertEqual(agent.rollouts, 1)

        async def evaluate(prompts, item):
            return Evaluation(0)

        agent = GEPA("Task", ScriptedProvider([RuntimeError("transport")]), evaluate)
        with self.assertRaisesRegex(RuntimeError, "transport"):
            await agent.run({"a": "seed"}, [0], [1], budget=4, minibatch_size=1)
        self.assertEqual(agent.rollouts, 2)
        self.assertEqual(agent.reflection_calls, 1)

    async def test_merge_combines_siblings_without_model_calls(self):
        evaluated = []

        async def evaluate(prompts, item):
            evaluated.append(item)
            return Evaluation(float(prompts[item] == "improved"))

        agent = GEPA("Task", ScriptedProvider([]), evaluate)
        await agent.run({"a": "seed", "b": "seed"}, ["a"], ["a", "b"], budget=2)
        agent.population.extend(
            [
                Candidate({"a": "improved", "b": "seed"}, (1, 0), (0,)),
                Candidate({"a": "seed", "b": "improved"}, (0, 1), (0,)),
            ]
        )
        agent.next_modules.extend([1, 0])
        agent.budget = 4
        self.assertTrue(await agent._try_merge())
        self.assertEqual(agent.population[-1].prompts, {"a": "improved", "b": "improved"})
        self.assertEqual(agent.population[-1].parents, (1, 2))
        self.assertEqual(agent.population[-1].scores, (1, 1))
        self.assertEqual(agent.next_modules[-1], 1)
        self.assertEqual(agent.rollouts, 4)
        self.assertEqual(agent.reflection_calls, 0)
        self.assertEqual(sorted(evaluated), ["a", "a", "b", "b"])
        self.assertFalse(await agent._try_merge())

    async def test_run_schedules_merge_for_complementary_lineages(self):
        async def evaluate(prompts, item):
            changed = [name for name, value in prompts.items() if value == "improved"]
            score = (
                len(changed) / 2
                if item == "train"
                else (float(item in changed) if changed else 0.5)
            )
            return Evaluation(score)

        provider = ScriptedProvider(["```\nimproved\n```"] * 2)
        result = await GEPA("Task", provider, evaluate).run(
            {"a": "seed", "b": "seed"},
            ["train"],
            ["a", "b"],
            budget=12,
            minibatch_size=1,
            max_merges=1,
            seed=3,
        )
        self.assertEqual(result["best"].prompts, {"a": "improved", "b": "improved"})
        self.assertEqual(result["best"].parents, (1, 2))
        self.assertEqual(result["merge_attempts"], 1)
        self.assertEqual(result["rollouts"], 12)
        self.assertEqual(result["reflection_calls"], 2)

    async def test_merge_rejects_regression_and_finishes_validation_once(self):
        for merged_score in (0.0, 1.0):
            calls = []

            async def evaluate(prompts, item):
                calls.append(item)
                return Evaluation(merged_score if "improved" in prompts.values() else 0)

            agent = GEPA("Task", ScriptedProvider([]), evaluate)
            await agent.run({"a": "seed", "b": "seed"}, [0], list(range(8)), budget=8)
            agent.population.extend(
                [
                    Candidate({"a": "improved", "b": "seed"}, (1,) * 4 + (0,) * 4, (0,)),
                    Candidate({"a": "seed", "b": "improved"}, (0,) * 4 + (1,) * 4, (0,)),
                ]
            )
            agent.next_modules.extend([1, 0])
            agent.budget = 16
            accepted = await agent._try_merge()
            self.assertEqual(accepted, bool(merged_score))
            self.assertEqual(agent.merge_attempts, 1)
            self.assertEqual(agent.rollouts, 16 if accepted else 13)
            self.assertEqual(len(agent.population), 4 if accepted else 3)
            self.assertEqual(len(calls[8:]), len(set(calls[8:])))

    async def test_merge_resolves_conflicts_from_better_parent(self):
        async def evaluate(prompts, item):
            return Evaluation(0)

        agent = GEPA("Task", ScriptedProvider([]), evaluate)
        await agent.run({"a": "seed", "b": "seed", "shared": "seed"}, [0], [1, 2], budget=2)
        agent.population.extend(
            [
                Candidate({"a": "left", "b": "seed", "shared": "left"}, (1, 0.5), (0,)),
                Candidate({"a": "seed", "b": "right", "shared": "right"}, (0.2, 1), (0,)),
            ]
        )
        agent.next_modules.extend([1, 0])
        child, i, j, ancestor = agent._propose_merge()
        self.assertEqual(child, {"a": "left", "b": "right", "shared": "left"})
        self.assertEqual((i, j, ancestor), (1, 2, 0))
        self.assertIsNone(agent._propose_merge())

    async def test_template_launch_directory_and_reproducibility(self):
        async def evaluate(prompts, item):
            return Evaluation(float(prompts["a"] == "new"))

        provider = ScriptedProvider(["```text\nnew\n```"] * 2)
        agent = GEPA("Describe botanical specimens", provider, evaluate)
        before = os.getcwd()
        try:
            os.chdir("/private/tmp")
            rendered = await GEPA.reflect.render(agent, "a", "seed", [])
            self.assertIn(agent.task, rendered)
            first = await agent.run({"a": "seed"}, [0, 1], [2], budget=4, minibatch_size=1, seed=8)
            second = await agent.run({"a": "seed"}, [0, 1], [2], budget=4, minibatch_size=1, seed=8)
        finally:
            os.chdir(before)
        self.assertEqual(first, second)
        self.assertEqual(first["best"].score, 1)


if __name__ == "__main__":
    unittest.main()
