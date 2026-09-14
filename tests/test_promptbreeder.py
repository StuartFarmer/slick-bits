"""Check PromptBreeder's self-mutation, tournament replacement, and operators."""

import math
import os
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from promptbreeder import Evaluation, Individual, PromptBreeder, Unit
from tests.providers import ScriptedProvider


async def similarity(a, b):
    return float(a == b)


async def evaluate(unit):
    return Evaluation(len(unit.prompts[0]), ("correct working",))


class PromptBreederTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "promptbreeder/prompts"
        )
        root.start()
        self.addCleanup(root.stop)

    async def test_disjoint_pairs_and_default_author_seeds(self):
        agent = PromptBreeder("Task", ScriptedProvider(["a"] * 20), evaluate, similarity)
        result = await agent.run(
            population_size=4, prompt_count=1, generations=2, operators=("first",)
        )
        self.assertEqual(result["evaluations"], 8)
        self.assertEqual(len(result["operators"]), 4)
        for start in (0, 2):
            self.assertEqual(
                sorted(i for pair in result["pairs"][start : start + 2] for i in pair), [0, 1, 2, 3]
            )
        self.assertEqual(len(result["history"]), 8)

    async def test_context_single_replacement_and_resampling(self):
        agent = PromptBreeder("Task", ScriptedProvider([]), evaluate, similarity)
        agent.rng = random.Random(0)
        unit = Unit(("p",), "m", context=("old1", "old2", "old3"))
        updated = agent._context(unit, ("new1", "new2", "new3"), 3)
        self.assertEqual(len(set(unit.context) & set(updated.context)), 2)
        self.assertEqual(agent._context(unit, ("new",), 0).context, ())
        parent = Individual(unit, Evaluation(1, ("new1", "new2")))
        with patch.object(agent.rng, "random", side_effect=[0.0] + [0.99] * 10):
            child = await agent._mutate(parent, [parent, parent], ["style"], 3, "context")
        self.assertEqual(child.context, ())
        self.assertEqual(unit.context, ("old1", "old2", "old3"))

    async def test_sequential_execution_and_text_continuations(self):
        provider = ScriptedProvider(["draft", "final"])
        agent = PromptBreeder("Task", ScriptedProvider([]), evaluate, similarity)
        unit = Unit(("plan", "answer"), "mutation", ("verified example",))
        outputs = await agent.predict(unit, "user input", provider=provider)
        self.assertEqual(outputs, ("draft", "final"))
        self.assertIn("verified example", provider.calls[0])
        self.assertIn("plan", provider.calls[0])
        self.assertIn("user input", provider.calls[1])
        self.assertIn("draft", provider.calls[1])
        self.assertIn("answer", provider.calls[1])
        self.assertNotIn("mutation", "".join(provider.calls))
        hint = await agent.zero(provider=ScriptedProvider(["1. First hint\ncontinued\n2. Other"]))
        self.assertEqual(hint, "First hint\ncontinued")
        text = await PromptBreeder.initialize.render(agent, "mutate", "style")
        self.assertIn("INSTRUCTION: Task", text)
        self.assertTrue(text.endswith("INSTRUCTION MUTANT:"))

    async def test_full_elite_lineage_and_archive(self):
        agent = PromptBreeder(
            "Task", ScriptedProvider(["a", "b", "c", "d", "e"]), evaluate, similarity
        )
        result = await agent.run(population_size=2, tournaments=1, operators=("first",))
        self.assertEqual(result["population"][0].unit.lineage, (("a", "b"),))
        self.assertEqual(result["history"][0].unit.context, ())
        self.assertEqual(result["best"].evaluation.score, 1)
        agent.population.clear()
        self.assertEqual(agent.best, result["best"])
        parent = Individual(
            Unit(("a", "b"), "m", lineage=(("ancestor0", "ancestor1"),)), Evaluation(1)
        )
        agent.provider = ScriptedProvider(["new"])
        with (
            patch.object(agent.rng, "randrange", return_value=1),
            patch.object(agent.rng, "random", return_value=0.9),
        ):
            await agent._mutate(parent, [parent, parent], ["style"], 0, "lineage")
        self.assertIn("ancestor1", agent.provider.calls[0])
        self.assertNotIn("ancestor0", agent.provider.calls[0])

    async def test_crossover_uses_fitness_and_any_donor_slot(self):
        parent = Individual(Unit(("a", "b"), "parent mutation"), Evaluation(0))
        donor = Individual(Unit(("c", "d"), "donor mutation"), Evaluation(3))
        zero = Individual(Unit(("e", "f"), "other mutation"), Evaluation(0))
        agent = PromptBreeder("Task", ScriptedProvider(["mutant"]), evaluate, similarity)
        with (
            patch.object(agent.rng, "random", return_value=0.0),
            patch.object(agent.rng, "choices", wraps=agent.rng.choices) as choose,
        ):
            child = await agent._mutate(parent, [parent, donor, zero], ["style"], 0, "first")
        self.assertTrue(set(child.prompts) & set(donor.unit.prompts))
        self.assertEqual(child.mutation, "parent mutation")
        self.assertEqual(choose.call_args.kwargs["weights"], [1.0, 0.0])
        self.assertEqual(parent.unit.prompts, ("a", "b"))

    async def test_eda_ranking_threshold_and_no_fitness_in_prompt(self):
        async def similar(a, b):
            return 0.96 if {a, b} == {"duplicate", "best"} else 0.95

        agent = PromptBreeder("Task", ScriptedProvider([]), evaluate, similar)
        population = [
            Individual(Unit((s,), "m"), Evaluation(score))
            for s, score in (("duplicate", 2), ("distinct", 3), ("best", 7))
        ]
        parents = await agent._diverse(population, 0, True)
        self.assertEqual(parents, ["distinct", "best"])
        text = await PromptBreeder.ranked.render(agent, parents, "mutate")
        self.assertIn("descending order of score", text)
        self.assertIn("(3) is the best response", text)
        self.assertLess(text.index("distinct"), text.index("best", text.index("(1) distinct")))
        self.assertNotIn("7", text)

    async def test_templates_work_from_other_directories_without_branches(self):
        root = Path(__file__).resolve().parents[1] / "promptbreeder"
        agent = PromptBreeder("Task", ScriptedProvider([]), evaluate, similarity)
        for template in (root / "prompts").glob("*.j2"):
            self.assertFalse(
                list(Environment().parse(template.read_text()).find_all((nodes.If, nodes.CondExpr)))
            )
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                for cwd in (root, Path(directory)):
                    os.chdir(cwd)
                    rendered = await PromptBreeder.initialize.render(agent, "mutate", "style")
                    self.assertIn("INSTRUCTION: Task", rendered)
            finally:
                os.chdir(previous)

    async def test_generation_and_evaluation_errors_propagate_without_retries(self):
        async def broken(unit):
            raise RuntimeError("evaluation offline")

        agent = PromptBreeder("Task", ScriptedProvider(["ok", " "]), broken, similarity)
        with self.assertRaisesRegex(RuntimeError, "evaluation offline"):
            await agent.run(population_size=2, prompt_count=1, tournaments=0)
        self.assertEqual(agent.evaluations, 1)
        self.assertEqual(agent.history, [])
        with self.assertRaisesRegex(ValueError, "blank"):
            await agent.first("p", "m", provider=agent.provider)
        self.assertEqual(agent.raw_responses, ["ok", " "])
        transport = ScriptedProvider([RuntimeError("transport offline")])
        with self.assertRaisesRegex(RuntimeError, "transport offline"):
            await agent.first("p", "m", provider=transport)
        self.assertEqual(len(transport.calls), 1)

    async def test_hypermutation_and_unconditional_loser_replacement(self):
        provider = ScriptedProvider(["winner", "loser", "new mutation", "x"])
        with patch("promptbreeder.agent.random.Random.random", return_value=0.9):
            result = await PromptBreeder("Task", provider, evaluate, similarity).run(
                ["initial mutation"],
                ["style"],
                population_size=2,
                prompt_count=1,
                tournaments=1,
                context_size=2,
                operators=("hyper_first",),
                seed=1,
            )
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(result["operators"], ["hyper_first"])
        self.assertEqual({p.unit.prompts[0] for p in result["population"]}, {"winner", "x"})
        child = next(p for p in result["population"] if p.unit.prompts == ("x",))
        self.assertEqual(child.unit.mutation, "new mutation")
        self.assertIn("new mutation", provider.calls[-1])
        self.assertIn("correct working", child.unit.context)

    async def test_every_operator_and_diversity(self):
        population = [
            Individual(
                Unit((s,), "mut", lineage=(("ancestor",),)), Evaluation(len(s), ("correct",))
            )
            for s in ("a", "bb", "a")
        ]
        for operator in PromptBreeder.operators:
            provider = ScriptedProvider(["new"] * 2)
            agent = PromptBreeder("Task", provider, evaluate, similarity)
            agent.rng = random.Random(2)
            child = await agent._mutate(population[0], population, ["style"], 2, operator)
            self.assertTrue(child.prompts[0])
        agent.rng = random.Random(0)
        self.assertEqual(await agent._diverse(population, 0, True), ["a", "bb"])

    async def test_rejections_and_missing_lamarckian_evidence(self):
        async def invalid(unit):
            return Evaluation(math.nan)

        agent = PromptBreeder("Task", ScriptedProvider([]), invalid, similarity)
        agent.evaluations = 0
        with self.assertRaises(ValueError):
            await agent._assess(Unit(("a",), "m"))
        with self.assertRaises(ValueError):
            await agent.zero(provider=ScriptedProvider([" "]))
        agent.rng = random.Random(0)
        parent = Individual(Unit(("a",), "m"), Evaluation(0))
        agent.provider = ScriptedProvider(["fallback"])
        with patch.object(agent.rng, "random", return_value=0.9):
            child = await agent._mutate(
                parent, [parent, replace(parent)], ["style"], 0, "lamarckian"
            )
        self.assertEqual(child.prompts, ("fallback",))
        self.assertEqual(agent.fallbacks[0][0], "lamarckian")
