"""Neutral mutation accepts recurrence and accounts for validation attempts."""

import math
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from lmca import LMCA, Validation, analyze, pairwise_distances
from lmca.gp import Primitive, Tree, subtree_mutate
from tests.providers import ScriptedProvider


async def validate(text):
    if text.startswith("bad"):
        return Validation(None, "invalid syntax")
    return Validation(text)


class ChainTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "lmca/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_repair_fallback_and_primary_reset_without_selection(self):
        primary = ScriptedProvider(["bad-one", "bad-two", "A", "A"])
        fallback = ScriptedProvider(["bad-three", "B"])
        agent = LMCA("Rewrite a document", primary, validate, fallback=fallback)
        result = await agent.run("A", steps=3, retries=1)
        self.assertEqual(result.programs, ("A", "B", "A", "A"))
        self.assertEqual(result.stop_reason, "budget")
        self.assertEqual(result.calls, 6)
        self.assertIn("bad-one", primary.calls[1])
        self.assertIn("bad-two", fallback.calls[0])
        self.assertIn("invalid syntax", fallback.calls[0])
        self.assertIn("bad-three", fallback.calls[1])
        self.assertIn("Parent program:\nB", primary.calls[2])
        self.assertNotIn("bad-three", primary.calls[2])
        self.assertEqual(
            [c["operation"] for c in agent.calls],
            ["mutate", "repair", "repair", "repair", "mutate", "mutate"],
        )
        self.assertEqual(
            [c["accepted"] for c in agent.calls], [False, False, False, True, True, True]
        )

    async def test_default_budget_is_six_attempts_per_model_and_no_fake_step(self):
        primary = ScriptedProvider(["bad"] * 6)
        fallback = ScriptedProvider(["bad"] * 6)
        result = await LMCA("Task", primary, validate, fallback=fallback).run("seed")
        self.assertEqual(result.programs, ("seed",))
        self.assertEqual(result.stop_reason, "validation_exhausted")
        self.assertEqual(result.calls, 12)

    async def test_validation_canonicalizes_and_blank_generation_is_repaired(self):
        async def canonical(text):
            return Validation(text.strip().upper())

        provider = ScriptedProvider([" ", " a ", "b"])
        agent = LMCA("Task", provider, canonical)
        result = await agent.run("seed", steps=2)
        self.assertEqual(result.programs, ("seed", "A", "B"))
        self.assertEqual(agent.calls[1]["response"], " a ")
        self.assertIn("Parent program:\nA", provider.calls[2])
        result = await agent.run("fresh", steps=0)
        self.assertEqual(result.programs, ("fresh",))
        self.assertEqual(agent.calls, [])

    async def test_transport_validator_and_tool_errors_propagate_without_fallback(self):
        async def broken(text):
            raise RuntimeError("validator bug")

        for responses, checker, error in (
            ([RuntimeError("network")], validate, RuntimeError),
            (["candidate"], broken, RuntimeError),
            ([("candidate", [{"name": "tool"}])], validate, ValueError),
        ):
            fallback = ScriptedProvider([])
            agent = LMCA("Task", ScriptedProvider(responses), checker, fallback=fallback)
            with self.assertRaises(error):
                await agent.run("seed")
            self.assertEqual(len(agent.calls), 1)
            self.assertIn("error", agent.calls[0])
            self.assertEqual(fallback.calls, [])
            self.assertEqual(agent.programs, ["seed"])

    async def test_templates_render_from_other_directory_and_have_no_branches(self):
        agent = LMCA(
            "Task",
            ScriptedProvider([]),
            validate,
            constraints="My grammar",
            instruction="Explore forms.",
        )
        old = Path.cwd()
        try:
            os.chdir(self.templates.parent)
            mutation = await LMCA.mutate.render(agent, "parent")
            repair = await LMCA.repair.render(agent, "bad", "failure")
        finally:
            os.chdir(old)
        self.assertIn("Explore forms.", mutation)
        self.assertIn("My grammar", mutation)
        self.assertIn("failure", repair)
        self.assertNotIn("Explore forms.", repair)
        for template in self.templates.glob("*.j2"):
            self.assertEqual(
                list(
                    Environment().parse(template.read_text()).find_all((nodes.If, nodes.CondExpr))
                ),
                [],
            )


class AnalysisTests(unittest.TestCase):
    def test_recurrence_cycles_parallel_edges_and_distinct_entropy_definitions(self):
        result = analyze(["A", "B", "A", "C", "A", "A"], skeleton=lambda p: "shape")
        graph = result["programs"]
        self.assertEqual(graph["cumulative_unique"], (1, 2, 2, 3, 3, 3))
        self.assertEqual(graph["visits"], {"A": 4, "B": 1, "C": 1})
        self.assertEqual(graph["cycle_lengths"], {1: 1, 2: 2})
        self.assertEqual(graph["mean_successor_entropy"], math.log2(3) / 3)
        expected = -(0.6 * math.log2(0.6) + 2 * 0.2 * math.log2(0.2)) / 3
        self.assertAlmostEqual(graph["mean_degree_entropy"], expected)
        self.assertEqual(result["skeletons"]["transitions"], {("shape", "shape"): 5})
        self.assertEqual(result["skeletons"]["cycle_lengths"], {1: 1})
        self.assertEqual(result["skeletons"]["revisit_fraction"], 1.0)
        self.assertEqual(result["successive_distances"], (1.0, 1.0, 1.0, 1.0, 0.0))

    def test_token_edit_distance_is_normalized_and_pairwise_matrix_is_symmetric(self):
        programs = ["a b", "a c", "a b c", ""]
        matrix = pairwise_distances(programs, tokenize=str.split)
        self.assertEqual(matrix[0], (0.0, 0.5, 1 / 3, 1.0))
        self.assertEqual(matrix, tuple(zip(*matrix)))
        empty = analyze([], skeleton=str)
        self.assertEqual(empty["programs"]["cycle_lengths"], {})
        self.assertEqual(empty["programs"]["mean_degree_entropy"], 0)
        single = analyze(["only"], skeleton=str)
        self.assertEqual(single["programs"]["cumulative_unique"], (1,))
        self.assertEqual(single["programs"]["revisit_fraction"], 0)

    def test_subtree_mutation_preserves_types_depth_and_seed_reproducibility(self):
        primitives = [
            Primitive("pair", "text", ("text", "text")),
            Primitive("word", "text"),
            Primitive("choose", "text", ("bool", "text")),
            Primitive("true", "bool"),
        ]
        initial = Tree(primitives[0], (Tree(primitives[1]), Tree(primitives[1])))

        def check(tree):
            self.assertEqual(
                tuple(c.primitive.returns for c in tree.children), tree.primitive.arguments
            )
            return 0 if not tree.children else 1 + max(check(c) for c in tree.children)

        chains = []
        for _ in range(2):
            rng = random.Random(7)
            chain = [initial]
            for _ in range(40):
                chain.append(subtree_mutate(chain[-1], primitives, rng=rng, max_depth=4))
                self.assertLessEqual(check(chain[-1]), 4)
                self.assertEqual(chain[-1].primitive.returns, "text")
            chains.append(chain)
        self.assertEqual(chains[0], chains[1])
        self.assertGreater(len({tree.render() for tree in chains[0]}), 1)
        self.assertEqual(initial.render(), "pair(word, word)")
