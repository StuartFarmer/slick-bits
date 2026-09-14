"""Exercise rationale ensembles through real Slick prompts with scripted outputs."""

import os
import unittest
from decimal import Decimal
from pathlib import Path
from random import Random
from unittest.mock import patch

from jinja2 import Environment, nodes

from rationale_ensembles import Example, RationaleEnsemble, parse_response
from tests.providers import ScriptedProvider


class RationaleEnsembleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "rationale_ensembles"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.root / "prompts")
        root.start()
        self.addCleanup(root.stop)
        self.examples = tuple(Example(f"input-{i}", f"seed-{i}", str(i)) for i in range(3))

    async def test_fixed_prompt_plurality_invalid_budget_and_tie(self):
        outputs = [
            "Reason one. The answer is B.",
            "Reason two. The answer is A.",
            "Reason three. The answer is A.",
            "Missing answer.",
            "Reason four. The answer is C.",
        ]
        provider = ScriptedProvider(outputs + ["R. The answer is Z.", "R. The answer is A."])
        agent = RationaleEnsemble("Task {{ literal }}", provider, examples=self.examples)
        result = await agent.run("data", samples=5)
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.counts, {"B": 1, "A": 2, "C": 1})
        self.assertEqual(result.consistency, 2 / 5)
        self.assertEqual(len(set(provider.calls)), 1)
        self.assertEqual([s.response for s in result.samples], outputs)
        self.assertIsNotNone(result.samples[3].error)
        tied = await agent.run("next", samples=2)
        self.assertEqual(tied.answer, "Z")
        self.assertEqual(len(agent.samples), 2)
        self.assertEqual(len(agent.calls), 7)

    async def test_shuffled_prompts_preserve_examples_and_seed_reproducibility(self):
        rendered = []
        for _ in range(2):
            provider = ScriptedProvider(["R. The answer is ok."] * 8)
            agent = RationaleEnsemble("Task", provider, examples=self.examples, rng=Random(17))
            await agent.run("query", method="prompt_order", samples=8)
            rendered.append(provider.calls)
            for context in provider.calls:
                for example in self.examples:
                    self.assertIn(f"Q: {example.input}\nA: {example.rationale}", context)
        self.assertEqual(rendered[0], rendered[1])
        self.assertGreater(len(set(rendered[0])), 1)
        self.assertEqual([e.rationale for e in self.examples], ["seed-0", "seed-1", "seed-2"])

    async def test_bootstrap_holds_out_exemplar_filters_answers_and_reuses_pools(self):
        bootstrap = ScriptedProvider(
            [
                "alternative-0. The answer is 0.",
                "wrong. The answer is 9.",
                "alternative-1. The answer is 1.",
                "malformed",
                "alternative-2. The answer is 2.",
                "other-2. The answer is 2.",
            ]
        )
        provider = ScriptedProvider(["R. The answer is final."] * 9)
        agent = RationaleEnsemble("Task", provider, examples=self.examples, rng=Random(4))
        pools = await agent.prepare_rationales(provider=bootstrap, samples_per_example=2)
        self.assertEqual(
            pools, (("alternative-0.",), ("alternative-1.",), ("alternative-2.", "other-2."))
        )
        self.assertEqual(len(bootstrap.calls), 6)
        for i, context in enumerate(bootstrap.calls):
            held_out = i // 2
            self.assertNotIn(f"seed-{held_out}", context)
            self.assertIn(f"Q: input-{held_out}\nA:", context)
            for j in range(3):
                if j != held_out:
                    self.assertIn(f"seed-{j}", context)
        self.assertIn("rejection", agent.calls[1])
        self.assertIn("rejection", agent.calls[3])
        await agent.run("query", method="input_rationale", rationale_pools=pools, samples=8)
        for context in provider.calls:
            self.assertEqual(sum(f"seed-{i}" in context for i in range(3)), 2)
            self.assertEqual(
                [context.index(f"input-{i}") for i in range(3)],
                sorted(context.index(f"input-{i}") for i in range(3)),
            )
            for i in range(3):
                self.assertIn(f"The answer is {i}.", context)
        await agent.run(
            "query", method="input_rationale", rationale_pools=pools, replace_all=True, samples=1
        )
        self.assertTrue(all(f"seed-{i}" not in provider.calls[-1] for i in range(3)))
        self.assertEqual(len(bootstrap.calls), 6)

    async def test_equivalence_applies_to_bootstrap_and_voting(self):
        provider = ScriptedProvider(["R. The answer is YES.", "R. The answer is Yes."])
        agent = RationaleEnsemble(
            "Classify",
            provider,
            examples=(Example("demo", "seed", "yes"),),
            normalize_answer=str.casefold,
        )
        pools = await agent.prepare_rationales(samples_per_example=1)
        self.assertEqual(pools, (("R.",),))
        self.assertEqual((await agent.run("data", samples=1)).answer, "yes")

    async def test_custom_response_format_and_answer_equivalence_for_another_task(self):
        agent = RationaleEnsemble(
            "Return an amount.",
            ScriptedProvider(["compute|18.00", "check|18", "other|26"]),
            parse_response=lambda text: tuple(text.split("|")),
            normalize_answer=lambda text: str(Decimal(text).normalize()),
        )
        result = await agent.run("arbitrary data", samples=3)
        self.assertEqual(result.answer, "18")
        self.assertEqual(result.counts, {"18": 2, "26": 1})
        self.assertEqual(result.samples[0].rationale, "compute")

    def test_text_parser_retains_decimal_and_ignores_extra_questions(self):
        self.assertEqual(
            parse_response("Work. The answer is 3.14.\nQ: invented\nThe answer is 9."),
            ("Work.", "3.14"),
        )
        for text in ("", "R. The answer is .", "The answer is A."):
            with self.assertRaises(ValueError):
                parse_response(text)

    async def test_empty_pool_and_all_invalid_fail_without_hidden_retries(self):
        agent = RationaleEnsemble(
            "Task", ScriptedProvider(["R. The answer is wrong."]), examples=self.examples[:1]
        )
        with self.assertRaisesRegex(ValueError, "no accepted rationales"):
            await agent.prepare_rationales(samples_per_example=1)
        self.assertEqual(len(agent.calls), 1)
        for pools in ((), ((),)):
            with self.assertRaisesRegex(ValueError, "rationale pool"):
                await agent.run("data", method="input_rationale", rationale_pools=pools)
        agent = RationaleEnsemble("Task", ScriptedProvider(["bad", "The answer is A."]))
        with self.assertRaisesRegex(ValueError, "no valid answers"):
            await agent.run("data", samples=2)
        self.assertEqual(len(agent.calls), 2)

    async def test_provider_errors_abort_and_retain_raw_and_partial_records(self):
        for error in (TimeoutError("offline"), ValueError("transport"), ("raw", ["tool"])):
            agent = RationaleEnsemble("Task", ScriptedProvider(["R. The answer is A.", error]))
            with self.assertRaises((ValueError, TimeoutError)):
                await agent.run("data", samples=3)
            self.assertEqual(len(agent.samples), 1)
            self.assertEqual(len(agent.calls), 2)
            self.assertIn("error", agent.calls[-1])
            if isinstance(error, tuple):
                self.assertEqual(agent.calls[-1]["response"], "raw")

        def broken_parser(text):
            raise RuntimeError("parser bug")

        agent = RationaleEnsemble("Task", ScriptedProvider(["raw"]), parse_response=broken_parser)
        with self.assertRaisesRegex(RuntimeError, "parser bug"):
            await agent.run("data", samples=2)
        self.assertEqual(agent.calls[-1]["response"], "raw")
        self.assertIn("error", agent.calls[-1])

    async def test_defaults_and_templates_from_both_launch_directories(self):
        provider = ScriptedProvider(["R. The answer is ok."] * 40)
        agent = RationaleEnsemble("Task {{ literal }}", provider, examples=self.examples)
        self.assertEqual(len((await agent.run("data")).samples), 40)
        previous = Path.cwd()
        try:
            for cwd in (self.root, self.root.parent):
                os.chdir(cwd)
                for operation in (RationaleEnsemble.generate, RationaleEnsemble.sample_rationale):
                    text = await operation.render(agent, "query", self.examples)
                    self.assertIn("Task {{ literal }}", text)
                    self.assertIn("Q: query\nA:", text)
            for template in (self.root / "prompts").glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
