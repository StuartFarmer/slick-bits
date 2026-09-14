"""CriSPO search, held-out selection, suffix ranking, and generation boundaries."""

import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from crispo import CriSPO, Example, fill_prompt
from crispo.agent import average_ranks
from tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1] / "crispo/prompts"
INPUT = "INSERT_INPUT_HERE"
CRITIQUE = "<critique>Aspect: precision. Remove unsupported claims.</critique>"


class CriSPOTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", ROOT)
        root.start()
        self.addCleanup(root.stop)
        self.train = [Example({INPUT: "training input"}, "training reference")]
        self.dev = [Example({INPUT: "DEV SECRET"}, "DEV REFERENCE")]

    async def test_train_history_dev_selection_and_duplicate_budget(self):
        provider = ScriptedProvider(
            [
                CRITIQUE,
                f"<instruction>overfit {INPUT}</instruction>",
                CRITIQUE,
                f"<instruction>middle {INPUT}</instruction>",
                CRITIQUE,
                f"<instruction>middle {INPUT}</instruction>",
            ]
        )
        requests = []

        async def respond(text):
            requests.append(text)
            return text.split()[0]

        async def evaluate(examples, outputs):
            scores = {"seed": 3, "overfit": 9, "middle": 5}
            if examples == self.dev:
                scores = {"seed": 8, "overfit": 1, "middle": 4}
            return {"quality": scores[outputs[0]]}

        agent = CriSPO("Any text task", provider, respond, evaluate, primary_metric="quality")
        result = await agent.run(
            f"seed {INPUT}", self.train, self.dev, iterations=3, history_size=2
        )
        self.assertEqual(result["best"].prompt, f"seed {INPUT}")
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual((agent.optimizer_calls, agent.critique_calls), (3, 3))
        self.assertEqual((agent.response_calls, agent.evaluations, agent.duplicates), (6, 6, 1))
        last = provider.calls[-1]
        self.assertNotIn(f"seed {INPUT}", last)
        self.assertLess(last.index(f"middle {INPUT}"), last.index(f"overfit {INPUT}"))
        self.assertIn("Remove unsupported claims", last)
        for context in provider.calls:
            self.assertNotIn("DEV SECRET", context)
            self.assertNotIn("DEV REFERENCE", context)
        self.assertTrue(all(INPUT not in text for text in requests))
        self.assertTrue(all("reference" not in text for text in requests))

    async def test_suffix_only_and_ranks_recomputed_on_each_split(self):
        provider = ScriptedProvider(
            [
                CRITIQUE,
                CRITIQUE,
                "<postscript>balanced</postscript>",
                CRITIQUE,
                "<postscript>extreme</postscript>",
                CRITIQUE,
            ]
        )
        requests = []

        async def respond(text):
            requests.append(text)
            return text.split("\n\n")[-1]

        async def evaluate(examples, outputs):
            values = {"seed": (10, 1), "balanced": (9, 2), "extreme": (8, 3)}
            if examples == self.dev:
                values = {"seed": (10, 1), "balanced": (9, 3), "extreme": (1, 2)}
            quality, secondary = values.get(outputs[0], (10, 0))
            return {"quality": quality, "secondary": secondary}

        agent = CriSPO("Any task", provider, respond, evaluate, primary_metric="quality")
        result = await agent.run(
            f"fixed {INPUT}",
            self.train,
            self.dev,
            iterations=0,
            initial_suffix="seed",
            suffix_iterations=2,
            suffix_metrics=("quality", "secondary"),
        )
        self.assertEqual(result["main_best"].prompt, f"fixed {INPUT}")
        self.assertEqual(result["best"].prompt, f"fixed {INPUT}\n\nbalanced")
        self.assertEqual(
            [c.text for c in result["suffix_history"]], ["seed", "balanced", "extreme"]
        )
        self.assertTrue(all(text.startswith("fixed ") for text in requests))
        self.assertEqual((agent.optimizer_calls, agent.critique_calls), (2, 4))
        self.assertIn("secondary", provider.calls[2])
        self.assertIn("fixed INSERT_INPUT_HERE", provider.calls[2])
        self.assertIn("<score>0</score>", provider.calls[2])
        self.assertEqual(provider.calls[4].count("<score>-0.5</score>"), 2)
        self.assertNotIn("DEV REFERENCE", "".join(provider.calls))
        # Official convention: zero-based competition ranks, averaged across metrics.
        self.assertEqual(
            average_ranks([{"x": 10, "y": 1}, {"x": 10, "y": 2}, {"x": 9, "y": 3}], ("x", "y")),
            [1.0, 0.5, 1.0],
        )

    async def test_rejected_generation_retains_raw_response_and_budget(self):
        async def respond(text):
            return "output"

        async def evaluate(examples, outputs):
            return {"quality": 1}

        for bad in (
            "missing tags",
            "<instruction> </instruction>",
            "<instruction>lost placeholder</instruction>",
            f"<instruction>{INPUT} {INPUT}</instruction>",
            f"<instruction>{INPUT} INSERT_UNKNOWN_HERE</instruction>",
            f"<instruction>{INPUT}</instruction><instruction>extra</instruction>",
        ):
            agent = CriSPO(
                "Task",
                ScriptedProvider([CRITIQUE, bad]),
                respond,
                evaluate,
                primary_metric="quality",
            )
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await agent.run(INPUT, self.train, self.dev, iterations=1)
            self.assertEqual(agent.generations[-1]["response"], bad)
            self.assertEqual((agent.optimizer_calls, agent.evaluations), (1, 2))
            self.assertEqual(len(agent.history), 1)

    async def test_measurement_and_transport_errors_propagate(self):
        async def respond(text):
            return text

        async def evaluate(examples, outputs):
            return {"quality": math.nan}

        agent = CriSPO("Task", ScriptedProvider([]), respond, evaluate, primary_metric="quality")
        with self.assertRaisesRegex(ValueError, "finite"):
            await agent.run(INPUT, self.train, self.dev, iterations=0)
        self.assertEqual(agent.evaluations, 1)

        async def valid(examples, outputs):
            return {"quality": 1}

        agent = CriSPO(
            "Task",
            ScriptedProvider([RuntimeError("transport")]),
            respond,
            valid,
            primary_metric="quality",
        )
        with self.assertRaisesRegex(RuntimeError, "transport"):
            await agent.run(INPUT, self.train, self.dev, iterations=0)
        self.assertEqual(agent.critique_calls, 1)

    async def test_separate_critic_and_rejected_suffix(self):
        provider = ScriptedProvider([f"<postscript>repeat {INPUT}</postscript>"])
        critic = ScriptedProvider([CRITIQUE, CRITIQUE])

        async def respond(text):
            return text

        async def evaluate(examples, outputs):
            return {"quality": 1, "secondary": 2}

        agent = CriSPO(
            "Task", provider, respond, evaluate, primary_metric="quality", critique_provider=critic
        )
        with self.assertRaisesRegex(ValueError, "postscript"):
            await agent.run(
                INPUT,
                self.train,
                self.dev,
                iterations=0,
                initial_suffix="",
                suffix_iterations=1,
                suffix_metrics=("secondary",),
            )
        self.assertEqual((len(provider.calls), len(critic.calls)), (1, 2))
        self.assertEqual(agent.suffix_history[0].prompt, INPUT)
        self.assertEqual(
            agent.generations[-1]["response"], f"<postscript>repeat {INPUT}</postscript>"
        )
        self.assertEqual((agent.optimizer_calls, agent.evaluations), (1, 4))

    async def test_placeholders_and_every_template_from_another_directory(self):
        template = "INSERT_CONTEXT_HERE\n" + INPUT + "\nINSERT_EXAMPLES_HERE"
        self.assertEqual(
            fill_prompt(
                template,
                {
                    INPUT: "literal INSERT_CONTEXT_HERE {{ code }}",
                    "INSERT_CONTEXT_HERE": "context",
                    "INSERT_EXAMPLES_HERE": "examples",
                },
            ),
            "context\nliteral INSERT_CONTEXT_HERE {{ code }}\nexamples",
        )
        with self.assertRaises(KeyError):
            fill_prompt(template, {INPUT: "input"})

        async def respond(text):
            return text

        async def evaluate(examples, outputs):
            return {"quality": 1}

        agent = CriSPO("Task", ScriptedProvider([]), respond, evaluate, primary_metric="quality")
        cwd = Path.cwd()
        try:
            os.chdir(ROOT.parent)
            for method, kwargs in (
                (CriSPO.critique, {"instruction": INPUT, "observations": []}),
                (CriSPO.revise, {"trajectory": []}),
                (
                    CriSPO.critique_suffix,
                    {
                        "main_prompt": INPUT,
                        "suffix": "s",
                        "observations": [],
                        "metrics": ("quality",),
                    },
                ),
                (
                    CriSPO.revise_suffix,
                    {"main_prompt": INPUT, "trajectory": [], "metrics": ("quality",)},
                ),
            ):
                self.assertIn("Task", await method.render(agent, **kwargs))
        finally:
            os.chdir(cwd)
        for template in ROOT.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
