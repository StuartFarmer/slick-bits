"""Check SPELL's semantic generation and roulette survival."""

import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from spell import SPELL, Individual
from tests.providers import ScriptedProvider


class SPELLTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "spell/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_reproduction_and_elite_survival(self):
        async def evaluate(text):
            return len(text)

        provider = ScriptedProvider(["A proposal: {strongest}", "{x}"])
        result = await SPELL("Task", provider, evaluate).run(
            ["aa", "bbb"], iterations=1, parent_counts=(1, 2)
        )
        self.assertEqual(result["best"].prompt, "strongest")
        self.assertEqual(result["evaluations"], 4)
        self.assertEqual(provider.calls[0].count("Score:"), 1)
        self.assertEqual(provider.calls[1].count("Score:"), 2)
        self.assertNotIn("strongest", provider.calls[1])

    async def test_exponential_weights_and_bad_output(self):
        async def evaluate(text):
            return math.inf

        agent = SPELL("Task", ScriptedProvider([]), evaluate)
        with self.assertRaises(ValueError):
            await agent.run(["a"], iterations=0)
        for output in ("unmarked", "{}", "{ }", "{x}{y}"):
            with self.assertRaises(ValueError):
                await agent.reproduce([], provider=ScriptedProvider([output]))
        from random import Random

        agent.rng = Random(0)
        selected = agent._select([Individual("bad", -1000), Individual("good", 1000)], 10)
        self.assertTrue(all(p.prompt == "good" for p in selected))

    async def test_nested_placeholders_are_preserved_and_malformed_braces_rejected(self):
        agent = SPELL("Summarize", ScriptedProvider([]), None)
        result = await agent.reproduce(
            [], provider=ScriptedProvider(["Reasoning. {Summarize {input} using {{style}}.}"])
        )
        self.assertEqual(result, "Summarize {input} using {{style}}.")
        for raw in ("{unclosed", "unopened}", "{one} {two}", "{{unclosed}"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                await agent.reproduce([], provider=ScriptedProvider([raw]))

    async def test_failures_keep_raw_response_and_count_attempted_calls(self):
        async def evaluate(text):
            if text == "broken":
                raise RuntimeError("measurement failed")
            return 1.0

        for raw, evaluations, status in [
            ("unmarked", 1, "generation_failed"),
            ("{broken}", 2, "evaluation_failed"),
            (RuntimeError("offline"), 1, "generation_failed"),
        ]:
            agent = SPELL("Task", ScriptedProvider([raw]), evaluate)
            with self.subTest(raw=raw), self.assertRaises((ValueError, RuntimeError)):
                await agent.run(["seed"], iterations=1, parent_counts=(1,))
            self.assertEqual(agent.evaluations, evaluations)
            self.assertEqual(agent.optimizer_calls, 1)
            self.assertEqual(agent.attempts[-1]["status"], status)
            self.assertEqual(
                agent.attempts[-1]["raw_response"], raw if isinstance(raw, str) else None
            )
            self.assertEqual(agent.population, [Individual("seed", 1)])

    async def test_paper_defaults_duplicate_initialization_and_no_evaluation_cache(self):
        async def evaluate(text):
            return 0.5

        agent = SPELL("Any task", ScriptedProvider(["{seed}"] * 10), evaluate)
        result = await agent.run(["seed"] * 20, iterations=1)
        self.assertEqual(len(result["population"]), 20)
        self.assertEqual(result["evaluations"], 30)
        self.assertEqual(result["optimizer_calls"], 10)
        self.assertEqual([len(a["parents"]) for a in result["attempts"]], [1] * 5 + [2] * 5)
        result = await agent.run(["seed"], parent_counts=())
        self.assertEqual(len(result["history"]), 501)
        self.assertEqual(result["attempts"], [])
        self.assertEqual(result["evaluations"], 1)

    async def test_roulette_has_exponential_odds_and_elite_survives_bad_draws(self):
        agent = SPELL("Task", ScriptedProvider([]), None)
        agent.rng = random.Random(42)
        weak, strong = Individual("weak", 0), Individual("strong", math.log(3))
        selected = agent._select([weak, strong], 10000)
        self.assertAlmostEqual(selected.count(strong) / len(selected), 0.75, delta=0.02)
        with patch.object(agent, "_select", return_value=[weak]):
            self.assertEqual(agent._survive([weak, weak], [strong]), [strong, weak])

    async def test_template_renders_paper_sections_from_another_directory(self):
        agent = SPELL("Explain a technical concept", ScriptedProvider([]), None)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                rendered = await SPELL.reproduce.render(agent, [Individual("Be concrete", 0.8)])
            finally:
                os.chdir(previous)
        sections = [
            "Explain a technical concept",
            "A prompt is to guide",
            "I want you to generate",
            "Be concrete",
            "0.8",
            "Now, generate only one",
            "inside curly brackets",
        ]
        positions = [rendered.index(section) for section in sections]
        self.assertEqual(positions, sorted(positions))
        template = Path(__file__).resolve().parents[1] / "spell/prompts/reproduce.j2"
        parsed = Environment().parse(template.read_text())
        self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
