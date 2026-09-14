"""Check MAD's ordered debate and stopping decisions through real Slick calls."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from mad import MultiAgentDebate
from tests.providers import ScriptedProvider

CONTINUE = '{"answer": "", "reason": "Unresolved issue"}'
DECIDE = '{"answer": "Chosen artifact", "reason": "Meets the constraints"}'


class MADTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.templates = Path(__file__).resolve().parents[1] / "mad/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.templates)
        root.start()
        self.addCleanup(root.stop)

    async def test_judge_stops_after_one_complete_round(self):
        provider = ScriptedProvider(["Opening A", "Challenge B", DECIDE])
        agent = MultiAgentDebate("Design a garden", provider)
        result = await agent.run()
        self.assertEqual(result.answer, "Chosen artifact")
        self.assertEqual((result.rounds, result.stop_reason, result.calls), (1, "judge", 3))
        self.assertEqual([turn.speaker for turn in result.turns], ["affirmative", "negative"])
        self.assertIn("Opening A", provider.calls[1])
        self.assertIn("Challenge B", provider.calls[2])
        self.assertEqual(agent.calls[-1]["response"], DECIDE)

    async def test_rebuttals_see_latest_opponent_and_full_debate_but_not_judge(self):
        provider = ScriptedProvider(
            ["Opening A", "Opening B", CONTINUE, "Revised A", "Revised B", DECIDE]
        )
        agent = MultiAgentDebate("Pick a file format", provider)
        result = await agent.run()
        self.assertEqual((result.rounds, result.calls), (2, 6))
        self.assertEqual(
            [call["operation"] for call in agent.calls],
            ["affirm", "oppose", "discriminate", "rebut", "rebut", "discriminate"],
        )
        self.assertIn("Opening B", provider.calls[3])
        self.assertIn("Revised A", provider.calls[4])
        self.assertNotIn("Unresolved issue", provider.calls[3])
        self.assertIn("Unresolved issue", provider.calls[5])
        self.assertIn("Opening A", provider.calls[5])

    async def test_round_limit_extracts_from_all_rounds_then_selects(self):
        provider = ScriptedProvider(
            [
                "A1",
                "B1",
                CONTINUE,
                "A2",
                "B2",
                CONTINUE,
                "A3",
                "B3",
                CONTINUE,
                "Candidate list",
                DECIDE,
            ]
        )
        agent = MultiAgentDebate("Any task", provider)
        result = await agent.run()
        self.assertEqual((result.rounds, result.stop_reason, result.calls), (3, "round_limit", 11))
        for answer in ("A1", "B1", "A2", "B2", "A3", "B3"):
            self.assertIn(answer, provider.calls[9])
            self.assertIn(answer, provider.calls[10])
        self.assertIn("Candidate list", provider.calls[10])
        self.assertNotIn("Unresolved issue", provider.calls[9])
        self.assertEqual([call["operation"] for call in agent.calls[-2:]], ["extract", "select"])

    async def test_role_providers_and_run_state_are_isolated(self):
        affirmative = ScriptedProvider(["A", "Fresh A"])
        negative = ScriptedProvider(["B", "Fresh B"])
        judge = ScriptedProvider([DECIDE, DECIDE])
        agent = MultiAgentDebate(
            "Task", affirmative, negative_provider=negative, judge_provider=judge
        )
        first = await agent.run()
        second = await agent.run()
        self.assertEqual(first.turns[0].content, "A")
        self.assertEqual(second.turns[0].content, "Fresh A")
        self.assertEqual(len(agent.calls), 3)
        self.assertEqual(len(agent.judgments), 1)
        self.assertEqual(len(affirmative.calls), 2)
        self.assertEqual(len(negative.calls), 2)
        self.assertEqual(len(judge.calls), 2)
        self.assertNotIn("Meets the constraints", judge.calls[1])

    async def test_invalid_outputs_abort_and_preserve_raw_failure(self):
        for responses in (
            ["   "],
            ["A", ""],
            ["A", "B", "not json"],
            ["A", "B", '{"answer": "X", "reason": ""}'],
            ["A", "B", '{"answer": true, "reason": "R"}'],
            ["A", "B", CONTINUE, ""],
            ["A", "B", CONTINUE, "Candidates", CONTINUE],
        ):
            agent = MultiAgentDebate("Task", ScriptedProvider(responses))
            with self.subTest(responses=responses), self.assertRaises(ValueError):
                await agent.run(max_rounds=1)
            self.assertEqual(len(agent.calls), len(responses))
            self.assertEqual(agent.calls[-1]["response"], responses[-1])
            self.assertIn("error", agent.calls[-1])
        agent = MultiAgentDebate("Task", ScriptedProvider([OSError("transport")]))
        with self.assertRaisesRegex(OSError, "transport"):
            await agent.run()
        self.assertEqual(len(agent.calls), 1)
        self.assertIn("error", agent.calls[0])

    async def test_templates_and_zero_round_extraction_from_another_directory(self):
        provider = ScriptedProvider(["Candidate", DECIDE])
        agent = MultiAgentDebate(
            "Task marker", provider, criteria="Criteria marker", answer_format="Format marker"
        )
        previous = Path.cwd()
        try:
            os.chdir(self.templates)
            result = await agent.run(max_rounds=0)
            self.assertEqual((result.rounds, result.calls), (0, 2))
            for operation, args in (
                (MultiAgentDebate.affirm, ()),
                (MultiAgentDebate.oppose, ()),
                (MultiAgentDebate.rebut, ("affirmative",)),
                (MultiAgentDebate.discriminate, ()),
                (MultiAgentDebate.extract, ()),
                (MultiAgentDebate.select, ("Candidates",)),
            ):
                rendered = await operation.render(agent, *args)
                for marker in ("Task marker", "Criteria marker", "Format marker"):
                    self.assertIn(marker, rendered)
                if operation in (MultiAgentDebate.discriminate, MultiAgentDebate.select):
                    self.assertIn('"properties"', rendered)
                    self.assertEqual(rendered.count("# Output Format"), 1)
            for template in self.templates.glob("*.j2"):
                parsed = Environment().parse(template.read_text())
                self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
