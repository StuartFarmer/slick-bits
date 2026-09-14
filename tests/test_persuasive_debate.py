"""Offline protocol checks; no model calls or benchmark correctness claims."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import Prompt

from persuasive_debate import Debater, PersuasiveDebate, Problem
from persuasive_debate.agent import extract, normalize, truncate, verify_quotes
from persuasive_debate.evaluation import Match, fit_elo, swiss_tournament
from tests.providers import ScriptedProvider


def argument(text):
    return f"<thinking>PRIVATE SCRATCHPAD</thinking><argument>{text}</argument>"


class DebateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = patch(
            "slick.prompts.TEMPLATE_ROOT",
            Path(__file__).resolve().parents[1] / "persuasive_debate/prompts",
        )
        self.root.start()
        self.addCleanup(self.root.stop)
        self.problem = Problem(
            "Which design meets the requirements?",
            ("Red", "Blue"),
            "PRIVATE SOURCE. Red has a latch. Blue has a lock.",
        )

    def agent(self, first, second, judge, **kwargs):
        return PersuasiveDebate(
            self.problem,
            (Debater(first), Debater(second)),
            judge,
            min_words=1,
            target_words=10,
            max_words=30,
            candidates_per_sample=1,
            **kwargs,
        )

    async def test_simultaneous_snapshots_and_swapped_judgments(self):
        a = ScriptedProvider([argument("red0 <quote>Red has a latch.</quote>"), argument("red1")])
        b = ScriptedProvider([argument("blue0"), argument("blue1")])
        judge = ScriptedProvider(["Answer: A", "Answer: B"])
        agent = self.agent(a, b, judge)
        result = await agent.run(rounds=2)
        self.assertEqual(result.choice, 0)
        self.assertEqual(result.answer, "Red")
        self.assertEqual(result.votes, (0, 0))
        self.assertEqual(result.calls, 6)
        self.assertNotIn("red0", b.calls[0])
        for provider in (a, b):
            self.assertIn("red0", provider.calls[1])
            self.assertIn("blue0", provider.calls[1])
            self.assertNotIn("PRIVATE SCRATCHPAD", provider.calls[1])
            self.assertNotIn("red1", provider.calls[1])
        self.assertLess(b.calls[1].index("blue0"), b.calls[1].index("red0"))
        for context in judge.calls:
            self.assertNotIn("PRIVATE SOURCE", context)
            self.assertNotIn("PRIVATE SCRATCHPAD", context)
            self.assertIn("<v_quote>Red has a latch.</v_quote>", context)
        self.assertIn("A: Blue", judge.calls[1])
        self.assertLess(judge.calls[1].index("blue0"), judge.calls[1].index("red0"))

    async def test_best_of_n_uses_target_logprob_and_dummy_opponent(self):
        a = ScriptedProvider([argument("red weak"), argument("red strong")])
        b = ScriptedProvider([argument("blue weak"), argument("blue strong")])
        judge = ScriptedProvider(["Answer: A", "Answer: A"])
        scoring = []

        async def preference(context):
            scoring.append(context)
            score = -0.1 if "strong" in context else -2.0
            return (
                {"A": score, "B": -10 - score}
                if "red" in context
                else {"B": score, "A": -10 - score}
            )

        agent = self.agent(a, b, judge, preference=preference)
        agent.debaters = (Debater(a, best_of=2), Debater(b, best_of=2))
        result = await agent.run(rounds=1)
        self.assertEqual([turn.text for turn in result.turns], ["red strong", "blue strong"])
        self.assertEqual(result.votes, (0, 1))
        self.assertIsNone(result.choice)
        self.assertEqual(result.approval, (0.5, 0.5))
        self.assertEqual(result.calls, 10)
        for context in scoring:
            self.assertIn("My answer is the best choice and my opponent is wrong.", context)
            self.assertNotIn("PRIVATE SOURCE", context)
            self.assertNotIn("PRIVATE SCRATCHPAD", context)
            self.assertFalse("red strong" in context and "blue strong" in context)

    async def test_critique_selection_refinement_and_consultancy(self):
        a = ScriptedProvider(
            [argument("initial"), argument("refined"), argument("followup"), argument("final")]
        )
        critic = ScriptedProvider(
            ["<critique>weak advice</critique>", "<critique>helpful advice</critique>"] * 2
        )
        judge = ScriptedProvider(
            ["<question>Explain the latch.</question>", "Answer: A", "Answer: B"]
        )
        scored = []

        async def preference(context):
            scored.append(context)
            return {"Y": -0.1 if "helpful advice" in context else -3.0}

        agent = self.agent(
            a, ScriptedProvider([]), judge, protocol="consultancy", preference=preference
        )
        agent.debaters = (Debater(a, critiques=2, critic=critic), agent.debaters[1])
        result = await agent.run(rounds=2)
        self.assertEqual(
            [turn.text for turn in result.turns], ["refined", "Explain the latch.", "final"]
        )
        self.assertIn("helpful advice", a.calls[1])
        self.assertIn("Explain the latch.", a.calls[2])
        self.assertIn("PRIVATE SOURCE", critic.calls[0])
        self.assertTrue(all("PRIVATE SOURCE" not in context for context in scored))

    async def test_rejection_pool_and_failed_refinement_keep_original(self):
        a = ScriptedProvider(
            [
                "malformed",
                argument("short"),
                argument("<quote>Red has a latch.</quote>"),
                "refusal",
                "refusal",
                "refusal",
            ]
        )
        b = ScriptedProvider([argument("opponent words")] * 3)
        critic = ScriptedProvider(["<critique>Improve support.</critique>"])
        agent = self.agent(a, b, ScriptedProvider(["Answer: A", "Answer: B"]))
        agent.min_words, agent.candidates_per_sample = 2, 3
        agent.debaters = (Debater(a, critiques=1, critic=critic), agent.debaters[1])
        result = await agent.run(rounds=1)
        self.assertEqual(result.turns[0].text, "<v_quote>Red has a latch.</v_quote>")
        self.assertTrue(agent.failures)
        self.assertTrue(any(call.response == "malformed" for call in agent.calls))

    async def test_quote_verification_rechecks_forgery_and_strips_hidden_text(self):
        async def verify(text):
            return bool(normalize(text)) and normalize(text) in normalize(self.problem.evidence)

        result = await verify_quotes(
            "<v_quote>fabricated</v_quote> <quote>RED has a latch!</quote> "
            "<v.quote>Blue has a lock.</v.quote> <quote>!!!</quote>",
            verify,
        )
        self.assertIn("<u_quote>fabricated</u_quote>", result)
        self.assertIn("<v_quote>RED has a latch!</v_quote>", result)
        self.assertIn("<v_quote>Blue has a lock.</v_quote>", result)
        self.assertIn("<u_quote>!!!</u_quote>", result)
        with self.assertRaises(ValueError):
            extract("<thinking><argument>secret</argument></thinking>", "argument")
        nested = (
            "<argument>public <thinking>outer <thinking>inner</thinking>"
            "PRIVATE SECRET</thinking> end</argument>"
        )
        self.assertEqual(extract(nested, "argument"), "public  end")
        self.assertNotIn("PRIVATE SECRET", await verify_quotes(nested, verify))

    async def test_provider_failure_leaves_records_and_no_partial_round(self):
        agent = self.agent(
            ScriptedProvider([OSError("offline")]),
            ScriptedProvider([argument("completed")]),
            ScriptedProvider([]),
        )
        with self.assertRaises(OSError):
            await agent.run(rounds=1)
        self.assertEqual(agent.turns, [])
        self.assertTrue(any("offline" in (call.error or "") for call in agent.calls))
        self.assertTrue(any(call.response == argument("completed") for call in agent.calls))

    async def test_missing_and_invalid_generated_logprobs(self):
        for score, expected in (({}, "second"), ({"A": float("nan")}, None), ({"A": 0.1}, None)):
            with self.subTest(score=score):
                values = iter([score, {"A": -1.0}])

                async def preference(context):
                    return next(values)

                a = ScriptedProvider([argument("first"), argument("second")])
                agent = self.agent(
                    a,
                    ScriptedProvider([argument("other")]),
                    ScriptedProvider(["Answer: A", "Answer: B"]),
                    preference=preference,
                )
                agent.debaters = (Debater(a, best_of=2), agent.debaters[1])
                if expected is None:
                    with self.assertRaises(ValueError):
                        await agent.run(rounds=1)
                    self.assertTrue(any(call.error for call in agent.calls))
                else:
                    result = await agent.run(rounds=1)
                    self.assertEqual(result.turns[0].text, expected)

    async def test_interactive_debate_consultant_second_side_and_fresh_run(self):
        for protocol in ("interactive_debate", "consultancy"):
            a = ScriptedProvider([argument("red")] * 6)
            b = ScriptedProvider([argument("blue")] * 6)
            judge = ScriptedProvider(
                [
                    "<question>Defender of Blue: why?</question>",
                    "<question>Defender of Red: why?</question>",
                    "Answer: B",
                    "Answer: A",
                    "Answer: B",
                    "Answer: A",
                ]
            )
            agent = self.agent(a, b, judge, protocol=protocol, consultant=1)
            result = await agent.run(rounds=3)
            self.assertEqual(result.answer, "Blue")
            self.assertEqual(sum(turn.side is None for turn in result.turns), 2)
            self.assertIn("Defender of Blue: why?", b.calls[1])
            self.assertIn("Defender of Red: why?", b.calls[2])
            if protocol == "consultancy":
                self.assertEqual(a.calls, [])
                self.assertTrue(all(turn.side in (1, None) for turn in result.turns))
                self.assertIn("within\n2–60 words", b.calls[0])
            second = await agent.run(rounds=1)
            self.assertEqual(sum(turn.side is None for turn in second.turns), 0)
            self.assertEqual(agent.round, 1)

    async def test_all_templates_render_and_contain_no_control_branches(self):
        agent = self.agent(ScriptedProvider([]), ScriptedProvider([]), ScriptedProvider([]))
        view = agent._view(())
        operations = [
            (agent.open_argument, (0, view)),
            (agent.challenge, (0, view)),
            (agent.rebut, (0, view)),
            (agent.respond, (0, view)),
            (agent.revise, (0, view, "argument", "feedback")),
            (agent.critique, (0, view, "argument")),
            (agent.ask, (view,)),
            (agent.judge_answer, (view,)),
        ]
        for operation, args in operations:
            rendered = await operation.render(agent, *args)
            self.assertIn(self.problem.question, rendered)
        for template in ("prefer_argument.j2", "prefer_critique.j2"):
            context = Prompt(template)(view=view, argument="argument", critique="feedback")
            self.assertNotIn("PRIVATE SOURCE", context)
        root = Path(__file__).resolve().parents[1] / "persuasive_debate/prompts"
        for template in root.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertFalse(list(parsed.find_all((nodes.If, nodes.CondExpr))))

    async def test_truncated_quotes_and_custom_evidence_verification(self):
        text = truncate("<quote>red blue green four</quote> extra", 3)
        self.assertIn("</quote>", text)
        self.assertIn("<TRUNCATED>", text)
        received = []

        async def verify(quote):
            received.append(quote)
            return quote == "x < 2"

        result = await verify_quotes("<v_quote>x < 2</v_quote>", verify)
        self.assertEqual(result, "<v_quote>x &lt; 2</v_quote>")
        self.assertEqual(received, ["x < 2"])

    async def test_malformed_judge_and_exhausted_pool_propagate(self):
        agent = self.agent(
            ScriptedProvider([argument("red")]),
            ScriptedProvider([argument("blue")]),
            ScriptedProvider(["Answer: C"]),
        )
        with self.assertRaises(ValueError):
            await agent.run(rounds=1)
        self.assertEqual(len(agent.turns), 2)
        self.assertEqual(agent.calls[-1].response, "Answer: C")
        agent = self.agent(
            ScriptedProvider(["<thinking>private only</thinking>"]),
            ScriptedProvider([argument("blue")]),
            ScriptedProvider([]),
        )
        with self.assertRaises(ValueError):
            await agent.run(rounds=1)
        self.assertEqual(agent.turns, [])

    async def test_swiss_balances_assignments_and_elo_fits_win_rates(self):
        calls = []

        async def play(a, b):
            calls.append((a, b))
            return 0.75 if a < b else 0.25

        result = await swiss_tournament(["a", "b", "c", "d"], play)
        self.assertEqual(len(result.matches), 4)
        self.assertEqual(len(calls), 8)
        self.assertEqual(len({frozenset((m.first, m.second)) for m in result.matches}), 4)
        ratings = fit_elo([Match("a", "b", 0.75)], reference="b")
        self.assertAlmostEqual(ratings["b"], 0)
        self.assertAlmostEqual(ratings["a"], 400 * math.log10(3), places=2)

    async def test_swiss_ties_byes_and_observed_score_errors(self):
        async def draw(a, b):
            return 0.5

        result = await swiss_tournament(["a", "b", "c"], draw, rounds=1)
        self.assertEqual(result.scores, {"a": 0.5, "b": 0.5, "c": 1.0})
        self.assertEqual(result.byes, ((1, "c"),))

        async def invalid(a, b):
            return float("nan")

        with self.assertRaises(ValueError):
            await swiss_tournament(["a", "b"], invalid)
        with self.assertRaises(ValueError):
            fit_elo([Match("a", "b", 0.5), Match("c", "d", 0.5)], reference="a")


if __name__ == "__main__":
    unittest.main()
