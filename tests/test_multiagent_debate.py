"""Offline checks of debate snapshots, isolated histories, and generation budgets."""

import unittest
from pathlib import Path
from unittest.mock import patch

from multiagent_debate import MultiAgentDebate
from tests.providers import ScriptedProvider


class DebateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = patch(
            "slick.prompts.TEMPLATE_ROOT",
            Path(__file__).resolve().parents[1] / "multiagent_debate/prompts",
        )
        self.root.start()
        self.addCleanup(self.root.stop)

    async def test_rounds_read_previous_peers_and_keep_separate_histories(self):
        provider = ScriptedProvider(["A0", "B0", "C0", "A1", "B1", "C1", "A2", "B2", "C2"])
        agent = MultiAgentDebate("Design a naming scheme.", [provider] * 3)
        result = await agent.run()
        self.assertEqual(
            result.rounds, (("A0", "B0", "C0"), ("A1", "B1", "C1"), ("A2", "B2", "C2"))
        )
        self.assertEqual(result.responses, ("A2", "B2", "C2"))
        self.assertEqual(result.calls, 9)
        self.assertEqual(provider.calls[:3], ["Design a naming scheme."] * 3)
        for index, peers in ((3, ("B0", "C0")), (4, ("A0", "C0")), (5, ("A0", "B0"))):
            current = provider.calls[index].split("Conversation so far:")[0]
            for peer in peers:
                self.assertIn(f"```{peer}```", current)
            self.assertNotIn("1```", current)
        self.assertNotIn("```B0```", provider.calls[4].split("Conversation so far:")[0])
        self.assertIn("```A1```", provider.calls[7])
        self.assertIn("Design a naming scheme.", provider.calls[7])
        self.assertEqual([entry["text"] for entry in agent.sessions[1].history], ["B0", "B1", "B2"])
        self.assertIn("```C0```", agent.sessions[1].history[1]["context"])

    async def test_initialization_only_and_fresh_sessions_on_rerun(self):
        first = ScriptedProvider(["red", "new red"])
        second = ScriptedProvider(["blue", "new blue"])
        agent = MultiAgentDebate("Pick a color.", [first, second], debate_rounds=0)
        self.assertEqual((await agent.run()).responses, ("red", "blue"))
        self.assertEqual((await agent.run()).responses, ("new red", "new blue"))
        self.assertEqual(agent.calls, 2)
        self.assertEqual(len(agent.sessions[0].history), 1)
        self.assertEqual(first.calls, ["Pick a color."] * 2)

    async def test_single_agent_reflects_without_summary_calls(self):
        provider = ScriptedProvider(["draft", "revision", "final"])
        summary = ScriptedProvider([])
        agent = MultiAgentDebate("Write a welcome message.", [provider], summarizer=summary)
        self.assertEqual((await agent.run()).responses, ("final",))
        self.assertEqual(agent.calls, 3)
        self.assertEqual(summary.calls, [])
        self.assertIn("verify", provider.calls[1])
        self.assertIn("draft", provider.calls[1])

    async def test_summary_excludes_self_and_uses_previous_round(self):
        provider = ScriptedProvider(["A0", "B0", "A1", "B1"])
        summary = ScriptedProvider(["summary of B", "summary of A"])
        agent = MultiAgentDebate(
            "Compare two designs.",
            [provider] * 2,
            debate_rounds=1,
            style="short",
            summarizer=summary,
        )
        result = await agent.run()
        self.assertEqual(result.calls, 6)
        self.assertEqual(result.responses, ("A1", "B1"))
        self.assertIn("```B0```", summary.calls[0])
        self.assertNotIn("A0", summary.calls[0])
        self.assertIn("```A0```", summary.calls[1])
        self.assertNotIn("A1", summary.calls[1])
        self.assertIn("```summary of B```", provider.calls[2])
        self.assertIn("Based off", provider.calls[2])
        self.assertNotIn("Conversation so far:", summary.calls[1])

    async def test_failures_preserve_raw_output_and_completed_rounds_without_retries(self):
        for failure, error in ((OSError("offline"), OSError), ("  ", ValueError)):
            with self.subTest(failure=failure):
                provider = ScriptedProvider(["A0", "B0", failure])
                agent = MultiAgentDebate("Task", [provider] * 2)
                with self.assertRaises(error):
                    await agent.run()
                self.assertEqual(agent.calls, 3)
                self.assertEqual(agent.rounds, [("A0", "B0")])
                self.assertEqual(len(provider.calls), 3)
                if error is ValueError:
                    self.assertEqual(agent.generations[-1], "  ")
                    self.assertEqual(agent.sessions[0].history[-1]["text"], "  ")


if __name__ == "__main__":
    unittest.main()
