"""Exercise CRITIC's real tool loop with scripted model responses."""

import asyncio
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, nodes
from slick import prompts

from critic import CRITIC, Critique
from critic.official_tools import official_google
from tests.providers import NativeScriptedProvider, ScriptedProvider


def request(query):
    return ("Check the source", [{"id": query, "name": "lookup", "arguments": {"query": query}}])


async def lookup(query: str) -> str:
    """Read an external specification by key."""
    return {"colour": "blue", "size": "large"}[query]


class CriticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        previous = prompts.TEMPLATE_ROOT
        self.addCleanup(setattr, prompts, "TEMPLATE_ROOT", previous)
        prompts.TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "critic/prompts"

    async def test_native_provider_continuation_and_pause_are_preserved(self):
        provider = NativeScriptedProvider(
            [
                {
                    "text": "checking",
                    "requests": request("colour")[1],
                    "status": "tools",
                    "continuation": {"opaque": "one"},
                },
                {
                    "text": "still checking",
                    "requests": [],
                    "status": "continue",
                    "continuation": {"opaque": "two"},
                },
                {
                    "text": Critique(correct=True, feedback="Supported").model_dump_json(),
                    "requests": [],
                    "status": "complete",
                    "continuation": {"opaque": "three"},
                },
            ]
        )
        agent = CRITIC("Task", provider, [lookup])
        result = await agent.run("Input", initial_output="blue")
        self.assertEqual(result.stop_reason, "verified")
        self.assertEqual(result.calls, 3)
        self.assertEqual(provider.calls[1]["continuation"], {"opaque": "one"})
        self.assertEqual(provider.calls[2]["continuation"], {"opaque": "two"})
        self.assertEqual(provider.calls[1]["results"][0]["content"], "blue")
        self.assertEqual(agent.calls[1]["response"], "still checking")

    async def test_official_crawler_adapter_owns_a_worker_loop_and_formats_evidence(self):
        async def page():
            return "external evidence"

        def search(query, **kwargs):
            self.assertEqual(query, "claim")
            self.assertEqual(kwargs, {"cache": True, "page_cache": True, "topk": 2})
            text = asyncio.get_event_loop().run_until_complete(page())
            return {"title": "Source", "link": "https://example.test", "page": text}

        tool = official_google(SimpleNamespace(search=search), evidence_length=8)
        evidence = json.loads(await tool.ainvoke({"query": "claim", "topk": 2}))
        self.assertEqual(
            evidence, {"title": "Source", "url": "https://example.test", "page": "external"}
        )

    async def test_interleaves_tools_revises_and_verifies_current_output(self):
        provider = ScriptedProvider(
            [
                "red",
                request("colour"),
                request("size"),
                Critique(correct=False, feedback="Use blue and large."),
                "large blue",
                request("colour"),
                Critique(correct=True, feedback="Matches the source."),
            ]
        )
        agent = CRITIC("Match the specification", provider, [lookup])
        result = await agent.run("Describe the object")
        self.assertEqual(result.output, "large blue")
        self.assertEqual(result.stop_reason, "verified")
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.calls, 7)
        self.assertIn("blue", provider.calls[2])
        self.assertIn("large", provider.calls[4])
        self.assertIn("Use blue and large.", provider.calls[4])
        self.assertEqual(result.history[0].output, "red")
        self.assertNotIn("Use blue and large.", provider.calls[5])
        self.assertEqual(len(result.history[0].evidence), 2)

    async def test_algorithm_one_budget_returns_last_revision_without_extra_verification(self):
        for budget in (0, 1, 3):
            responses = ["initial"]
            for index in range(budget):
                responses.extend(
                    [request("colour"), Critique(correct=False, feedback="Fix"), str(index)]
                )
            result = await CRITIC("Task", ScriptedProvider(responses), [lookup]).run(
                "Input", max_iterations=budget
            )
            self.assertEqual(result.output, str(budget - 1) if budget else "initial")
            self.assertEqual(result.stop_reason, "budget")
            self.assertEqual(result.calls, 1 + 3 * budget)
            self.assertEqual(len(result.history), budget)

    async def test_no_evidence_cannot_be_accepted_and_raw_failure_survives(self):
        for response in (
            "not json",
            '{"correct": "yes", "feedback": "ok"}',
            Critique(correct=True, feedback="I think so"),
        ):
            agent = CRITIC("Task", ScriptedProvider([response]), [lookup])
            with self.assertRaises(ValueError):
                await agent.run("Input", initial_output="draft")
            self.assertEqual(agent.output, "draft")
            self.assertIn("response", agent.calls[-1])
            self.assertIn("error", agent.calls[-1])

    async def test_tool_errors_are_feedback_and_turn_budget_does_not_execute_pending_work(self):
        provider = ScriptedProvider(
            [request("missing"), Critique(correct=False, feedback="No evidence"), "unknown"]
        )
        result = await CRITIC("Task", provider, [lookup]).run(
            "Input", initial_output="draft", max_iterations=1
        )
        self.assertTrue(result.history[0].evidence[0]["is_error"])
        self.assertIn("missing", provider.calls[1])
        agent = CRITIC("Task", ScriptedProvider([request("colour")] * 8), [lookup])
        with self.assertRaisesRegex(RuntimeError, "turn limit"):
            await agent.run("Input", initial_output="draft")
        self.assertEqual(len(agent.calls), 8)
        self.assertEqual(len(agent.verification_session.pending_requests), 1)
        self.assertEqual(sum(len(e["work"]) for e in agent.verification_session.history), 8)

    async def test_stagnation_is_not_verification_and_runs_reset(self):
        responses = [request("colour"), Critique(correct=False, feedback="Fix"), "draft"] * 2
        agent = CRITIC("Task", ScriptedProvider(responses), [lookup])
        result = await agent.run("Input", initial_output="draft", unchanged_patience=2)
        self.assertEqual(result.stop_reason, "unchanged")
        self.assertEqual(len(result.history), 2)
        reset = await agent.run("Other", initial_output="supplied", max_iterations=0)
        self.assertEqual(reset.calls, 0)
        self.assertEqual(reset.history, ())
        self.assertEqual(len(result.history), 2)

    async def test_blank_and_transport_failures_preserve_current_output(self):
        for bad in ("  ", TimeoutError("offline")):
            agent = CRITIC(
                "Task", ScriptedProvider([Critique(correct=False, feedback="Fix"), bad]), [lookup]
            )
            with self.assertRaises((ValueError, TimeoutError)):
                await agent.run("Input", initial_output="draft")
            self.assertEqual(agent.output, "draft")
            self.assertEqual(len(agent.history), 1)
            self.assertIn("error", agent.calls[-1])

    async def test_all_templates_render_from_another_directory(self):
        agent = CRITIC("Task", ScriptedProvider([]), [lookup], examples="Demonstrations")
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(prompts.TEMPLATE_ROOT.parent)
        rendered = [
            await CRITIC.generate.render(agent, "Input"),
            await CRITIC.critique.render(agent, "Input", "draft"),
            await CRITIC.revise.render(
                agent, "Input", "draft", Critique(correct=False, feedback="Fix"), ()
            ),
        ]
        for text in rendered:
            self.assertIn("Task", text)
            self.assertIn("Input", text)
            self.assertIn("Demonstrations", text)
        self.assertIn('"correct"', rendered[1])
        for template in prompts.TEMPLATE_ROOT.glob("*.j2"):
            self.assertEqual(
                list(
                    Environment().parse(template.read_text()).find_all((nodes.If, nodes.CondExpr))
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
