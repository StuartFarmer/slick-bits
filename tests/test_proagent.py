"""Offline cooperative planning checks with the shared scripted provider."""

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from pydantic import ValidationError
from slick import prompts
from slick.providers import ProviderError

from proagent import (
    Behavior,
    Observation,
    Plan,
    ProAgent,
    Skill,
    SkillCall,
    SkillFailure,
    Verification,
)
from tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1] / "proagent/prompts"


def plan(name="work", intention="fetch supplies"):
    return Plan(
        analysis="Divide the remaining work.",
        intentions={"partner": intention},
        skill=SkillCall(name=name),
    )


class ProAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        patcher.start()
        self.addCleanup(patcher.stop)

    def agent(self, responses, **overrides):
        async def verify(state, skill):
            return Verification("ready")

        async def control(state, skill):
            return skill.name

        async def step(state, action):
            return state + 1

        settings = dict(
            task="Complete a shared project",
            provider=ScriptedProvider(responses),
            skills=(
                Skill("work", "Perform the next unit of work."),
                Skill("help", "Assist partner."),
            ),
            teammates=("partner",),
            ground=lambda state: Observation(state, f"State {state}"),
            verify=verify,
            control=control,
            step=step,
        )
        settings.update(overrides)
        return ProAgent(**settings)

    async def test_active_skill_persists_until_observed_completion(self):
        async def verify(state, skill):
            return Verification("complete" if state == 2 and skill.name == "work" else "ready")

        agent = self.agent([plan(), plan("help")], verify=verify)
        result = await agent.run(0, max_steps=3)
        self.assertEqual(result.state, 3)
        self.assertEqual(result.stop_reason, "budget")
        self.assertEqual([row.action for row in result.decisions], ["work", "work", "help"])
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(agent.memory[0].status, "complete")
        self.assertEqual(agent.memory[0].actions, 2)

    async def test_invalid_precondition_explains_then_replans_before_acting(self):
        controlled = []

        async def verify(state, skill):
            return (
                Verification("invalid", "resource unavailable")
                if skill.name == "work"
                else Verification("ready")
            )

        async def control(state, skill):
            controlled.append(skill.name)
            return skill.name

        agent = self.agent(
            [plan(), "The required resource is missing; assist instead.", plan("help")],
            verify=verify,
            control=control,
        )
        result = await agent.run(0, max_steps=1)
        self.assertEqual(controlled, ["help"])
        self.assertEqual(result.state, 1)
        self.assertEqual(
            [row["operation"] for row in agent.calls], ["plan", "analyze_failure", "replan"]
        )
        self.assertIn("resource unavailable", agent.calls[1]["prompt"])
        self.assertIn("assist instead", agent.calls[2]["prompt"])

    async def test_belief_revision_uses_observations_without_overwriting_predictions(self):
        async def verify(state, skill):
            return Verification("complete" if state == 1 and skill.name == "work" else "ready")

        def ground(state):
            behaviors = (Behavior(0, "partner", "delivered supplies"),) if state else ()
            return Observation(state, f"State {state}", behaviors=behaviors)

        agent = self.agent([plan(), plan("help", "assemble")], verify=verify, ground=ground)
        await agent.run(0, max_steps=2)
        prompt = agent.calls[1]["prompt"]
        self.assertIn("fetch supplies", prompt)
        self.assertIn("delivered supplies", prompt)
        self.assertEqual(agent.memory[0].plan.intentions["partner"], "fetch supplies")
        self.assertEqual(agent.behaviors, [Behavior(0, "partner", "delivered supplies")])

    async def test_bad_generated_plans_consume_bounded_attempts_and_keep_raw_text(self):
        agent = self.agent(["not JSON", plan("unknown")], max_plan_attempts=2)
        result = await agent.run(0, max_steps=3)
        self.assertEqual(result.state, 0)
        self.assertEqual(result.stop_reason, "exhausted")
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(agent.calls[0]["response"], "not JSON")
        self.assertTrue(all(row.get("error") for row in agent.calls))

    async def test_terminal_observation_and_zero_budget_do_not_call_models(self):
        agent = self.agent([], ground=lambda state: Observation(state, "finished", terminal=True))
        result = await agent.run(0, max_steps=2)
        self.assertEqual(result.stop_reason, "terminal")
        self.assertEqual(agent.calls, [])
        other = self.agent([])
        self.assertEqual((await other.run(0, max_steps=0)).stop_reason, "budget")
        self.assertEqual(other.calls, [])

    async def test_three_round_failure_analysis_preserves_verifier_authority(self):
        async def verify(state, skill):
            return Verification("invalid", "precondition failed")

        agent = self.agent(
            [
                plan(),
                "Check found a failure.",
                "Double-check confirmed it.",
                "Try help.",
                plan("help"),
            ],
            verify=verify,
            verification_rounds=3,
            max_plan_attempts=2,
        )
        result = await agent.run(0, max_steps=1)
        self.assertEqual(result.stop_reason, "exhausted")
        self.assertEqual(
            [row["operation"] for row in agent.calls],
            ["plan", "analyze_failure", "double_check", "conclude_failure", "replan"],
        )
        self.assertIn("Double-check confirmed it", agent.calls[3]["prompt"])
        self.assertIn("Try help", agent.calls[4]["prompt"])

    async def test_controller_failure_replans_but_unexpected_errors_propagate(self):
        async def control(state, skill):
            if skill.name == "work":
                raise SkillFailure("Route blocked")
            return "assist action"

        agent = self.agent([plan(), "Choose an accessible skill.", plan("help")], control=control)
        decision = await agent.act(0)
        self.assertEqual(decision.action, "assist action")
        self.assertEqual(agent.memory[0].status, "failed")
        self.assertIn("Route blocked", agent.calls[1]["prompt"])

        async def broken(state, skill):
            raise RuntimeError("controller bug")

        agent = self.agent([plan()], control=broken)
        with self.assertRaisesRegex(RuntimeError, "controller bug"):
            await agent.act(0)
        self.assertEqual(len(agent.calls), 1)
        self.assertIn("controller bug", agent.events[-1]["error"])

    async def test_unknown_teammates_and_invalid_schema_are_rejected(self):
        unknown = plan().model_copy(update={"intentions": {"stranger": "take over"}})
        invalid = plan().model_dump()
        invalid["extra_action"] = "execute"
        agent = self.agent(
            [unknown, json.dumps(invalid), ("raw tool response", [object()])], max_plan_attempts=3
        )
        result = await agent.run(0, max_steps=1)
        self.assertEqual(result.stop_reason, "exhausted")
        self.assertEqual(agent.memory, [])
        self.assertEqual(agent.events, [])
        self.assertEqual(len(agent.attempts), 3)
        self.assertEqual(agent.calls[-1]["response"], "raw tool response")

    async def test_recent_k_and_belief_history_have_separate_scopes(self):
        async def verify(state, skill):
            complete = state > 0 and skill.name == ("work" if state == 1 else "help")
            return Verification("complete" if complete else "ready")

        agent = self.agent([plan(), plan("help"), plan()], verify=verify, recent_k=1)
        await agent.run(0, max_steps=3)
        history, beliefs = agent.retrieve()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["step"], 2)
        self.assertEqual(len(beliefs["predictions"]), 3)
        agent.recent_k = 0
        agent.belief_revision = False
        self.assertEqual(agent.retrieve(), ([], {}))

    async def test_already_completed_proposals_cannot_create_an_infinite_loop(self):
        async def verify(state, skill):
            return Verification("complete")

        agent = self.agent([plan(), plan()], verify=verify, max_plan_attempts=2)
        result = await agent.run(0, max_steps=1)
        self.assertEqual(result.stop_reason, "exhausted")
        self.assertEqual(len(agent.calls), 2)
        self.assertIn("already complete", agent.calls[1]["prompt"])

    async def test_final_observations_are_retained_and_behavior_events_deduplicated(self):
        event = Behavior(0, "partner", "observed work")

        def ground(state):
            return Observation(state, str(state), (event,) if state else (), terminal=state == 2)

        agent = self.agent([plan()], ground=ground)
        result = await agent.run(0, max_steps=2)
        self.assertEqual(result.stop_reason, "terminal")
        self.assertEqual(result.state, 2)
        self.assertEqual(agent.behaviors, [event])

    async def test_provider_explainer_and_step_errors_do_not_retry_actions(self):
        agent = self.agent([ProviderError("offline")])
        with self.assertRaisesRegex(ProviderError, "offline"):
            await agent.run(0, max_steps=1)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(agent.events, [])

        async def invalid(state, skill):
            return Verification("invalid", "unavailable")

        agent = self.agent([plan(), " "], verify=invalid)
        with self.assertRaisesRegex(ValueError, "blank"):
            await agent.run(0, max_steps=1)
        self.assertEqual(agent.calls[-1]["response"], " ")

        executed = []

        async def broken_step(state, action):
            executed.append(action)
            raise RuntimeError("step failed")

        agent = self.agent([plan()], step=broken_step)
        with self.assertRaisesRegex(RuntimeError, "step failed"):
            await agent.run(0, max_steps=2)
        self.assertEqual(executed, ["work"])

    async def test_provider_validation_errors_are_not_candidate_rejections(self):
        with self.assertRaises(ValidationError) as raised:
            Plan.model_validate({})
        agent = self.agent([raised.exception], max_plan_attempts=1)
        with self.assertRaises(ValidationError):
            await agent.act(0)
        self.assertEqual(len(agent.calls), 1)
        self.assertNotIn("response", agent.calls[0])

    async def test_cancellation_is_recorded_and_propagated(self):
        started = asyncio.Event()

        async def control(state, skill):
            started.set()
            await asyncio.Event().wait()

        agent = self.agent([plan()], control=control)
        task = asyncio.create_task(agent.act(0))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIn("CancelledError", agent.events[-1]["error"])

    async def test_all_prompt_operations_render_and_use_no_jinja_conditionals(self):
        agent = self.agent([])
        observation = Observation(0, "Observed state")
        calls = (
            (ProAgent.plan, (observation, [], {})),
            (ProAgent.replan, (observation, [], {}, "failure evidence")),
            (ProAgent.analyze_failure, (observation, {})),
            (ProAgent.double_check, (observation, {}, "initial")),
            (ProAgent.conclude_failure, (observation, {}, "initial", "review")),
        )
        for method, args in calls:
            rendered = await method.render(agent, *args)
            self.assertIn("Complete a shared project", rendered)
            self.assertIn("Observed state", rendered)
            self.assertIn("Perform the next unit", rendered)
            if method in (ProAgent.plan, ProAgent.replan):
                self.assertIn('"intentions"', rendered)
                self.assertIn("Return only JSON", rendered)
        for path in ROOT.glob("*.j2"):
            parsed = Environment().parse(path.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
