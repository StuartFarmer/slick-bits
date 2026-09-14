"""Check ExpeL's learning loop with real Slick boundaries and scripted output."""

import os
import unittest
from pathlib import Path

from jinja2 import Environment, nodes
from slick import prompts

from expel import Action, ExpeL, Experience, Insight, Outcome
from expel.agent import update_insights
from tests.providers import ScriptedProvider


class ExpeLTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "expel" / "prompts"
        previous = prompts.TEMPLATE_ROOT
        prompts.TEMPLATE_ROOT = self.root
        self.addCleanup(setattr, prompts, "TEMPLATE_ROOT", previous)
        self.resets = []
        self.actions = []
        self.embedded = []

    async def reset(self, task):
        self.resets.append(task)
        return "initial observation"

    async def step(self, action):
        self.actions.append(action)
        return Outcome("observed " + action, done=True, succeeded=action == "good", reward=0.25)

    async def embed(self, text):
        self.embedded.append(text)
        return [1.0, 0.0] if text in {"train", "held-out"} else [0.0, 2.0]

    def agent(self, responses=(), **kwargs):
        return ExpeL(
            "arbitrary task interface",
            ScriptedProvider(responses),
            self.reset,
            self.step,
            self.embed,
            **kwargs,
        )

    async def test_full_learning_loop_and_evaluation_isolation(self):
        manual = Experience("manual", "manual trace", True)
        agent = self.agent(
            [
                Action(thought="try", action="bad"),
                "check first",
                Action(thought="retry", action="good"),
                "ADD 1: Check before acting.",
                "UPVOTE 1: Check before acting.",
                Action(thought="use memory", action="good"),
            ],
            manual_examples=(manual,),
        )
        result = await agent.run(["train"], ["held-out"], max_retries=2, k=1)
        self.assertEqual(self.resets, ["train", "train", "held-out"])
        self.assertEqual(result.success_rate, 1.0)  # success is independent of reward
        self.assertEqual(len(agent.pool), 3)
        self.assertEqual([item.succeeded for item in agent.pool], [True, False, True])
        self.assertEqual(agent.insights, (Insight("Check before acting.", 3),))
        self.assertEqual(
            [c["operation"] for c in agent.calls],
            ["act", "reflect", "act", "compare", "summarize", "act"],
        )
        retry = agent.calls[2]["prompt"]
        self.assertIn("check first", retry)
        self.assertNotIn("observed bad", retry)
        self.assertNotIn("check first", agent.calls[3]["prompt"])
        final = agent.calls[-1]["prompt"]
        self.assertIn("Check before acting.", final)
        self.assertIn("observed good", final)
        self.assertNotIn("manual trace", final)
        self.assertNotIn("observed bad", final)
        self.assertNotIn("check first", final)

    async def test_retry_and_step_budgets_and_reflection_scope(self):
        agent = self.agent(
            [
                Action(thought="", action="bad"),
                "lesson one",
                Action(thought="", action="bad"),
                "lesson two",
                Action(thought="", action="bad"),
                Action(thought="", action="good"),
            ]
        )
        experiences = await agent.gather(["first", "second"], max_retries=2, max_steps=5)
        self.assertEqual(len(experiences), 4)
        self.assertEqual(self.resets, ["first", "first", "first", "second"])
        self.assertIn("lesson one", agent.calls[4]["prompt"])
        self.assertIn("lesson two", agent.calls[4]["prompt"])
        self.assertNotIn("lesson one", agent.calls[-1]["prompt"])

        async def continuing(action):
            return Outcome("keep going")

        bounded = self.agent([Action(thought="", action="next")] * 2)
        bounded.step = continuing
        result = await bounded.gather(["limit"], max_retries=0, max_steps=2)
        self.assertFalse(result[0].succeeded)
        self.assertEqual(result[0].steps, 2)
        self.assertIn("budget", result[0].feedback.lower())
        zero = await self.agent().gather(["zero"], max_retries=0, max_steps=0)
        self.assertEqual(zero[0].steps, 0)

    async def test_extraction_pairs_then_distinct_success_chunks(self):
        agent = self.agent(["NONE"] * 6)
        agent.pool = [
            Experience("a", "failure a1", False),
            Experience("a", "failure a2", False),
            Experience("a", "success a", True),
            Experience("a", "other success a", True),
            Experience("b", "success b", True),
            Experience("c", "success c", True),
            Experience("d", "unpaired failure", False),
        ]
        await agent.extract_insights(chunk_size=2, seed=7)
        self.assertEqual([c["operation"] for c in agent.calls], ["compare"] * 4 + ["summarize"] * 2)
        chunks = "\n".join(c["prompt"] for c in agent.calls[4:])
        for trace in ("success a", "success b", "success c"):
            self.assertEqual(chunks.count(trace), 1)
        self.assertNotIn("other success a", chunks)
        self.assertNotIn("unpaired failure", "\n".join(c["prompt"] for c in agent.calls))

    async def test_retrieval_uses_task_inner_product_successes_and_stable_ties(self):
        agent = self.agent()
        a = Experience("train", "unrelated trace", True)
        b = Experience("other", "train", True)
        agent.pool = [
            Experience("held-out", "failed", False),
            a,
            b,
            Experience("train", "duplicate successful task", True),
        ]
        self.assertEqual(await agent.retrieve("held-out", k=1), (a,))
        self.assertEqual(await agent.retrieve("other", k=3), (b, a))
        self.assertNotIn("unrelated trace", self.embedded)
        self.assertEqual(self.embedded.count("train"), 1)
        calls = len(self.embedded)
        self.assertEqual(await agent.retrieve("unused", k=0), ())
        self.assertEqual(len(self.embedded), calls)

    async def test_transfer_and_fixed_target_demonstrations(self):
        source = self.agent(
            ["Verify the available evidence."], insights=(Insight("Inspect evidence.", 5),)
        )
        demo = Experience("target example", "target trace", True)
        paragraph = await source.adapt_insights("target interface", (demo,))
        self.assertIn("target trace", source.calls[0]["prompt"])
        target = self.agent(
            [Action(thought="", action="good")],
            manual_examples=(demo,),
            insights=(Insight(paragraph),),
        )
        result = await target.evaluate(["new target"], k=0)
        self.assertEqual(result.success_rate, 1.0)
        self.assertIn(paragraph, target.calls[0]["prompt"])
        self.assertIn("target trace", target.calls[0]["prompt"])
        self.assertEqual(len(target.pool), 1)

    async def test_failures_are_logged_and_do_not_retry_or_mutate_insights(self):
        for response in [
            "not json",
            '{"thought":"x","action":" "}',
            ("raw", ["unexpected tool"]),
            TimeoutError("offline"),
        ]:
            agent = self.agent([response])
            with self.assertRaises((ValueError, TimeoutError)):
                await agent.gather(["task"])
            self.assertEqual(len(agent.calls), 1)
            self.assertIn("error", agent.calls[0])
            self.assertEqual(agent.pool, [])
        agent = self.agent(["ADD 1: Valid.\nEDIT 8: Invalid."])
        agent.pool = [Experience("task", "success", True)]
        with self.assertRaises(ValueError):
            await agent.extract_insights()
        self.assertEqual(agent.insights, ())
        self.assertIn("EDIT 8", agent.calls[0]["response"])

    async def test_provider_routing_and_extraction_rebuild(self):
        reflector = ScriptedProvider(["retry differently"])
        extractor = ScriptedProvider(["ADD 1: Inspect first.", "UPVOTE 1: Inspect first."] * 2)
        agent = self.agent(
            [Action(thought="", action="bad"), Action(thought="", action="good")],
            reflection_provider=reflector,
            insight_provider=extractor,
        )
        await agent.gather(["train"], max_retries=1)
        first = await agent.extract_insights()
        self.assertEqual(first, await agent.extract_insights())
        self.assertEqual(len(agent.provider.calls), 2)
        self.assertEqual(len(reflector.calls), 1)
        self.assertEqual(len(extractor.calls), 4)

    async def test_evaluation_failures_do_not_retry_or_enter_training_memory(self):
        agent = self.agent([Action(thought="", action="bad")] * 2)
        result = await agent.evaluate(["one", "two"], k=0)
        self.assertEqual(result.success_rate, 0.0)
        self.assertEqual(self.resets, ["one", "two"])
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(agent.pool, [])
        self.assertEqual(agent.insights, ())
        self.assertEqual((await agent.evaluate([], k=0)).success_rate, 0.0)

    async def test_callback_errors_preserve_partial_state_without_retries(self):
        async def failed_step(action):
            raise RuntimeError("environment unavailable")

        agent = self.agent([Action(thought="", action="execute")])
        agent.step = failed_step
        with self.assertRaisesRegex(RuntimeError, "environment unavailable"):
            await agent.gather(["task"], max_retries=3)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(agent.pool, [])
        self.assertIn("Action 1: execute", agent.trajectory)

        async def bad_embedding(text):
            return [float("nan")]

        agent.pool = [Experience("task", "success", True)]
        agent.embed = bad_embedding
        with self.assertRaisesRegex(ValueError, "non-finite"):
            await agent.retrieve("query")

    async def test_templates_render_from_another_directory_without_branches(self):
        previous = os.getcwd()
        os.chdir("/tmp")
        self.addCleanup(os.chdir, previous)
        agent = self.agent(
            [Action(thought="", action="good"), "reflection", "NONE", "NONE", "transfer"]
        )
        experience = Experience("task", "trace", True)
        await agent.evaluate(["task"], k=0)
        await agent._invoke(agent.reflect, experience, ())
        await agent._invoke(agent.compare, experience, Experience("task", "bad", False))
        await agent._invoke(agent.summarize, (experience,))
        await agent.adapt_insights("target", (experience,))
        self.assertIn('"action"', agent.calls[0]["prompt"])
        for path in self.root.glob("*.j2"):
            tree = Environment().parse(path.read_text())
            self.assertEqual(list(tree.find_all((nodes.If, nodes.CondExpr))), [])


class InsightTests(unittest.TestCase):
    def test_votes_edit_add_deletion_and_snapshot_indices(self):
        initial = (Insight("First.", 1), Insight("Second.", 2), Insight("Third.", 3))
        updated = update_insights(
            initial, "DOWNVOTE 1: First.\nEDIT 2: Improved.\nUPVOTE 3: Third.\nADD 4: New."
        )
        self.assertEqual(
            updated, (Insight("Third.", 4), Insight("Improved.", 3), Insight("New.", 2))
        )
        self.assertEqual(initial[0].importance, 1)
        self.assertEqual(update_insights(updated, "NONE"), updated)

    def test_invalid_batches_are_rejected(self):
        for text in [
            "",
            "UPVOTE 0: One.",
            "UPVOTE 2: One.",
            "UPVOTE 1: Wrong.",
            "ADD 2: One.",
            "ADD 7: New.",
            "UPVOTE 1: One.\nEDIT 1: Other.",
            "ADD 2: New.\nUPVOTE 2: New.",
            "ADD 2: ",
            "\n".join(f"ADD {i}: Rule {i}." for i in range(2, 7)),
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                update_insights((Insight("One."),), text)
