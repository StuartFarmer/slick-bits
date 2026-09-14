"""Check SCULPT's assessment order, local actions, grouping, and beam selection."""

import unittest
from pathlib import Path
from unittest.mock import patch

from sculpt import SCULPT, Section
from sculpt.agent import Action, Actions, Assessment, Examples, Feedback, InvalidAction
from tests.providers import ScriptedProvider


class SCULPTTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "sculpt/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_structure_precedes_errors_grouped_feedback_and_examples(self):
        feedback = Feedback(references=[["Task"]], feedback="Clarify the rule")
        provider = ScriptedProvider(
            [
                Assessment(feedback=[feedback]),
                Actions(actions=[Action(kind="rephrase", path=["Task"], body="improved")]),
                Assessment(feedback=[feedback]),
                Assessment(feedback=[feedback]),
                Actions(
                    actions=[Action(kind="examples", path=["Task"], instruction="Add illustration")]
                ),
                Examples(examples=["useful illustration"]),
            ]
        )
        observed = []

        async def errors(text):
            observed.append(text)
            return [{"input": "x", "prediction": "wrong", "expected": "right"}]

        async def evaluate(text):
            return len(text)

        original = [Section(title="Task", body="old"), Section(title="Fixed", body="keep")]
        result = await SCULPT("Write manuals", provider, evaluate, errors).run(
            original, iterations=1, beam_size=1, error_batches=2
        )
        self.assertIn("improved", observed[0])
        self.assertIn("useful illustration", result["best"]["prompt"])
        self.assertIn("# Fixed\nkeep", result["best"]["prompt"])
        self.assertEqual(original[0].body, "old")
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(result["error_evaluations"], 1)
        self.assertEqual(result["optimizer_calls"], 6)
        self.assertEqual(len(result["history"]), 2)
        for context in provider.calls:
            self.assertIn("Write manuals", context)

    async def test_actions_preserve_tree_and_reject_invalid_paths_or_reorders(self):
        async def evaluate(text):
            return 1

        async def errors(text):
            return []

        agent = SCULPT("Task", ScriptedProvider([]), evaluate, errors)
        original = [Section(title="A", body="a"), Section(title="B", body="b")]
        changed = await agent._apply(
            original,
            Actions(
                actions=[
                    Action(
                        kind="create", path=[], position=1, section=Section(title="C", body="c")
                    ),
                    Action(kind="reorder", path=[], order=["B", "C", "A"]),
                    Action(kind="rephrase", path=["A"], body="new"),
                ]
            ),
        )
        self.assertEqual([node.title for node in changed], ["B", "C", "A"])
        self.assertEqual(original[0].body, "a")
        changed = await agent._apply(
            changed,
            Actions(
                actions=[
                    Action(
                        kind="merge",
                        path=[],
                        sources=[["B"], ["C"]],
                        position=0,
                        section=Section(title="Merged", body="combined"),
                    ),
                    Action(kind="delete", path=["A"]),
                ]
            ),
        )
        self.assertEqual([node.title for node in changed], ["Merged"])
        for action in (
            Action(kind="delete", path=["missing"]),
            Action(kind="reorder", path=[], order=["A", "A"]),
        ):
            with self.assertRaises(InvalidAction):
                await agent._apply(original, Actions(actions=[action]))

    async def test_bad_actor_rejected_without_evaluation_and_parent_retained(self):
        feedback = Feedback(references=[["A"]], feedback="Improve")
        provider = ScriptedProvider(
            [
                Assessment(feedback=[feedback]),
                Actions(actions=[Action(kind="delete", path=["absent"])]),
            ]
        )

        async def evaluate(text):
            return 1

        async def errors(text):
            return []

        result = await SCULPT("Task", provider, evaluate, errors).run(
            [Section(title="A", body="old")], iterations=1
        )
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["best"]["tree"][0].body, "old")
        self.assertIn("absent", result["history"][0]["rejection"])

    async def test_example_provider_valueerror_is_not_an_edit_rejection(self):
        feedback = Feedback(references=[["A"]], feedback="Improve")
        provider = ScriptedProvider(
            [
                Assessment(feedback=[feedback]),
                Actions(actions=[Action(kind="examples", path=["A"], instruction="Improve")]),
                ValueError("provider failed"),
            ]
        )

        async def evaluate(text):
            return 1

        async def errors(text):
            return []

        with self.assertRaisesRegex(ValueError, "provider failed"):
            await SCULPT("Task", provider, evaluate, errors).run([Section(title="A")])


if __name__ == "__main__":
    unittest.main()
