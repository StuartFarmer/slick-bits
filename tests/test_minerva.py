"""Offline checks of Minerva sampling, answer voting, and Slick boundaries."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from minerva import Example, Minerva, extract_final_answer
from tests.providers import ScriptedProvider


def solution(answer, rationale="A solution"):
    return f"{rationale}\nFinal Answer: The final answer is {answer}. I hope it is correct."


class MinervaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "minerva/prompts"
        patcher = patch("slick.prompts.TEMPLATE_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_plurality_top_n_and_independent_samples(self):
        responses = [solution(a, f"Unique rationale {i}") for i, a in enumerate("abbcc")]
        provider = ScriptedProvider(responses)
        agent = Minerva("Classify this document {{ literally }}", provider)
        result = await agent.run(k=5, n=2)
        self.assertEqual(result.answer, "b")
        self.assertEqual(result.output, responses[1])
        self.assertEqual([vote.answer for vote in result.selected], ["b", "c"])
        self.assertEqual([vote.count for vote in result.votes], [2, 2, 1])
        self.assertEqual(result.votes[0].sample_indices, (1, 2))
        self.assertEqual(len(set(provider.calls)), 1)
        self.assertNotIn("Unique rationale", provider.calls[0])
        self.assertEqual(result.calls, 5)
        self.assertIsNone(result.pass_at_k)

    async def test_invalid_answers_consume_budget_and_do_not_vote(self):
        provider = ScriptedProvider(["", "unfinished", solution(""), solution("valid")])
        agent = Minerva("Any task", provider)
        result = await agent.run(k=4)
        self.assertEqual(result.answer, "valid")
        self.assertEqual(result.votes[0].count, 1)
        self.assertEqual(len(result.samples), 4)
        self.assertTrue(all(sample.error for sample in result.samples[:3]))
        self.assertEqual(agent.calls[0]["response"], "")
        empty = await Minerva("Any task", ScriptedProvider(["bad"])).run(k=1)
        self.assertIsNone(empty.answer)
        self.assertIsNone(empty.output)
        self.assertEqual(empty.selected, ())

    async def test_custom_extraction_normalization_and_posthoc_evaluation(self):
        seen = []
        provider = ScriptedProvider([" RED ", "red", "blue"])

        async def evaluate(answer):
            self.assertEqual(len(provider.calls), 3)
            seen.append(answer)
            return answer == "blue"

        agent = Minerva(
            "Assign a colour",
            provider,
            evaluate,
            extract=str.strip,
            normalize=str.casefold,
        )
        result = await agent.run(k=3, n=1)
        self.assertEqual(result.answer, "RED")
        self.assertEqual(result.votes[0].count, 2)
        self.assertEqual(seen, ["RED", "red", "blue"])
        self.assertTrue(result.pass_at_k)
        self.assertFalse(result.maj_at_n)
        self.assertEqual(result.correctness, (False, False, True))

    async def test_failures_propagate_and_keep_records_without_retries(self):
        for failure in (TimeoutError("offline"), ("raw", ["tool request"])):
            agent = Minerva("Task", ScriptedProvider([solution("ok"), failure]))
            with self.assertRaises((TimeoutError, ValueError)):
                await agent.run(k=3)
            self.assertEqual(len(agent.calls), 2)
            self.assertEqual(len(agent.samples), 1)
            self.assertIn("error", agent.calls[-1])
        agent = Minerva("Task", ScriptedProvider([solution("ok")]), normalize=lambda _: 1 / 0)
        with self.assertRaises(ZeroDivisionError):
            await agent.run(k=1)
        self.assertEqual(agent.calls[0]["response"], solution("ok"))

    async def test_evaluation_failure_and_run_reset(self):
        async def evaluate(answer):
            raise RuntimeError("grader failed")

        agent = Minerva("Task", ScriptedProvider([solution("a"), solution("b")]), evaluate)
        with self.assertRaisesRegex(RuntimeError, "grader failed"):
            await agent.run(k=2)
        self.assertEqual(len(agent.samples), 2)
        agent = Minerva("Task", ScriptedProvider([solution("a"), solution("b")]))
        first = await agent.run(k=1)
        second = await agent.run(k=1)
        self.assertEqual((first.answer, second.answer), ("a", "b"))
        self.assertEqual(second.calls, 1)
        self.assertEqual(len(agent.samples), 1)

    async def test_template_examples_binding_and_launch_directories(self):
        example = Example("Example question", "Example solution", "label")
        agent = Minerva("Task {{ literal }}", ScriptedProvider([]), examples=[example])
        previous = Path.cwd()
        try:
            for directory in (self.root.parent, self.root.parent.parent):
                os.chdir(directory)
                rendered = await Minerva.solve.render(agent)
                self.assertIn("Task {{ literal }}", rendered)
                self.assertIn(solution("label", "Example solution"), rendered)
                self.assertTrue(rendered.endswith("Solution:"))
        finally:
            os.chdir(previous)
        parsed = Environment().parse((self.root / "solve.j2").read_text())
        self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])

    def test_extraction_preserves_domain_text_and_rejects_truncation(self):
        for answer in ("3.14", r"\frac{1}{\sqrt{3}}", "A.B.", "a = b", "line one\nline two"):
            self.assertEqual(extract_final_answer(solution(answer)), answer)
        self.assertEqual(extract_final_answer(solution("old") + "\n" + solution("new")), "new")
        self.assertEqual(extract_final_answer(solution("yes").replace("Answer:", "answer:")), "yes")
        self.assertIsNone(extract_final_answer("Final Answer: The final answer is unfinished"))
        self.assertIsNone(extract_final_answer(r"The answer is \boxed{42}"))


if __name__ == "__main__":
    unittest.main()
