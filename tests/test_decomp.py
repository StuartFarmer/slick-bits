"""Offline checks of DECOMP's execution semantics using the shared provider."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from decomp import Decomp, Program, paper_agent
from decomp.library import Answer, Handlers
from tests.providers import ScriptedProvider


async def echo(question):
    return question


class DecompTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).parents[1] / "decomp/prompts"
        self.root = patch("slick.prompts.TEMPLATE_ROOT", root)
        self.root.start()
        self.addCleanup(self.root.stop)

    async def test_hierarchical_letters_with_real_slick_boundaries(self):
        provider = ScriptedProvider(
            [
                'QS: [split] What are the words in "John Smith"?',
                '{"answer": ["John", "Smith"]}',
                'QS: (project_values) [str_position] What is the second letter in "#1"?',
                'QS: [split] What are the letters in "John"?',
                '{"answer": ["J", "o", "h", "n"]}',
                "QS: [arr_position] What is at position 2 in #1?",
                '{"answer": "o"}',
                "[EOQ]",
                'QS: [split] What are the letters in "Smith"?',
                '{"answer": ["S", "m", "i", "t", "h"]}',
                "QS: [arr_position] What is at position 2 in #1?",
                '{"answer": "m"}',
                "[EOQ]",
                "QS: [merge] Concatenate #2 using a space.",
                '{"answer": "o m"}',
                "QS: [EOQ]",
            ]
        )
        agent = paper_agent(provider)
        result = await agent.run('Concatenate second letters of "John Smith".', program="letters")
        self.assertEqual(result.answer, "o m")
        self.assertEqual(len(result.steps), 7)
        self.assertIn('#2: ["o", "m"]', provider.calls[-1])
        self.assertIn('What is the second letter in "Smith"?', provider.calls[8])
        self.assertEqual(result.calls, len(provider.calls))

    async def test_recursive_history_is_local(self):
        provider = ScriptedProvider(
            [
                "[reverse] small",
                "[echo] child",
                "[EOQ]",
                "[echo] parent saw #1",
                "[EOQ]",
            ]
        )
        agent = Decomp("Reverse lists", provider, {"echo": echo}, {"reverse": Program("examples")})
        result = await agent.run("large", program="reverse")
        self.assertEqual(result.answer, "parent saw child")
        self.assertNotIn("QC: large", provider.calls[1])
        self.assertIn('#1: "child"', provider.calls[3])

    async def test_foreach_merge_is_flat_unique_and_ordered(self):
        async def source(question):
            return ["a", "b", "a"]

        async def lookup(question):
            return [{"name": question}, {"name": "shared"}]

        provider = ScriptedProvider(
            ["[source] input", "(project_values_flat.unique) [lookup] #1", "[EOQ]"]
        )
        agent = Decomp("task", provider, {"source": source, "lookup": lookup})
        result = await agent.run("question")
        self.assertEqual(result.answer, [{"name": "a"}, {"name": "shared"}, {"name": "b"}])

    async def test_references_are_single_pass_and_token_exact(self):
        responses = ["[echo] value"] * 9 + ["[echo] #1 #9", "[echo] #10 / #1", "[EOQ]"]
        agent = Decomp("task", ScriptedProvider(responses), {"echo": echo})
        self.assertEqual((await agent.run("q")).answer, "value value / value")

        async def source(question):
            return ["#99", 'a"b']

        provider = ScriptedProvider(["[source] input", '[foreach] [echo] "#1"', "[EOQ]"])
        agent = Decomp("task", provider, {"source": source, "echo": echo})
        self.assertEqual((await agent.run("q")).answer, ['"#99"', '"a\\"b"'])

    async def test_invalid_programs_fail_before_dispatch(self):
        for bad in (
            "[missing] q",
            "[EOQ]",
            "[echo] #1",
            "[echo] #0",
            "[echo] q\nA: invented",
            "(bad) [echo] q",
            "[echo]",
            "[echo] q\n[echo] second",
        ):
            with self.subTest(bad=bad):
                agent = Decomp("task", ScriptedProvider([bad]), {"echo": echo})
                with self.assertRaises(ValueError):
                    await agent.run("q")
                self.assertEqual(agent.generations[0], bad)

    async def test_each_leaf_renders_and_parses_with_owner_bound(self):
        library = Handlers()
        library.context = "a context fact"
        for name in (
            "split",
            "arr_position",
            "merge",
            "list_split",
            "reverse_base",
            "qa",
            "singlehop_qa",
            "multihop_qa",
            "cot",
            "gpt_ans",
        ):
            with self.subTest(name=name):
                method = getattr(Handlers, name)
                rendered = await method.render(library, "unique question")
                self.assertIn("unique question", rendered)
                self.assertIn('"required": ["answer"]', rendered)
                result = await getattr(library, name)(
                    "question", provider=ScriptedProvider(['{"answer": null}'])
                )
                self.assertEqual(result, Answer(answer=None))
        agent = paper_agent(ScriptedProvider(["[split] q", "{}"]))
        with self.assertRaises(ValidationError):
            await agent.run("q")
        self.assertEqual(agent.calls, 2)
        self.assertEqual(agent.steps, [])

    async def test_math_and_null_values_reach_eoq(self):
        provider = ScriptedProvider(
            [
                "[cot] How many packs?",
                '{"answer": "4 * 30 / 15 = 8 packs for 30 days."}',
                "[gpt_ans] #1",
                '{"answer": 8}',
                "[EOQ]",
                "[arr_position] position 9 in []",
                '{"answer": null}',
                "[EOQ]",
            ]
        )
        agent = paper_agent(provider)
        self.assertEqual((await agent.run("How many packs?", program="math")).answer, 8)
        self.assertIn("8 packs for 30 days", provider.calls[3])
        self.assertIsNone((await agent.run("empty array")).answer)
        self.assertEqual(agent.calls, 3)

    async def test_bundled_reversal_and_shared_budget(self):
        responses = [
            '[list_split] Split ["a", "b", "c", "d"].',
            '{"answer": [["a", "b"], ["c", "d"]]}',
            "[arr_position] position 1 in #1",
            '{"answer": ["a", "b"]}',
            "[arr_position] position 2 in #1",
            '{"answer": ["c", "d"]}',
            "[reverse] Reverse #2.",
            '[reverse_base] Reverse ["a", "b"].',
            '{"answer": ["b", "a"]}',
            "[EOQ]",
            "[reverse] Reverse #3.",
            '[reverse_base] Reverse ["c", "d"].',
            '{"answer": ["d", "c"]}',
            "[EOQ]",
            "[merge] Concatenate lists #5 and #4.",
            '{"answer": ["d", "c", "b", "a"]}',
            "[EOQ]",
        ]
        agent = paper_agent(ScriptedProvider(responses))
        result = await agent.run('Reverse ["a", "b", "c", "d"].', program="reverse")
        self.assertEqual(result.answer, ["d", "c", "b", "a"])
        self.assertEqual(result.calls, 17)
        self.assertEqual(
            result.steps[-1].questions, ('Concatenate lists ["d", "c"] and ["b", "a"].',)
        )
        agent = paper_agent(ScriptedProvider(responses))
        with self.assertRaisesRegex(RuntimeError, "call budget"):
            await agent.run("reverse", program="reverse", max_calls=10)
        self.assertEqual(agent.calls, 10)

    async def test_override_replaces_program_and_handler_errors_are_not_retried(self):
        agent = paper_agent(
            ScriptedProvider(["[str_position] word", "[EOQ]"]), handlers={"str_position": echo}
        )
        self.assertEqual((await agent.run("q")).answer, "word")
        self.assertEqual(agent.calls, 3)
        self.assertNotIn("open_qa", agent.programs)

        async def broken(question):
            raise LookupError("index is missing")

        agent = Decomp("task", ScriptedProvider(["[broken] q"]), {"broken": broken})
        with self.assertRaisesRegex(LookupError, "index is missing"):
            await agent.run("q")
        self.assertEqual(agent.calls, 2)

    async def test_budget_depth_and_failure_propagation(self):
        agent = Decomp("task", ScriptedProvider(["[echo] q"]), {"echo": echo})
        with self.assertRaisesRegex(RuntimeError, "call budget"):
            await agent.run("q", max_calls=1)
        self.assertEqual(agent.calls, 1)
        agent = Decomp("task", ScriptedProvider(["[decomp] q"]), {})
        with self.assertRaisesRegex(RuntimeError, "depth"):
            await agent.run("q", max_depth=0)
        provider = ScriptedProvider([OSError("transport")])
        with self.assertRaisesRegex(OSError, "transport"):
            await Decomp("task", provider, {}).run("q")
        self.assertEqual(len(provider.calls), 1)

    async def test_empty_foreach_and_wrong_shapes(self):
        for value in ([], "scalar", ["item"]):

            async def source(question):
                return value

            agent = Decomp(
                "task",
                ScriptedProvider(["[source] q", "[foreach_merge] [echo] #1", "[EOQ]"]),
                {"source": source, "echo": echo},
            )
            if value == []:
                self.assertEqual((await agent.run("q")).answer, [])
            else:
                with self.assertRaises(ValueError):
                    await agent.run("q")

    async def test_symbolic_retrieval_reaches_final_reader(self):
        documents = [{"title": "Example", "text": "The answer is blue."}]

        async def retrieve(question):
            return documents

        provider = ScriptedProvider(
            [
                "[retrieve_odqa] What color?",
                "[retrieve] What color?",
                "[singlehop_qa] Documents: #1 Question: What color?",
                json.dumps({"answer": {"documents": documents, "answer": "blue"}}),
                "[EOQ]",
                "[multihop_qa] Evidence: #1 Question: What color?",
                '{"answer": "blue"}',
                "[EOQ]",
            ]
        )
        agent = paper_agent(provider, retrieve=retrieve)
        self.assertEqual((await agent.run("What color?", program="open_qa")).answer, "blue")
        self.assertIn("The answer is blue.", provider.calls[-2])


if __name__ == "__main__":
    unittest.main()
