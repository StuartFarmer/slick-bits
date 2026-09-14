"""Exercise TAUCHI's task loop, retrieval, reflection, and rejection boundaries offline."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes

from tauchi_gpt import Document, Evaluation, LocalMemory, TauchiGPT
from tests.providers import ScriptedProvider


async def embed(texts):
    return [[1.0, float("blue" in text.lower())] for text in texts]


async def unfinished(goal, steps):
    return Evaluation(False, "Continue toward the goal")


def draft(text, citations=()):
    return json.dumps({"content": text, "citations": list(citations)})


class TauchiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "tauchi_gpt/prompts"
        root = patch("slick.prompts.TEMPLATE_ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)

    async def test_task_creation_reprioritization_and_result_retrieval(self):
        for goal in ("Plan a route", "Explain an argument"):
            provider = ScriptedProvider(
                [
                    '{"tasks": ["First", "Second"]}',
                    draft("blue first result"),
                    '{"tasks": ["Third", " first "]}',
                    '{"task_ids": [3, 2]}',
                    draft("third result"),
                    '{"tasks": []}',
                ]
            )
            agent = TauchiGPT(goal, provider, embed, unfinished)
            result = await agent.run(max_steps=2)
            self.assertEqual([step.task.name for step in result.steps], ["First", "Third"])
            self.assertEqual([task.name for task in result.pending], ["Second"])
            self.assertEqual(result.stop_reason, "budget")
            self.assertEqual(result.calls, 6)
            self.assertIn("blue first result", provider.calls[4])
            self.assertEqual(result.steps[1].context[0].kind, "result")
            self.assertTrue(all(goal in call for call in provider.calls))

    async def test_fixed_reflection_cycles_then_external_completion(self):
        for cycles in (0, 4, 5):
            responses = [draft("initial")]
            for index in range(cycles):
                responses.extend([f"critique {index}", draft(f"revision {index}")])

            async def completed(goal, steps):
                self.assertEqual(len(steps[-1].reflections), cycles)
                return Evaluation(True, "Verified")

            agent = TauchiGPT("Any goal", ScriptedProvider(responses), embed, completed)
            result = await agent.run(initial_tasks=["Do it"], reflection_cycles=cycles)
            self.assertEqual(result.stop_reason, "completed")
            self.assertEqual(result.calls, 1 + 2 * cycles)
            self.assertEqual(result.output, f"revision {cycles - 1}" if cycles else "initial")
            self.assertEqual(len(result.evaluations), 1)

    async def test_queue_exhaustion_zero_budget_and_run_isolation(self):
        provider = ScriptedProvider(
            [draft("secret previous result"), '{"tasks": []}', draft("new result"), '{"tasks": []}']
        )
        agent = TauchiGPT("Goal", provider, embed, unfinished)
        zero = await agent.run(max_steps=0)
        self.assertEqual(zero.calls, 0)
        self.assertEqual(zero.stop_reason, "budget")
        first = await agent.run(initial_tasks=["One"])
        second = await agent.run(initial_tasks=["Two"])
        self.assertEqual(first.stop_reason, "exhausted")
        self.assertEqual(second.steps[0].context, ())
        self.assertNotIn("secret previous result", provider.calls[2])
        self.assertEqual(len(first.steps), 1)

    async def test_invalid_generated_order_preserves_pending_and_raw_response(self):
        for order in ([2, 2], [2], [2, 999], [True, 3]):
            raw = json.dumps({"task_ids": order})
            agent = TauchiGPT(
                "Goal",
                ScriptedProvider(
                    [
                        draft("result"),
                        '{"tasks": []}',
                        raw,
                    ]
                ),
                embed,
                unfinished,
            )
            with self.assertRaises(ValueError):
                await agent.run(initial_tasks=["One", "Two", "Three"])
            self.assertEqual([task.id for task in agent.pending], [2, 3])
            self.assertEqual(agent.calls[-1]["response"], raw)
            self.assertIn("error", agent.calls[-1])

    async def test_citations_refer_to_retrieved_exact_passages(self):
        documents = [Document("notes.txt", "The blue route is short.")]
        citation = {"chunk_id": "document:0:0", "quote": "blue route"}
        provider = ScriptedProvider([draft("Use blue", [citation]), '{"tasks": []}'])
        result = await TauchiGPT("Route", provider, embed, unfinished).run(
            documents, initial_tasks=["Choose"]
        )
        self.assertEqual(result.steps[0].context[0].source, "notes.txt")
        for bad in (
            {"chunk_id": "missing", "quote": "blue route"},
            {"chunk_id": "document:0:0", "quote": "invented"},
        ):
            raw = draft("Use blue", [bad])
            agent = TauchiGPT("Route", ScriptedProvider([raw]), embed, unfinished)
            with self.assertRaisesRegex(ValueError, "citation"):
                await agent.run(documents, initial_tasks=["Choose"])
            self.assertEqual(agent.steps, [])
            self.assertEqual(agent.pending[0].name, "Choose")
            self.assertEqual(agent.calls[-1]["response"], raw)

    async def test_failures_propagate_without_retry(self):
        for response in (
            "not json",
            draft("   "),
            TimeoutError("offline"),
            (draft("output"), ["tool request"]),
        ):
            agent = TauchiGPT("Goal", ScriptedProvider([response]), embed, unfinished)
            with self.assertRaises((ValueError, TimeoutError)):
                await agent.run(initial_tasks=["Do it"])
            self.assertEqual(len(agent.calls), 1)
            self.assertIn("error", agent.calls[-1])

        async def broken(goal, steps):
            raise RuntimeError("evaluation failed")

        agent = TauchiGPT("Goal", ScriptedProvider([draft("output")]), embed, broken)
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            await agent.run(initial_tasks=["Do it"])
        self.assertEqual(len(agent.steps), 1)

    async def test_local_memory_chunk_offsets_cosine_and_atomic_embedding_failure(self):
        async def vectors(texts):
            return [[0.0, 1.0] if "blue" in text else [1.0, 0.0] for text in texts]

        memory = LocalMemory(vectors, chunk_size=8, overlap=2)
        await memory.add(Document("source", "red red blue blue"), "document:0", "document")
        hits = await memory.retrieve("blue", 2)
        self.assertEqual(hits[0].text, "d blue b")
        for chunk in memory.chunks:
            self.assertEqual(chunk.text, "red red blue blue"[chunk.start : chunk.end])
            self.assertEqual(len(chunk.digest), 64)
        count = len(memory.chunks)

        async def invalid(texts):
            return [[float("nan"), 0.0] for _ in texts]

        memory.embed = invalid
        with self.assertRaises(ValueError):
            await memory.add(Document("bad", "invalid"), "document:1", "document")
        self.assertEqual(len(memory.chunks), count)

    async def test_all_templates_render_from_agent_directory(self):
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root.parent)
        agent = TauchiGPT(
            "Goal",
            ScriptedProvider(
                [
                    '{"tasks": ["One", "Two", "Three"]}',
                    draft("output"),
                    "critique",
                    draft("revision"),
                    '{"tasks": []}',
                    '{"task_ids": [3, 2]}',
                ]
            ),
            embed,
            unfinished,
        )
        await agent.run(max_steps=1, reflection_cycles=1)
        self.assertEqual(len(agent.calls), 6)
        for template in self.root.glob("*.j2"):
            tree = Environment().parse(template.read_text())
            self.assertEqual(list(tree.find_all((nodes.If, nodes.CondExpr))), [])
        self.assertIn("Goal", await TauchiGPT.initialize.render(agent))


if __name__ == "__main__":
    unittest.main()
