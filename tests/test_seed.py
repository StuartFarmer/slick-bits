import itertools
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from slick import tool

from seed import (
    SEED,
    CandidateRejected,
    Config,
    Example,
    Module,
    Prediction,
    SeedOptimizer,
    classification_confidence,
    generation_confidence,
)
from seed.agent import VerifiedCode
from seed.batching import form_batches
from seed.planning import execution_cost, priority_order
from tests.providers import ScriptedProvider


async def accuracy(examples, predictions):
    return sum(p is not None and p.value == e.output for e, p in zip(examples, predictions)) / len(
        examples
    )


class SeedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch("slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "seed/prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_optimizer_gap_cached_validation_and_order(self):
        calls = []

        def module(key, family, cost, answers):
            async def predict(inputs):
                calls.append(key)
                return [None if a is None else Prediction(a) for a in answers]

            return Module(key, family, cost, predict)

        llm = module("llm", "LLM", 1, [1, 1, 1, 1])
        code = module("code", "CodeGen", 0.01, [1, 1, 0, None])
        examples = [Example(str(i), 1) for i in range(4)]
        optimizer = SeedOptimizer(examples, accuracy)
        result = await optimizer.run([llm, code], gap=0.25, mode="exhaustive")
        self.assertEqual([m.key for m in result.best.modules], ["code", "llm"])
        self.assertAlmostEqual(result.best.cost, 0.26)
        self.assertEqual(result.best.effectiveness, 0.75)
        self.assertEqual(calls, ["llm", "code"])
        strict = await optimizer.run([llm, code], gap=0, mode="generic")
        self.assertEqual(strict.best.effectiveness, 1)
        self.assertEqual(calls, ["llm", "code"])

    def test_priority_order_minimizes_equation_one(self):
        stats = {"a": (0.3, 0.2), "b": (0.1, 0.9), "c": (1, 0)}
        modules = [Module(k, k, c, None) for k, (c, _) in stats.items()]
        fallback = {k: p for k, (_, p) in stats.items()}
        ordered = priority_order(modules, fallback)
        self.assertEqual(
            execution_cost(ordered, fallback),
            min(execution_cost(p, fallback) for p in itertools.permutations(modules)),
        )

    async def test_llm_batches_validate_ids_and_preserve_null_false(self):
        provider = ScriptedProvider(['{"answers":[{"id":1,"value":null},{"id":0,"value":false}]}'])
        agent = SEED("Any JSON task", provider, accuracy, config=Config(code_branches=0))
        outputs = await agent.label(["first", "second"])
        self.assertEqual([p.value for p in outputs], [False, None])
        self.assertEqual(agent.llm_records, 2)
        self.assertEqual(len(agent.cache), 2)
        self.assertEqual(len(agent.attempts), 1)
        self.assertIn("raw", agent.attempts[0])

    async def test_invalid_batch_is_not_cached(self):
        provider = ScriptedProvider(['{"answers":[{"id":0,"value":1},{"id":0,"value":2}]}'])
        agent = SEED("t", provider, accuracy)
        with self.assertRaisesRegex(ValueError, "IDs"):
            await agent.label(["a", "b"])
        self.assertFalse(agent.cache)
        self.assertIn("error", agent.attempts[0])

    async def test_runtime_reoptimizes_without_training_on_validation(self):
        provider = ScriptedProvider(
            [
                '{"answers":[{"id":0,"value":"gold"}]}',
                '{"answers":[{"id":0,"value":"learned"}]}',
            ]
        )
        fitted = []

        async def fit(examples):
            fitted.append(tuple(examples))

            async def predict(inputs):
                return [Prediction("gold" if x == "heldout" else "learned", 0.99) for x in inputs]

            return predict

        agent = SEED(
            "t",
            provider,
            accuracy,
            train_model=fit,
            config=Config(code_branches=0, min_train=1, reoptimize_every=1, batch_size=1),
        )
        await agent.run([Example("heldout", "gold")])
        outputs = await agent.predict(["new", "another"])
        self.assertEqual([p.value for p in outputs], ["learned", "learned"])
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(fitted, [(Example("new", "learned"),)])
        self.assertNotIn("heldout", agent.cache)
        self.assertEqual(len(agent.history), 2)

    async def test_code_evolution_uses_development_errors_and_preserves_static_results(self):
        original = "def solve(input):\n    return None\n"
        repaired = 'def solve(input):\n    return {"value": input}\n'
        provider = ScriptedProvider(
            [
                '{"strategies":["Parse the input."]}',
                json.dumps({"source": original}),
                '{"strategies":["Handle the second format."]}',
                json.dumps({"source": repaired}),
                '{"answers":[{"id":0,"value":"heldout"}]}',
            ]
        )
        executed = []

        async def execute(source, inputs):
            executed.append((source, tuple(inputs)))
            return [Prediction(x) if source == repaired or x == "first" else None for x in inputs]

        agent = SEED(
            "Echo task",
            provider,
            accuracy,
            execute_code=execute,
            config=Config(code_branches=1, evolution_iterations=1),
        )
        await agent.run(
            [Example("heldout", "heldout")],
            examples=[Example("first", "first"), Example("second", "second")],
        )
        self.assertEqual(agent.codes[0].source, repaired)
        self.assertEqual(agent.codegen_calls, 4)
        for call in provider.calls[:4]:
            self.assertNotIn("heldout", call)
        self.assertIn('"second"', provider.calls[2])
        self.assertEqual((await agent.predict(["fresh"]))[0].value, "fresh")
        before = len(executed), len(provider.calls)
        await agent.reoptimize()
        self.assertEqual((len(executed), len(provider.calls)), before)

    async def test_codegen_rejection_budget_and_raw_response(self):
        provider = ScriptedProvider(
            [
                '{"strategies":["one", "two"]}',
                '{"source":"not python at all"}',
                '{"source":"def wrong(): pass"}',
            ]
        )
        agent = SEED("t", provider, accuracy, config=Config(max_codegen_calls=3))
        await agent._initialize_code([Example("a", "a")])
        self.assertEqual(agent.codegen_calls, 3)
        self.assertIn("CandidateRejected", agent.attempts[-1]["error"])
        self.assertIn("def wrong", str(agent.attempts[-1]["raw"]))
        self.assertIsNone(await agent._code_call(agent.advise, [], 1))
        self.assertEqual(len(provider.calls), 3)

    async def test_cache_distance_threshold_and_snapshot(self):
        async def embed(inputs):
            vectors = {"a": [1, 0], "close": [0.99, 0.01], "far": [-1, 0]}
            return np.array([vectors[x] for x in inputs])

        agent = SEED("t", None, accuracy, embed=embed, config=Config(distance_thresholds=(0, 0.1)))
        agent.cache["a"] = Example("a", False)
        modules = await agent._adaptive_modules()
        strict, loose = modules
        self.assertEqual((await strict.predict(["a"]))[0].value, False)
        self.assertEqual(await strict.predict(["close"]), [None])
        self.assertEqual((await loose.predict(["close"]))[0].value, False)
        self.assertEqual(await loose.predict(["far"]), [None])
        agent.cache["far"] = Example("far", True)
        self.assertEqual(await loose.predict(["far"]), [None])

    async def test_tool_session_is_fresh_per_batch(self):
        @tool
        def lookup(key: str) -> str:
            """Look up a key in supplied task data."""
            return "found " + key

        request = {"id": "lookup-1", "name": "lookup", "arguments": {"key": "a"}}
        provider = ScriptedProvider(
            [
                ("", [request]),
                '{"answers":[{"id":0,"value":"found a"}]}',
                '{"answers":[{"id":0,"value":"b"}]}',
            ]
        )
        agent = SEED(
            "Lookup task",
            provider,
            accuracy,
            tools=[lookup],
            config=Config(batch_size=1, few_shot=0),
        )
        outputs = await agent.label(["a", "b"])
        self.assertEqual([p.value for p in outputs], ["found a", "b"])
        self.assertEqual(len(agent.attempts[0]["raw"]["history"]), 2)
        self.assertEqual(len(agent.attempts[1]["raw"]["history"]), 1)

    async def test_generated_examples_and_every_template_render(self):
        provider = ScriptedProvider(['{"examples":[{"input":"a","output":null}]}'])
        agent = SEED("Example task", provider, accuracy)
        self.assertEqual(await agent._code_call(agent.generate_examples, 1), [Example("a", None)])
        operations = [
            (SEED.query, (["a"], [])),
            (SEED.advise, ([], 2)),
            (SEED.generate, ("strategy", [])),
            (SEED.advise_repair, ("code", [], 2)),
            (SEED.repair, ("code", [], "strategy")),
            (SEED.generate_examples, (2,)),
        ]
        for operation, args in operations:
            rendered = await operation.render(agent, *args)
            self.assertIn("Example task", rendered)
            self.assertIn('"properties"', rendered)
        from jinja2 import Environment, nodes

        for path in (Path(__file__).parents[1] / "seed/prompts").glob("*.j2"):
            tree = Environment().parse(path.read_text())
            self.assertFalse(list(tree.find_all((nodes.If, nodes.CondExpr))))

    async def test_ensemble_ties_abstain_and_sequential_stops(self):
        calls = []

        async def execute(source, inputs):
            calls.append(source)
            return [Prediction(json.loads(source)) for _ in inputs]

        codes = [
            VerifiedCode("false", 0.8, frozenset(), ()),
            VerifiedCode("null", 0.9, frozenset(), ()),
        ]
        agent = SEED("t", None, accuracy, execute_code=execute, config=Config(ensemble="vote"))
        self.assertEqual(await agent._ensemble(codes, ["a"]), [None])
        agent.config = Config(ensemble="weighted")
        self.assertEqual(await agent._ensemble(codes, ["a"]), [Prediction(None)])
        agent.config = Config(ensemble="sequential")
        calls.clear()
        self.assertEqual(await agent._ensemble(codes, ["a"]), [Prediction(False)])
        self.assertEqual(calls, ["false"])

    def test_paper_confidence_formulas(self):
        self.assertEqual(classification_confidence([0.5, 0.5]), 0)
        self.assertAlmostEqual(classification_confidence([0.1, 0.9]), 0.8)
        self.assertAlmostEqual(classification_confidence([0.1, 0.1, 0.8]), 0.7)
        self.assertAlmostEqual(
            generation_confidence([math.log(0.5), math.log(0.8)]), math.sqrt(0.4)
        )

    async def test_executor_failure_policy(self):
        async def rejected(source, inputs):
            raise CandidateRejected("resource limit")

        agent = SEED("t", None, accuracy, execute_code=rejected)
        self.assertIsNone(await agent._verify("source", [Example("a", 1)]))
        self.assertIn("resource limit", agent.code_evaluations[-1]["error"])

        async def broken(source, inputs):
            raise RuntimeError("worker unavailable")

        agent.execute_code = broken
        with self.assertRaisesRegex(RuntimeError, "worker unavailable"):
            await agent._verify("source", [Example("a", 1)])

    def test_code_dominance_preserves_cautious_complement(self):
        cautious = VerifiedCode("cautious", 1.0, frozenset({0}), (Prediction(1), None, None))
        broad = VerifiedCode(
            "broad", 2 / 3, frozenset({0, 1}), (Prediction(1), Prediction(1), Prediction(0))
        )
        agent = SEED("t", None, accuracy)
        self.assertEqual(agent._filter_codes([cautious, broad]), [cautious, broad])

    async def test_balanced_fewshot_selection_uses_all_json_labels(self):
        agent = SEED("t", None, accuracy, config=Config(few_shot=3))
        examples = [Example("a", False), Example("b", False), Example("c", None)]
        demos = await agent._demonstrations(["new"], examples, "balanced")
        self.assertEqual([d["output"] for d in demos[:2]], [False, None])

    async def test_nonfinite_generated_json_is_rejected(self):
        agent = SEED("t", ScriptedProvider(['{"answers":[{"id":0,"value":NaN}]}']), accuracy)
        with self.assertRaises(ValueError):
            await agent.label(["x"])
        self.assertFalse(agent.cache)

    def test_all_batching_modes_cover_remainders_exactly_once(self):
        embeddings = np.array([[i, i % 2] for i in range(11)], dtype=float)
        for mode in ("RND", "CLS", "FAR", "PRX", "DIV"):
            batches = form_batches(11, 4, mode, embeddings, seed=3)
            self.assertEqual(sorted(i for b in batches for i in b), list(range(11)))
            self.assertTrue(all(0 < len(b) <= 4 for b in batches))
            self.assertEqual(batches, form_batches(11, 4, mode, embeddings, seed=3))
        self.assertEqual(form_batches(0, 4), [])

    def test_closest_and_farthest_select_opposite_second_examples(self):
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
        close = form_batches(4, 2, "CLS", embeddings, seed=0)
        far = form_batches(4, 2, "FAR", embeddings, seed=0)
        self.assertEqual(close[0], [3, 2])
        self.assertEqual(far[0], [3, 0])

    async def test_shuffled_batches_restore_original_order(self):
        provider = ScriptedProvider(
            [
                '{"answers":[{"id":0,"value":"c"},{"id":1,"value":"b"}]}',
                '{"answers":[{"id":0,"value":"a"},{"id":1,"value":"e"}]}',
                '{"answers":[{"id":0,"value":"d"}]}',
            ]
        )
        agent = SEED("Echo", provider, accuracy, config=Config(batch_size=2))
        outputs = await agent.label(list("abcde"))
        self.assertEqual([p.value for p in outputs], list("abcde"))
        self.assertEqual(agent.llm_records, 5)

    async def test_malformed_generation_and_bad_measurement(self):
        provider = ScriptedProvider(["not JSON"])
        agent = SEED("t", provider, accuracy)
        self.assertIsNone(await agent._code_call(agent.advise, [], 1))
        self.assertIn("not JSON", str(agent.attempts[0]["raw"]))

        async def predict(inputs):
            return [Prediction(1, float("nan")) for _ in inputs]

        optimizer = SeedOptimizer([Example("x", 1)], accuracy)
        with self.assertRaisesRegex(ValueError, "confidence"):
            await optimizer.run([Module("bad", "LLM", 1, predict)])

    async def test_initial_abstaining_code_gets_one_repair_iteration(self):
        original = "def solve(input): return None"
        repaired = "def solve(input): return {'value': input}"
        provider = ScriptedProvider(
            [
                '{"strategies":["parse"]}',
                json.dumps({"source": original}),
                '{"strategies":["handle input"]}',
                json.dumps({"source": repaired}),
            ]
        )

        async def execute(source, inputs):
            return [Prediction(x) if source == repaired else None for x in inputs]

        agent = SEED(
            "t",
            provider,
            accuracy,
            execute_code=execute,
            config=Config(code_branches=1, evolution_iterations=1),
        )
        await agent._compile_code([Example("a", "a")])
        self.assertEqual([c.source for c in agent.codes], [repaired])

    async def test_nonfinite_generated_example_consumes_rejected_attempt(self):
        provider = ScriptedProvider(['{"examples":[{"input":"a","output":NaN}]}'])
        agent = SEED("t", provider, accuracy)
        self.assertIsNone(await agent._code_call(agent.generate_examples, 1))
        self.assertEqual(agent.codegen_calls, 1)
        self.assertIn("CandidateRejected", agent.attempts[0]["error"])
        self.assertIn("NaN", str(agent.attempts[0]["raw"]))


if __name__ == "__main__":
    unittest.main()
