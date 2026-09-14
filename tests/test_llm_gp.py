"""Exercise generic genetic-programming decisions with the shared scripted provider."""

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import Session, prompts

import llm_gp
from tests.providers import ScriptedProvider


class LLMGPChecks(unittest.TestCase):
    def setUp(self):
        self.root = patch.object(
            prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[1] / "llm_gp/prompts"
        )
        self.root.start()
        self.addCleanup(self.root.stop)

    def test_variation_keeps_elite_and_uses_injected_task_and_evaluation(self):
        self.assertTrue(hasattr(llm_gp, "LLMGP"), "export the generic LLMGP agent")

        async def check():
            for task in ("Rewrite customer support replies.", "Arrange workshop activities."):
                provider = ScriptedProvider(
                    ['{"content":"aa"}', '{"content":"bbbb"}', '{"contents":["z","ccc"]}']
                )
                evaluated = []

                async def evaluate(content):
                    evaluated.append(content)
                    return float(len(content))

                agent = llm_gp.LLMGP(
                    task,
                    provider,
                    evaluate,
                    population_size=2,
                    generations=2,
                    crossover_rate=1,
                    mutation_rate=0,
                )
                result = await agent.run()
                self.assertEqual(result.best.content, "z")
                self.assertEqual(result.evaluations, 4)
                self.assertEqual(evaluated, ["aa", "bbbb", "z", "ccc"])
                self.assertEqual(result.calls, 3)
                self.assertTrue(all(task in call for call in provider.calls))
                self.assertEqual(result.stop_reason, "generations")

        asyncio.run(check())

    def test_full_flow_selects_replaces_and_designates(self):
        self.assertTrue(hasattr(llm_gp, "LLMGP"), "export the generic LLMGP agent")

        async def check():
            provider = ScriptedProvider(
                [
                    '{"content":"aa"}',
                    '{"content":"bbbb"}',
                    '{"ids":[1,1]}',
                    '{"contents":["z","ccc"]}',
                    '{"ids":[2,0]}',
                    '{"ids":[0]}',
                ]
            )

            async def evaluate(content):
                return float(len(content))

            agent = llm_gp.LLMGP(
                "Improve an onboarding plan.",
                provider,
                evaluate,
                variant="llm-gp",
                population_size=2,
                generations=2,
                crossover_rate=1,
                mutation_rate=0,
            )
            result = await agent.run(session=Session(provider=provider))
            self.assertEqual([p.content for p in result.population], ["z", "aa"])
            self.assertEqual(result.designated_best.content, "z")
            self.assertEqual(result.calls, 6)
            self.assertEqual(result.evaluations, 4)
            self.assertIn("Repeated IDs are allowed", provider.calls[2])
            self.assertIn("next population", provider.calls[4])
            self.assertIn("final designated result", provider.calls[5])

        asyncio.run(check())

    def test_budget_and_invalid_candidates_are_explicit(self):
        self.assertTrue(hasattr(llm_gp, "LLMGP"), "export the generic LLMGP agent")

        async def check():
            provider = ScriptedProvider(['{"content":"invalid"}', '{"content":"ok"}'])

            async def evaluate(content):
                return float("nan") if content == "invalid" else 1.0

            agent = llm_gp.LLMGP(
                "Rank release notes.",
                provider,
                evaluate,
                population_size=2,
                max_calls=2,
            )
            result = await agent.run()
            self.assertEqual(result.stop_reason, "budget")
            self.assertEqual(result.best.content, "ok")
            self.assertEqual(result.evaluations, 2)
            self.assertTrue(result.errors)

        asyncio.run(check())

    def test_invalid_evaluator_fails_when_used(self):
        async def check():
            provider = ScriptedProvider(['{"content":"a"}'])
            agent = llm_gp.LLMGP("Rank notes.", provider, None)
            with self.assertRaises(TypeError):
                await agent.run()
            self.assertEqual(len(provider.calls), 1)

        asyncio.run(check())

    def test_provider_failures_and_cancellation_propagate(self):
        async def check():
            async def evaluate(content):
                raise asyncio.CancelledError

            provider = ScriptedProvider([ValueError("provider configuration failed")])
            with self.assertRaisesRegex(ValueError, "provider configuration failed"):
                await llm_gp.LLMGP("Rank notes.", provider, evaluate, max_calls=1).run()
            with self.assertRaises(asyncio.CancelledError):
                await llm_gp.LLMGP(
                    "Rank notes.", ScriptedProvider(['{"content":"a"}']), evaluate
                ).run()

        asyncio.run(check())

    def test_invalid_selection_uses_existing_candidates_and_score_order(self):
        async def check():
            provider = ScriptedProvider(
                [
                    '{"content":"aa"}',
                    '{"content":"bbbb"}',
                    '{"ids":[99,99]}',
                    '{"contents":["z","ccc"]}',
                    '{"ids":[0,0]}',
                    '{"ids":[99]}',
                ]
            )

            async def evaluate(content):
                return float(len(content))

            result = await llm_gp.LLMGP(
                "Rank notes.",
                provider,
                evaluate,
                variant="llm-gp",
                population_size=2,
                generations=2,
                crossover_rate=1,
                mutation_rate=0,
            ).run()
            self.assertEqual([p.content for p in result.population], ["z", "aa"])
            self.assertEqual(result.designated_best.content, "z")
            self.assertEqual(len(result.errors), 3)
            self.assertEqual(result.calls, 6)

        asyncio.run(check())

    def test_mutation_without_crossover(self):
        async def check():
            provider = ScriptedProvider(
                [
                    '{"content":"aa"}',
                    '{"content":"bbbb"}',
                    '{"content":"z"}',
                    '{"content":"ccc"}',
                ]
            )

            async def evaluate(content):
                return float(len(content))

            result = await llm_gp.LLMGP(
                "Rank notes.",
                provider,
                evaluate,
                population_size=2,
                generations=2,
                crossover_rate=0,
                mutation_rate=1,
            ).run()
            self.assertEqual(result.best.content, "z")
            self.assertEqual(result.calls, 4)

        asyncio.run(check())

    def test_full_offspring_population_competes_with_elite(self):
        async def check():
            provider = ScriptedProvider(
                ['{"content":"old"}', '{"content":"worse"}', '{"contents":["aa","z"]}']
            )

            async def evaluate(content):
                return len(content)

            result = await llm_gp.LLMGP(
                "Minimize length.",
                provider,
                evaluate,
                population_size=2,
                generations=2,
                crossover_rate=1,
                mutation_rate=0,
            ).run()
            self.assertEqual([p.content for p in result.population], ["z", "aa"])
            self.assertEqual(result.evaluations, 4)
            self.assertEqual([p.content for p in result.history[0]], ["old", "worse"])

        asyncio.run(check())

    def test_tournaments_sample_distinct_competitors(self):
        async def check():
            async def evaluate(content):
                return len(content)

            agent = llm_gp.LLMGP("Minimize length.", None, evaluate, population_size=2)
            agent.population = [
                llm_gp.Individual(content="a", score=1),
                llm_gp.Individual(content="bbbb", score=4),
            ]
            for _ in range(20):
                parents = await agent._choose_parents(None)
                self.assertEqual([p.content for p in parents], ["a", "a"])

        asyncio.run(check())

    def test_code_whitespace_is_preserved_and_blank_output_rejected(self):
        async def check():
            code = "    return value\n\n"
            provider = ScriptedProvider(
                [json.dumps({"content": " \n"}), json.dumps({"content": code})]
            )
            evaluated = []

            async def evaluate(content):
                evaluated.append(content)
                return 1

            result = await llm_gp.LLMGP(
                "Generate an indented function body.",
                provider,
                evaluate,
                population_size=1,
                generations=1,
            ).run()
            self.assertEqual(result.best.content, code)
            self.assertEqual(evaluated, [code])
            self.assertEqual(result.calls, 2)
            self.assertEqual(len(result.errors), 1)

        asyncio.run(check())

    def test_malformed_variation_keeps_parents_and_counts_failures(self):
        async def check():
            provider = ScriptedProvider(
                [
                    "not JSON",
                    '{"content":"a"}',
                    '{"content":"bb"}',
                    '{"contents":["only one child"]}',
                    '{"content":" "}',
                    "{}",
                ]
            )

            async def evaluate(content):
                return len(content)

            result = await llm_gp.LLMGP(
                "Minimize length.",
                provider,
                evaluate,
                population_size=2,
                generations=2,
                crossover_rate=1,
                mutation_rate=1,
            ).run()
            self.assertEqual([p.content for p in result.population], ["a", "a"])
            self.assertEqual(result.calls, 6)
            self.assertEqual(result.evaluations, 4)
            self.assertEqual(len(result.errors), 4)

        asyncio.run(check())

    def test_odd_population_maximization_and_partial_budget(self):
        async def check():
            async def evaluate(content):
                return len(content)

            responses = [
                '{"content":"a"}',
                '{"content":"bb"}',
                '{"content":"ccc"}',
                '{"contents":["dddd","eeeee"]}',
                '{"contents":["ffffff","discard"]}',
            ]
            for budget, expected_size, expected_best, stop in (
                (0, 0, None, "budget"),
                (4, 2, "eeeee", "budget"),
                (5, 3, "ffffff", "generations"),
            ):
                result = await llm_gp.LLMGP(
                    "Maximize length.",
                    ScriptedProvider(responses),
                    evaluate,
                    population_size=3,
                    generations=2,
                    crossover_rate=1,
                    mutation_rate=0,
                    maximize=True,
                    max_calls=budget,
                ).run()
                self.assertEqual(len(result.population), expected_size)
                self.assertEqual(result.best.content if result.best else None, expected_best)
                self.assertEqual(result.calls, budget)
                self.assertEqual(result.stop_reason, stop)
                self.assertEqual(result.evaluations, 0 if not budget else 3 + expected_size)

        asyncio.run(check())

    def test_all_templates_render_with_explicit_owner_binding(self):
        async def check():
            agent = llm_gp.LLMGP("Evolve a sorting function.", None, None, maximize=True)
            population = [llm_gp.Individual(content="candidate", score=1)]
            operations = [
                (llm_gp.LLMGP.initialize, ()),
                (llm_gp.LLMGP.crossover, (["a", "b"], ["sample"])),
                (llm_gp.LLMGP.mutate, ("a", ["sample"])),
                (llm_gp.LLMGP.select_parents, (population,)),
                (llm_gp.LLMGP.replace_population, (population,)),
                (llm_gp.LLMGP.designate_best, (population,)),
            ]
            for operation, args in operations:
                rendered = await operation.render(agent, *args)
                self.assertIn(agent.task, rendered)
                self.assertIn('"properties"', rendered)
                self.assertIn("Return JSON", rendered)
            self.assertIn("Higher scores are better", rendered)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
