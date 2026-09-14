"""Exercise generic genetic-programming decisions with the shared scripted provider."""

import asyncio
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
                self.assertEqual(evaluated, ["aa", "bbbb", "z"])
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
            self.assertEqual(result.calls, 3)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
