import json
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_evolution import LLMEvolution
from tests.providers import ScriptedProvider


class LLMEvolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_selection_and_operator_chain_with_elitist_survival(self):
        seen = []

        async def evaluate(text):
            seen.append(text)
            return {"a": 2, "b": 1, "child": 3, "worse": 0}[text]

        responses = [
            {"indices": [0, 1]},
            {"operator": "cross", "solution": "intermediate"},
            {"operator": "mutate", "solution": "child"},
            {"indices": [0, 1]},
            {"operator": "cross", "solution": "intermediate2"},
            {"operator": "mutate", "solution": "worse"},
            {"indices": [0, 0]},
        ]
        provider = ScriptedProvider([json.dumps(x) for x in responses])
        with patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "llm_evolution/prompts"
        ):
            agent = LLMEvolution(
                "task", provider, evaluate, crossover_knowledge="cross", mutation_knowledge="mutate"
            )
            result = await agent.run(["a", "b"], rounds=2, population_size=2, candidates=1)
            self.assertEqual([p["solution"] for p in result["population"]], ["child", "a"])
            self.assertIn("intermediate", provider.calls[2])
            self.assertNotIn("intermediate", seen)
            with self.assertRaisesRegex(ValueError, "distinct"):
                await agent.select(result["population"], provider=provider)
