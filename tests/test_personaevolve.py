import json
import unittest
from pathlib import Path
from unittest.mock import patch

from personaevolve import PersonaEvolve
from tests.providers import ScriptedProvider


class PersonaTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_surplus_personas_change_and_identity_is_preserved(self):
        initial = {str(i): {"identity": str(i), "trait": "A"} for i in range(4)}

        async def simulate(personas):
            return {name: persona["trait"] for name, persona in personas.items()}

        provider = ScriptedProvider(
            [json.dumps({"fields": {"trait": "B"}})] * 2
            + [json.dumps({"fields": {"identity": "changed", "trait": "B"}})]
        )
        with patch(
            "slick.prompts.TEMPLATE_ROOT", Path(__file__).parents[1] / "personaevolve/prompts"
        ):
            agent = PersonaEvolve(
                "task", provider, simulate, {"A": 0.5, "B": 0.5}, editable=["trait"]
            )
            result = await agent.run(initial, rounds=3)
            self.assertEqual(result["history"][-1]["gaps"], {"A": 0.0, "B": 0.0})
            self.assertEqual(result["simulation_calls"], 2)
            self.assertTrue(all(persona["trait"] == "A" for persona in initial.values()))
            self.assertTrue(
                all(name == persona["identity"] for name, persona in result["personas"].items())
            )
            with self.assertRaisesRegex(ValueError, "editable fields"):
                await agent.rewrite(initial["0"], "A", "B", "", provider=provider)
