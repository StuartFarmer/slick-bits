"""Offline checks of LLM-PSRO's numerical solver and generation/evaluation boundary."""

import ast
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts
from slick.providers import ProviderError

from llm_psro import LLMPSRO, CandidateRejected, Mixture, Proposal, fictitious_play
from tests.providers import ScriptedProvider


class LLMPSROTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "llm_psro/prompts"
        self.patch = patch.object(prompts, "TEMPLATE_ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_fictitious_play_cyclic_and_dominant_games(self):
        for matrix in (
            ((0, -1, 1), (1, 0, -1), (-1, 1, 0)),
            ((0, -1, -1), (1, 0, -1), (1, 1, 0)),
            ((0,),),
        ):
            weights = fictitious_play(matrix, iterations=20000, seed=7)
            self.assertAlmostEqual(sum(weights), 1)
            self.assertTrue(all(weight >= 0 for weight in weights))
            exploitability = max(sum(a * x for a, x in zip(row, weights)) for row in matrix)
            self.assertLess(exploitability, 0.025)
            self.assertEqual(weights, fictitious_play(matrix, iterations=20000, seed=7))

    async def test_rounds_payoffs_full_source_prompts_and_exact_budgets(self):
        population = ["rock", "paper"]
        provider = ScriptedProvider([Proposal(content="scissors"), Proposal(content="rock")] * 2)
        played = []

        async def play(left, right, seed):
            played.append((left, right, seed))
            beats = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
            return 0.0 if left == right else (1.0 if beats[left] == right else -1.0)

        agent = LLMPSRO("Play cyclic competition", provider, play)
        result = await agent.run(
            population, rounds=2, candidates_per_round=2, games_per_pair=12, fp_iterations=1000
        )
        self.assertEqual(population, ["rock", "paper"])
        self.assertEqual(len(result), 4)
        self.assertEqual(result[2], "scissors")
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(agent.games_played, (2**2 + 3**2 + 2 * 2) * 12)
        self.assertEqual(agent.games_played, len(played))
        for index, record in enumerate(agent.history):
            self.assertEqual(record.population, result[: 2 + index])
            self.assertEqual(len(record.mixture.weights), 2 + index)
            for i, row in enumerate(record.payoffs):
                self.assertEqual(row[i], 0)
                for j, value in enumerate(row):
                    self.assertEqual(value, -record.payoffs[j][i])
            for call in provider.calls[2 * index : 2 * index + 2]:
                self.assertIn(record.mixture.source, call)
                for source in record.population:
                    self.assertIn(source, call)
        self.assertEqual(agent.attempts[0]["win_rate"], 1)

    async def test_win_rate_selection_differs_from_average_score_and_balances_seats(self):
        provider = ScriptedProvider([Proposal(content="safe"), Proposal(content="risky")])
        calls = []
        results = {"safe": iter([1, 0, 0, 0]), "risky": iter([1, 1, -1, -1])}

        async def play(left, right, seed):
            calls.append((left, right, seed))
            if left == right:
                return 1.0  # A first-seat advantage must vanish in the symmetric metagame.
            candidate = right if left == "initial" else left
            outcome = next(results[candidate])
            return -outcome if left == "initial" else outcome

        agent = LLMPSRO("Any competition", provider, play)
        result = await agent.run(["initial"], rounds=1, candidates_per_round=2, games_per_pair=4)
        self.assertEqual(result, ("initial", "risky"))
        self.assertEqual(agent.history[0].payoffs, ((0.0,),))
        self.assertEqual([row["win_rate"] for row in agent.attempts], [0.25, 0.5])
        self.assertEqual([row["mean_score"] for row in agent.attempts], [0.25, 0])
        self.assertEqual([row[2] for row in calls[4:8]], [row[2] for row in calls[8:12]])
        self.assertEqual([row[0] for row in calls[4:8]], ["safe", "initial"] * 2)

    async def test_rejections_keep_raw_output_and_consume_fixed_attempts(self):
        provider = ScriptedProvider(
            ["not json", '{"content":"   "}', Proposal(content="broken"), Proposal(content="ok")]
        )

        async def play(left, right, seed):
            if "broken" in (left, right):
                raise CandidateRejected("invalid interface")
            return 0.0

        agent = LLMPSRO("Task", provider, play)
        result = await agent.run(["initial"], rounds=1, candidates_per_round=4, games_per_pair=2)
        self.assertEqual(result, ("initial", "ok"))
        self.assertEqual(len(agent.attempts), 4)
        self.assertTrue(all(row.get("error") for row in agent.attempts[:3]))
        self.assertEqual(agent.attempts[0]["response"], "not json")
        self.assertEqual(agent.attempts[1]["response"], '{"content":"   "}')
        self.assertEqual(agent.games_played, 5)  # Includes the failed match attempt.

    async def test_seat_bias_cancels_and_equal_win_rates_keep_first(self):
        async def play(left, right, seed):
            return 1.0

        provider = ScriptedProvider([Proposal(content="c"), Proposal(content="d")])
        agent = LLMPSRO("First player always wins", provider, play)
        result = await agent.run(["a", "b"], rounds=1, candidates_per_round=2, games_per_pair=4)
        self.assertEqual(result, ("a", "b", "c"))
        self.assertEqual(agent.history[0].payoffs, ((0.0, 0.0), (0.0, 0.0)))
        self.assertEqual([row["win_rate"] for row in agent.attempts], [0.5, 0.5])

    async def test_initialization_and_failures_are_bounded(self):
        async def play(left, right, seed):
            return float("nan")

        provider = ScriptedProvider([ProviderError("offline"), "bad", Proposal(content="seed")])
        agent = LLMPSRO("Task", provider, play)
        result = await agent.run(population_size=1, rounds=0)
        self.assertEqual(result, ("seed",))
        self.assertEqual(len(agent.attempts), 3)
        self.assertIn("offline", agent.attempts[0]["error"])
        agent = LLMPSRO("Task", ScriptedProvider(["bad"] * 3), play)
        with self.assertRaisesRegex(RuntimeError, "initial"):
            await agent.run(population_size=1)
        self.assertEqual(len(agent.attempts), 3)
        with self.assertRaisesRegex(ValueError, "outcome"):
            await agent.run(["initial"], rounds=1, games_per_pair=1)

        async def draw(left, right, seed):
            return 0.0

        agent = LLMPSRO("Task", ScriptedProvider(["bad"] * 2), draw)
        with self.assertRaisesRegex(RuntimeError, "response"):
            await agent.run(["initial"], rounds=1, games_per_pair=1, candidates_per_round=2)
        self.assertEqual(len(agent.attempts), 2)
        self.assertEqual(agent.population, ["initial"])

    async def test_unexpected_evaluator_errors_propagate(self):
        async def play(left, right, seed):
            if left != right:
                raise RuntimeError("simulator bug")
            return 0.0

        agent = LLMPSRO("Task", ScriptedProvider([Proposal(content="new")]), play)
        with self.assertRaisesRegex(RuntimeError, "simulator bug"):
            await agent.run(["initial"], rounds=1, games_per_pair=1, candidates_per_round=1)
        self.assertIn("simulator bug", agent.attempts[0]["error"])

    async def test_templates_and_mixture_are_problem_agnostic(self):
        async def play(left, right, seed):
            return 0.0

        source = "// arbitrary language\nchoose_action();\n"
        mixture = Mixture((source, "another_strategy()"), (1.0, 0.0))
        self.assertEqual(mixture.sample(random.Random(2)), source)
        assignments = ast.parse(mixture.source).body
        self.assertEqual(ast.literal_eval(assignments[0].value), mixture.population)
        self.assertEqual(ast.literal_eval(assignments[1].value), mixture.weights)
        for task in ("Design auction bidders", "Compete at routing packets"):
            agent = LLMPSRO(task, ScriptedProvider([]), play)
            for method, args in ((LLMPSRO.initialize, ()), (LLMPSRO.respond, (mixture,))):
                rendered = await method.render(agent, *args)
                self.assertIn(task, rendered)
                self.assertIn('"properties"', rendered)
                self.assertNotIn("Checkers", rendered)
        for template in self.root.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
