"""Offline checks for AdaEvolve's adaptive search and Slick boundaries."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts

from adaevolve import AdaEvolve, Candidate, Config, Evaluation, Island, Tactic, Tactics
from tests.providers import ScriptedProvider


async def score(content):
    return Evaluation(float(content), feedback="Measured by the caller")


def tactics(*ideas):
    return Tactics(
        ideas=[
            Tactic(
                idea=x,
                description=x,
                what_to_optimize="score",
                cautions="Keep constraints",
                approach_type=x,
            )
            for x in ideas
        ]
    )


class AdaEvolveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "adaevolve/prompts"
        self.root = patch.object(prompts, "TEMPLATE_ROOT", root)
        self.root.start()
        self.addCleanup(self.root.stop)

    async def test_generic_search_direction_budget_and_failures(self):
        for maximize, expected in ((True, 3), (False, -2)):
            provider = ScriptedProvider(
                ['{"content":"3"}', "bad JSON", '{"content":"nan"}', '{"content":"-2"}']
            )
            agent = AdaEvolve("Optimize any text", provider, score, maximize=maximize)
            best = await agent.run("1", iterations=4)
            self.assertEqual(best.score, expected)
            self.assertEqual((agent.mutation_calls, agent.meta_calls, agent.evaluations), (4, 0, 4))
            self.assertEqual(sum(i.visits for i in agent.islands), 4)
            self.assertEqual(len(agent.attempts), 4)
            self.assertEqual(agent.attempts[1]["raw"], "bad JSON")
            self.assertIn("finite", agent.attempts[2]["error"])
            self.assertTrue(all("Optimize any text" in c for c in provider.calls))

    async def test_paper_updates_global_normalization_and_decay(self):
        agent = AdaEvolve("Task", ScriptedProvider([]), score)
        await agent.run("100", iterations=0)
        rich = agent.islands[0]
        poor = Island([Candidate(99, "1", 1)])
        agent.islands.append(poor)
        agent.update_state(rich, Candidate(1, "110", 110))
        agent.update_state(poor, Candidate(2, "1.5", 1.5))
        self.assertAlmostEqual(rich.signal, 0.1 * (10 / (100 + 1e-8)) ** 2)
        self.assertGreater(rich.reward, poor.reward)
        self.assertAlmostEqual(poor.reward, 0.5 / (110 + 1e-8))
        previous_signal, previous_reward = rich.signal, rich.reward
        agent.update_state(rich, None)
        self.assertAlmostEqual(rich.signal, previous_signal * 0.9)
        self.assertAlmostEqual(rich.reward, previous_reward * 0.9)
        self.assertAlmostEqual(rich.decayed_visits, 1.9)
        self.assertEqual(rich.visits, 2)
        self.assertEqual(agent.best.score, 110)
        self.assertEqual(agent.select_island(), 1)  # Only the original second island is unvisited.

    async def test_migration_is_a_ring_snapshot_without_bandit_credit(self):
        agent = AdaEvolve("Task", ScriptedProvider([]), score)
        await agent.run("1", iterations=0)
        agent.islands.append(Island([agent.best]))
        agent.update_state(agent.islands[0], Candidate(1, "2", 2))
        before = [(i.reward, i.visits, i.decayed_visits) for i in agent.islands]
        agent.migrate()
        self.assertEqual([i.best.score for i in agent.islands], [2, 2, 1])
        self.assertEqual(before, [(i.reward, i.visits, i.decayed_visits) for i in agent.islands])
        self.assertGreater(agent.islands[1].signal, 0)

    async def test_spawning_tactic_rotation_outcomes_and_total_call_limit(self):
        provider = ScriptedProvider(['{"content":"1"}'] * 20)
        meta = ScriptedProvider([tactics("first", "second"), tactics("third")])
        config = Config(
            initial_islands=1,
            meta_warmup=1,
            spawn_cooldown=2,
            max_islands=2,
            tactic_uses=1,
            migration_interval=0,
        )
        agent = AdaEvolve(
            "Task",
            provider,
            score,
            meta_provider=meta,
            evaluator_context="External scoring rules",
            config=config,
        )
        await agent.run("1", iterations=20, max_calls=7)
        self.assertEqual(agent.mutation_calls + agent.meta_calls, 7)
        self.assertEqual(agent.evaluations, agent.mutation_calls + 1)
        self.assertEqual(len(agent.islands), 2)
        self.assertIn("first", provider.calls[1])
        self.assertIn("second", provider.calls[2])
        self.assertIn("External scoring rules", meta.calls[0])
        self.assertIn("Measured by the caller", meta.calls[0])
        self.assertIn("first", meta.calls[1])
        self.assertEqual([t["uses"] for t in agent.tactic_history[:2]], [1, 1])
        self.assertTrue(all(t["improvement"] == 0 for t in agent.tactic_history))
        self.assertTrue(any(e["operation"] == "spawn" for e in agent.events))

    async def test_templates_and_generated_contract(self):
        from jinja2 import Environment, nodes
        from pydantic import ValidationError

        agent = AdaEvolve("Task", ScriptedProvider([]), score)
        parent = Candidate(0, "  keep whitespace\n", 1)
        for operation, args in (
            (AdaEvolve.explore, (parent, [])),
            (AdaEvolve.exploit, (parent, [])),
            (AdaEvolve.explore_guided, (parent, [], tactics("idea").ideas[0])),
            (AdaEvolve.exploit_guided, (parent, [], tactics("idea").ideas[0])),
            (AdaEvolve.generate_tactics, (parent, [], [])),
        ):
            rendered = await operation.render(agent, *args)
            self.assertIn('"properties"', rendered)
            self.assertIn("Task", rendered)
        for template in prompts.TEMPLATE_ROOT.glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertFalse(list(parsed.find_all((nodes.If, nodes.CondExpr))))
        content = await agent.explore(
            parent, [], provider=ScriptedProvider(['{"content":" x\\n"}'])
        )
        self.assertEqual(content, " x\n")
        with self.assertRaises(ValidationError):
            await agent.generate_tactics(
                parent, [], [], provider=ScriptedProvider(['{"ideas":[]}'])
            )

    async def test_provider_errors_abort_and_invalid_evaluations_preserve_seed(self):
        from pydantic import ValidationError

        try:
            Tactics.model_validate({"ideas": []})
        except ValidationError as validation_error:
            provider_error = validation_error
        agent = AdaEvolve("Task", ScriptedProvider([provider_error, '{"content":"2"}']), score)
        with self.assertRaises(ValidationError) as raised:
            await agent.run("1", iterations=2)
        self.assertIs(raised.exception, provider_error)
        self.assertEqual(agent.mutation_calls, 1)

        error = ValueError("provider failed")
        agent = AdaEvolve("Task", ScriptedProvider([error]), score)
        with self.assertRaises(ValueError) as raised:
            await agent.run("1", iterations=2)
        self.assertIs(raised.exception, error)
        self.assertIn("provider failed", agent.attempts[0]["error"])

        async def reject(content):
            return Evaluation(1) if content == "seed" else Evaluation(error="invalid candidate")

        agent = AdaEvolve("Task", ScriptedProvider(['{"content":"bad"}']), reject)
        self.assertEqual((await agent.run("seed", iterations=1)).content, "seed")
        self.assertEqual(agent.attempts[0]["error"], "invalid candidate")
        self.assertTrue(math.isfinite(agent.best.score))

    async def test_ucb_and_sampling_use_current_rewards_and_top_quartile(self):
        agent = AdaEvolve("Task", ScriptedProvider([]), score)
        await agent.run("1", iterations=0)
        for island in agent.islands:
            island.visits = 10
            island.decayed_visits = 5
        agent.islands[0].reward = 1
        agent.islands[1].reward = 2
        self.assertEqual(agent.select_island(), 1)
        agent.islands[0].visits = 1
        self.assertEqual(agent.select_island(), 0)
        island = Island([Candidate(i, f"artifact {i}", i) for i in range(8)])
        with patch.object(agent.rng, "random", return_value=1):
            for _ in range(20):
                parent, inspirations, explore = agent.sample(island)
                self.assertFalse(explore)
                self.assertIn(parent.score, (6, 7))
                self.assertEqual(
                    [p.score for p in inspirations],
                    [i for i in range(7, -1, -1) if i != parent.score][:3],
                )
        with patch.object(agent.rng, "random", return_value=0):
            parents = {agent.sample(island)[0].id for _ in range(100)}
        self.assertEqual(parents, set(range(8)))

    async def test_negative_zero_and_large_improvement_signals(self):
        for initial, improved in ((0, 1), (-2, -1), (1, 4)):
            agent = AdaEvolve("Task", ScriptedProvider([]), score)
            await agent.run(str(initial), iterations=0)
            island = agent.islands[0]
            before = agent.intensity(island)
            agent.update_state(island, Candidate(1, str(improved), improved))
            expected = 0.1 * ((improved - initial) / (abs(initial) + 1e-8)) ** 2
            self.assertAlmostEqual(island.signal / expected, 1)
            self.assertLess(agent.intensity(island), before)
            intensity = agent.intensity(island)
            agent.update_state(island, None)
            self.assertGreater(agent.intensity(island), intensity)

    async def test_invalid_guidance_and_evaluation_timeouts_consume_budget(self):
        async def evaluate(content):
            if content == "slow":
                raise TimeoutError("caller timeout")
            return Evaluation(1)

        agent = AdaEvolve(
            "Task",
            ScriptedProvider(['{"content":"slow"}'] * 2),
            evaluate,
            meta_provider=ScriptedProvider(['{"ideas":[]}']),
            config=Config(meta_warmup=1, initial_islands=1),
        )
        best = await agent.run("seed", iterations=2, max_calls=3)
        self.assertEqual(best.content, "seed")
        self.assertEqual((agent.mutation_calls, agent.meta_calls, agent.evaluations), (2, 1, 3))
        self.assertEqual(agent.islands[0].visits, 2)
        self.assertIn("caller timeout", agent.attempts[0]["error"])
        self.assertEqual(agent.attempts[1]["raw"], '{"ideas":[]}')

    async def test_no_unusable_tactic_calls_and_exact_text_diversity_port(self):
        from adaevolve.agent import text_distance

        provider = ScriptedProvider(['{"content":"1"}'])
        meta = ScriptedProvider([])
        agent = AdaEvolve(
            "Task",
            provider,
            score,
            meta_provider=meta,
            config=Config(meta_warmup=1, initial_islands=1),
        )
        await agent.run("1", iterations=20, max_calls=1)
        self.assertEqual((len(provider.calls), len(meta.calls)), (1, 0))
        self.assertAlmostEqual(text_distance("hello world", "hello there"), 0.7 * 2 / 3)
        self.assertEqual(text_distance("", ""), 0)


if __name__ == "__main__":
    unittest.main()
