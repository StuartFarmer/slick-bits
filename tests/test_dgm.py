"""Offline checks of DGM recursion, selection, and candidate admission."""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts

from dgm import DGM, SEED_AGENT, Config, Evaluation, ExecutionError
from tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1] / "dgm"
PROPOSAL = json.dumps(
    {
        "improvement_proposal": "Try independent drafts",
        "implementation_suggestion": "Add a second generation",
        "problem_description": "Improve the general workflow using feedback",
    }
)
DIP = SEED_AGENT + "\n# second generation\n"
BEST = SEED_AGENT + "\n# best\n"


class DGMTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", ROOT / "prompts")
        root.start()
        self.addCleanup(root.stop)

    async def test_parent_runs_itself_and_lower_score_is_a_stepping_stone(self):
        executed = []
        provider = ScriptedProvider([PROPOSAL, DIP, PROPOSAL, "draft", BEST])

        async def execute(source, instruction, generate):
            executed.append(source)
            if "second generation" in source:
                await generate("draft a strategy")
            return await generate(instruction)

        async def evaluate(source):
            return Evaluation({SEED_AGENT: 0.5, DIP: 0.2, BEST: 0.9}[source], True, "log")

        agent = DGM("Improve arbitrary writing tasks", provider, evaluate, execute=execute)
        with patch.object(
            agent.rng, "choices", side_effect=lambda population, **kw: [population[-1]]
        ):
            result = await agent.run(config=Config(iterations=2))
        self.assertEqual(executed, [SEED_AGENT, DIP])
        self.assertEqual([c.parent_id for c in result.archive], [None, 0, 1])
        self.assertEqual(result.best.source, BEST)
        self.assertEqual(result.evaluation_calls, 3)
        self.assertEqual(len(provider.calls), 5)
        self.assertIn("log", provider.calls[0])
        self.assertIn(DIP.strip(), provider.calls[2])
        self.assertTrue(all(a.error is None for a in result.attempts))

    async def test_selection_matches_official_weights_and_batch_snapshot(self):
        async def execute(source, instruction, generate):
            return await generate(instruction)

        async def evaluate(source):
            return Evaluation({SEED_AGENT: 0.5, DIP: 0.0, BEST: 1.0}[source], True)

        provider = ScriptedProvider([PROPOSAL, DIP, PROPOSAL, BEST, PROPOSAL, BEST])
        agent = DGM("Any task", provider, evaluate, execute=execute)
        with patch.object(agent.rng, "choices", wraps=agent.rng.choices) as choices:
            result = await agent.run(config=Config(iterations=3, batch_size=2))
        self.assertEqual([a.parent_id for a in result.attempts[:2]], [0, 0])
        weights = choices.call_args.kwargs["weights"]
        self.assertAlmostEqual(weights[0], 0.5 / 3)
        self.assertAlmostEqual(weights[1], 0.0066928509242848554)
        self.assertAlmostEqual(weights[2], 0.9933071490757153)
        self.assertEqual(choices.call_args.kwargs["k"], 1)

    async def test_ablations_choose_latest_or_execute_fixed_seed(self):
        async def evaluate(source):
            return Evaluation(0.5, True)

        for mode in ["no_self_improve", "no_open_ended"]:
            with self.subTest(mode=mode):
                executed = []

                async def execute(source, instruction, generate):
                    executed.append(source)
                    return await generate(instruction)

                agent = DGM(
                    "Any task",
                    ScriptedProvider([PROPOSAL, DIP, PROPOSAL, BEST]),
                    evaluate,
                    execute=execute,
                )
                with patch.object(agent.rng, "choices", side_effect=lambda p, **kw: [p[-1]]):
                    result = await agent.run(config=Config(iterations=2, mode=mode))
                self.assertEqual(
                    executed, [SEED_AGENT, SEED_AGENT if mode == "no_self_improve" else DIP]
                )
                self.assertEqual(result.attempts[-1].parent_id, 1)

    async def test_rejections_keep_raw_output_and_consume_attempts(self):
        async def execute(source, instruction, generate):
            return await generate(instruction)

        async def evaluate(source):
            return Evaluation(0.5, source != DIP)

        provider = ScriptedProvider(
            [
                "bad json",
                PROPOSAL,
                "def broken(:",
                PROPOSAL,
                DIP,
                PROPOSAL,
                SEED_AGENT,
                PROPOSAL,
                BEST,
            ]
        )
        agent = DGM("Any task", provider, evaluate, execute=execute)
        result = await agent.run(config=Config(iterations=5))
        self.assertEqual([c.source for c in result.archive], [SEED_AGENT, BEST])
        self.assertEqual(len(result.attempts), 5)
        self.assertEqual(result.evaluation_calls, 3)
        self.assertTrue(all(a.error for a in result.attempts[:4]))
        self.assertIn("bad json", [g.response for g in result.generations])
        self.assertIn("def broken(:", [g.response for g in result.generations])
        # The rejected child did not increase the root's child-count penalty.
        with patch.object(agent.rng, "choices", wraps=agent.rng.choices) as choices:
            agent.select_parents(1, "dgm")
        self.assertEqual(choices.call_args.kwargs["weights"], [0.25, 0.5])

    async def test_runtime_budget_execution_failure_and_nonfinite_measurement(self):
        async def evaluate(source):
            return Evaluation(0.5, True)

        async def execute(source, instruction, generate):
            await generate("first")
            return await generate("second")

        agent = DGM("Task", ScriptedProvider([PROPOSAL, "first"]), evaluate, execute=execute)
        result = await agent.run(config=Config(iterations=1, max_generations=1))
        self.assertIn("budget", result.attempts[0].error)
        self.assertEqual(result.evaluation_calls, 1)

        async def failed(source, instruction, generate):
            raise ExecutionError("worker timed out")

        agent = DGM("Task", ScriptedProvider([PROPOSAL]), evaluate, execute=failed)
        result = await agent.run(config=Config(iterations=1))
        self.assertIn("timed out", result.attempts[0].error)

        async def invalid_score(source):
            return Evaluation(math.nan, True)

        agent = DGM("Task", ScriptedProvider([]), invalid_score, execute=execute)
        with self.assertRaisesRegex(ValueError, "score"):
            await agent.run(config=Config(iterations=0))

        async def invalid_child_score(source):
            return Evaluation(0.5 if source == SEED_AGENT else math.inf, True)

        agent = DGM(
            "Task",
            ScriptedProvider([PROPOSAL, "draft", BEST]),
            invalid_child_score,
            execute=execute,
        )
        result = await agent.run(config=Config(iterations=1))
        self.assertIn("score", result.attempts[0].error)
        self.assertEqual(result.evaluation_calls, 2)
        self.assertEqual(len(result.archive), 1)

    async def test_infrastructure_errors_propagate_with_attempt_record(self):
        async def evaluate(source):
            return Evaluation(0.5, True)

        async def execute(source, instruction, generate):
            raise RuntimeError("worker unavailable")

        agent = DGM("Task", ScriptedProvider([PROPOSAL]), evaluate, execute=execute)
        with self.assertRaisesRegex(RuntimeError, "worker unavailable"):
            await agent.run(config=Config(iterations=1))
        self.assertEqual(len(agent.result.attempts), 1)

    async def test_completed_sibling_survives_a_later_infrastructure_failure(self):
        async def evaluate(source):
            return Evaluation(0.9 if source == BEST else 0.5, True)

        async def execute(source, instruction, generate):
            return await generate(instruction)

        agent = DGM(
            "Task",
            ScriptedProvider([PROPOSAL, BEST, RuntimeError("offline")]),
            evaluate,
            execute=execute,
        )
        with self.assertRaisesRegex(RuntimeError, "offline"):
            await agent.run(config=Config(iterations=2, batch_size=2))
        self.assertEqual(agent.result.best.source, BEST)
        self.assertTrue(agent.result.attempts[0].accepted)

    async def test_latest_agent_ablation_advances_each_attempt_and_survives_rejection(self):
        executed = []

        async def evaluate(source):
            return Evaluation(0.5, True)

        async def execute(source, instruction, generate):
            executed.append(source)
            return await generate(instruction)

        agent = DGM(
            "Task",
            ScriptedProvider([PROPOSAL, DIP, PROPOSAL, "", PROPOSAL, BEST]),
            evaluate,
            execute=execute,
        )
        result = await agent.run(config=Config(iterations=3, batch_size=3, mode="no_open_ended"))
        self.assertEqual(executed, [SEED_AGENT, DIP, DIP])
        self.assertEqual([a.parent_id for a in result.attempts], [0, 1, 1])

    async def test_templates_render_from_another_directory(self):
        agent = DGM("Any task", ScriptedProvider([]), None, execute=None)
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                from dgm.agent import Candidate, Proposal

                parent = Candidate(0, None, SEED_AGENT, Evaluation(0.5, True, "feedback"))
                proposal = Proposal.model_validate_json(PROPOSAL)
                rendered = [
                    await DGM.diagnose.render(agent, parent),
                    await DGM.modify.render(agent, parent, proposal),
                    await DGM.generate.render(agent, "Any task"),
                ]
        finally:
            os.chdir(previous)
        self.assertTrue(all("Any task" in text for text in rendered))
        self.assertIn('"required"', rendered[0])
        for path in (ROOT / "prompts").glob("*.j2"):
            tree = Environment().parse(path.read_text())
            self.assertFalse(list(tree.find_all((nodes.If, nodes.CondExpr))))


if __name__ == "__main__":
    unittest.main()
