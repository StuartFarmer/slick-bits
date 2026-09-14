"""Small offline checks: python -m unittest qube.test_qube."""

import asyncio
import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from slick import prompts

from qube.problems import capset, evaluate, l2_bound, tour_length, two_opt, weibull
from qube.search import Cluster, Database, Program, Recent


def setUpModule():
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"


class QubeChecks(unittest.TestCase):
    def test_templates_from_another_directory(self):
        from qube.problems import SEEDS
        from qube.run import generate

        async def check():
            original_directory = Path.cwd()
            with tempfile.TemporaryDirectory() as directory:
                try:
                    os.chdir(directory)
                    for task in SEEDS:
                        rendered = await generate.render(task, SEEDS[task], SEEDS[task])
                        self.assertIn(SEEDS[task], rendered)
                        self.assertIn("<parent_v1>", rendered)
                finally:
                    os.chdir(original_directory)

        asyncio.run(check())

    def test_offspring_quality_selection_and_reset(self):
        # Own score must not substitute for observed offspring quality.
        a = Program("def priority(): return 0", (10.0,), 10.0)
        b = Program("def priority(): return 1", (2.0,), 2.0)
        db = Database(a, islands=4, k=2, reset_interval=4, seed=7)
        island = db.islands[0]
        island.add(b)
        ca, cb = island.clusters.values()
        self.assertEqual(ca.uiq(10, 2), math.inf)
        ca.visits, ca.offspring_sum, ca.offspring_count = 4, 12, 4
        cb.visits, cb.offspring_sum, cb.offspring_count = 4, 32, 4
        self.assertAlmostEqual(ca.uiq(10, 2), 3 + 2 * math.sqrt(math.log(10) / 4))
        self.assertGreater(cb.uiq(10, 2), ca.uiq(10, 2))
        # Equal aggregate scores with different per-instance behavior stay separate.
        island.add(Program("third", (1.0, 3.0), 2.0))
        self.assertEqual(len(island.clusters), 3)
        island.clusters.pop((1.0, 3.0))
        parents, clusters = island.select(random.Random(1), 10, 2, 1)
        self.assertEqual({p.code for p in parents}, {a.code, b.code})
        db.record(0, clusters, Program("child", (6.0,), 6.0))
        self.assertEqual((ca.offspring_count, cb.offspring_count), (5, 5))
        self.assertEqual(ca.offspring_sum, 18)
        db.record(0, clusters, None)
        self.assertEqual(ca.offspring_count, 5)
        # Rank reset islands by offspring quality, not own best score.
        for i, current in enumerate(db.islands):
            for cluster in current.clusters.values():
                cluster.visits = 10
                cluster.offspring_count = 1
                cluster.offspring_sum = i
        old = list(db.islands)
        resets = db.reset()
        self.assertEqual({r["island"] for r in resets}, {0, 1})
        self.assertIs(db.islands[2], old[2])
        self.assertIsNot(db.islands[0], old[0])
        self.assertTrue(all(c.visits == 0 for c in db.islands[0].clusters.values()))

    def test_single_cluster_and_metrics(self):
        seed = Program("def f(): return 1", (1.0,), 1.0)
        db = Database(seed, islands=1, k=0, reset_interval=0)
        index, parents, clusters = db.select(batch_size=4)
        self.assertEqual(len(parents), 2)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].visits, 4)
        db.record(index, clusters, seed)
        self.assertEqual(clusters[0].offspring_count, 1)
        self.assertEqual(db.reset(), [])
        recent = Recent(2)
        recent.add(seed, [seed])
        recent.add(None, [seed])
        self.assertEqual(
            recent.metrics(),
            {"recent_best_score": 1.0, "recent_proportion_of_change": 0.0},
        )
        recent.add(None, [seed])
        self.assertIsNone(recent.metrics()["recent_best_score"])
        c = Cluster([seed])
        self.assertEqual(c.uiq(1, 0), 1)

    def test_top_two_and_short_program_bias(self):
        from qube.search import Island, edit_distance

        island = Island()
        for score in [100, 2, 3]:
            island.add(Program(f"def f(): return {score}", (score,), score))
        clusters = list(island.clusters.values())
        for c, quality in zip(clusters, [0, 5, 10]):
            c.visits, c.offspring_count, c.offspring_sum = 1, 1, quality
        _, chosen = island.select(random.Random(0), 5, 0, 1)
        self.assertEqual(chosen, [clusters[2], clusters[1]])
        short = Program("x", (1,), 1)
        long = Program("x" * 100, (1,), 1)
        cluster = Cluster([short, long])
        self.assertTrue(all(cluster.sample(random.Random(i), 1) == short for i in range(20)))
        self.assertEqual(edit_distance(["a", "b"], ["b", "c"]), 2)

    def test_problem_invariants(self):
        self.assertEqual(l2_bound([6, 6, 6], 10), 3)
        self.assertEqual(l2_bound([6, 4, 6, 4], 10), 2)

        # Independent exhaustive optimum on small instances bounds L2 from above.
        def optimum(items, bins=()):
            if not items:
                return len(bins)
            item, *rest = items
            answers = [optimum(rest, bins + (item,))]
            for i, used in enumerate(bins):
                if used + item <= 10:
                    answers.append(optimum(rest, bins[:i] + (used + item,) + bins[i + 1 :]))
            return min(answers)

        rng = random.Random(9)
        for _ in range(30):
            items = [rng.randint(1, 10) for _ in range(6)]
            self.assertLessEqual(l2_bound(items, 10), optimum(items))
        data = [{"capacity": 10, "items": [6, 4, 6, 4], "lower_bound": 2}]
        result = evaluate("binpack", lambda item, bins: -(bins - item), data)
        self.assertEqual(result, {"signature": [2.0], "score": 0.0})
        with self.assertRaises(ValueError):
            evaluate("binpack", lambda item, bins: np.full(len(bins), np.nan), data)
        vectors = capset(3, lambda element, n: 0)
        for i, a in enumerate(vectors):
            for j in range(i + 1, len(vectors)):
                for b in vectors[j + 1 :]:
                    self.assertTrue(np.any((a + vectors[j] + b) % 3))
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        distances = np.linalg.norm(square[:, None] - square[None, :], axis=-1)
        route = two_opt([0, 2, 1, 3], distances)
        self.assertEqual(sorted(route), list(range(4)))
        self.assertAlmostEqual(tour_length(route, distances), 4)
        result = evaluate(
            "tsp",
            lambda d, r: np.zeros_like(d),
            [{"cities": square.tolist(), "optimum": 4}],
            iterations=2,
        )
        self.assertEqual(result["score"], 0)
        self.assertEqual(weibull(20, 2, 7), weibull(20, 2, 7))

    def test_parsing_and_invalid_inputs(self):
        from qube.problems import read_or_library, validate_data
        from qube.run import clean_code, evaluate_code, make_parser

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "or.txt"
            path.write_text("1\ntiny\n10 4 2\n6\n4\n6\n4\n")
            self.assertEqual(read_or_library(path)[0]["lower_bound"], 2)
            path.write_text("1 tiny 10 4 2 6")
            with self.assertRaises(ValueError):
                read_or_library(path)
        for code in ["print(1)", "def wrong(a, b): return 0", "def priority(: pass"]:
            with self.assertRaises((ValueError, SyntaxError)):
                clean_code(code, "binpack")
        with self.assertRaises(ValueError):
            validate_data("binpack", [{"capacity": 10, "items": [11]}])
        args = make_parser().parse_args([])
        with self.assertRaises(ValueError):
            asyncio.run(evaluate_code("def priority(a,b): return 7", args, []))

    def test_failed_generation_counts_towards_budget(self):
        from slick.providers import ProviderError

        from qube.run import evolve, make_parser

        with tempfile.TemporaryDirectory() as folder:
            args = make_parser().parse_args(
                [
                    "--samples",
                    "3",
                    "--items",
                    "5",
                    "--instances",
                    "1",
                    "--output",
                    folder,
                ]
            )
            # Only external model failure is replaced; the search loop and scoring are real.
            with patch("qube.run.DemoProvider.acall", side_effect=ProviderError("test failure")):
                result = asyncio.run(evolve(args))
            self.assertEqual((result["generated"], result["rejected"]), (3, 3))
            self.assertEqual(result["best_score"], result["initial_score"])

    def test_slick_offline_end_to_end(self):
        from qube.problems import SEEDS
        from qube.run import DemoProvider, evolve, generate, make_parser

        async def check():
            provider = DemoProvider("binpack")
            answer = await generate(
                "binpack", SEEDS["binpack"], SEEDS["binpack"], provider=provider
            )
            self.assertIn("def priority", answer)
            self.assertIn(SEEDS["binpack"], provider.context)
            with tempfile.TemporaryDirectory() as folder:
                args = make_parser().parse_args(
                    [
                        "--samples",
                        "5",
                        "--islands",
                        "2",
                        "--reset-interval",
                        "4",
                        "--samplers",
                        "2",
                        "--items",
                        "12",
                        "--instances",
                        "2",
                        "--output",
                        folder,
                    ]
                )
                summary = await evolve(args)
                self.assertEqual(summary["generated"], 5)
                self.assertEqual(summary["accepted"], 5)
                self.assertEqual(summary["resets"], 1)
                self.assertTrue((Path(folder) / "best.py").exists())
                self.assertEqual(len((Path(folder) / "history.jsonl").read_text().splitlines()), 5)

        asyncio.run(check())

    @unittest.skipUnless(os.environ.get("QUBE_TEST_DOCKER") == "1", "opt-in Docker integration")
    def test_docker_output_is_bounded_before_completion(self):
        from qube.run import evaluate_code, make_parser, validate_args

        async def check():
            args = make_parser().parse_args(
                ["--provider", "litellm", "--model", "unused", "--eval-timeout", "2"]
            )
            validate_args(args)
            data = [{"capacity": 10, "items": [6, 4], "lower_bound": 1}]
            for descriptor in (1, 2):
                code = (
                    "def priority(item, bins):\n    import os, time\n"
                    f'    for _ in range(17): os.write({descriptor}, b"x" * 65536)\n'
                    "    time.sleep(4)\n    return np.zeros_like(bins)\n"
                )
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    await evaluate_code(code, args, data)

        asyncio.run(check())

    @unittest.skipUnless(os.environ.get("QUBE_TEST_DOCKER") == "1", "opt-in Docker integration")
    def test_docker_isolation_and_timeout(self):
        from qube.problems import SEEDS
        from qube.run import evaluate_code, make_parser, validate_args

        async def check():
            args = make_parser().parse_args(["--provider", "litellm", "--model", "unused"])
            validate_args(args)
            data = [{"capacity": 10, "items": [6, 4], "lower_bound": 1}]
            program = await evaluate_code(SEEDS["binpack"], args, data)
            self.assertEqual(program.score, 0)
            # The default image is unprivileged, read-only, and has no external network.
            probes = [
                "open('/app/probe', 'w').write('no')",
                "__import__('socket').create_connection(('1.1.1.1', 80), timeout=1)",
                "np.full(len(bins), np.nan)",
            ]
            for expression in probes:
                with self.assertRaises(ValueError):
                    await evaluate_code(
                        f"def priority(item, bins):\n    return {expression}\n",
                        args,
                        data,
                    )
            args.eval_timeout = 1
            with self.assertRaises(TimeoutError):
                await evaluate_code("def priority(item, bins):\n    while True: pass\n", args, data)
            process = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "-aq",
                "--filter",
                "name=qube-",
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            self.assertEqual(stdout.strip(), b"")

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
