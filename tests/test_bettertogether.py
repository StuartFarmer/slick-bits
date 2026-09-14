import unittest

from bettertogether import BetterTogether


class BetterTogetherTests(unittest.IsolatedAsyncioTestCase):
    async def test_sequence_continues_from_latest_and_keeps_best_snapshot(self):
        observed = []

        async def prompt(program, examples):
            observed.append(program["version"])
            program["version"] += 1
            return program

        async def weights(program, examples):
            program["version"] += 1
            return program

        async def evaluate(program):
            return {0: 0, 1: 5, 2: 1, 3: 2}[program["version"]]

        initial = {"version": 0}
        result = await BetterTogether("Task", prompt, weights, evaluate).run(initial, [1, 2])
        self.assertEqual(observed, [0, 2])
        self.assertEqual(initial, {"version": 0})
        self.assertEqual(result["best"]["program"], {"version": 1})
        self.assertEqual(result["evaluations"], 4)
