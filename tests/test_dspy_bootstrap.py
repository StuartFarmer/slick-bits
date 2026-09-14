import unittest

from dspy_bootstrap import DSPyBootstrap, Example, Prediction, Trace


class BootstrapTests(unittest.IsolatedAsyncioTestCase):
    def test_labeled_demos_are_sampled_separately_for_each_module(self):
        train = [Example({"x": i}, {"y": i}) for i in range(5)]
        agent = DSPyBootstrap(["first", "second"], None, None, None)

        program = agent.labeled(train, 2)

        # Official LabeledFewShot advances one seeded RNG across predictors.
        self.assertEqual([demo.inputs["x"] for demo in program["first"]], [3, 4])
        self.assertEqual([demo.inputs["x"] for demo in program["second"]], [0, 2])
        self.assertEqual(agent.labeled(train, 2), program)

    async def test_zero_bootstrap_budget_keeps_random_candidates_without_teacher_calls(self):
        train = [Example({"x": i}, {"y": i}) for i in range(5)]

        async def teacher(program, inputs, rollout):
            self.fail("a zero bootstrap budget must not call the teacher")

        async def student(program, inputs, rollout):
            return Prediction({"y": inputs["x"]})

        result = await DSPyBootstrap(
            ["first", "second"],
            teacher,
            student,
            lambda ex, pred: float(ex.outputs == pred.outputs),
        ).run(train, [train[0]], candidates=2, max_bootstrapped=0, max_labeled=2)

        self.assertEqual(
            [candidate["seed"] for candidate in result["candidates"]], [-3, -2, -1, 0, 1]
        )
        self.assertEqual(result["teacher_calls"], 0)
        self.assertEqual(result["validation_calls"], 5)
        for candidate in result["candidates"][1:]:
            self.assertTrue(all(len(demos) == 2 for demos in candidate["program"].values()))
            self.assertTrue(
                all(demo in train for demos in candidate["program"].values() for demo in demos)
            )

    async def test_successful_traces_module_routing_and_no_self_label(self):
        train = [Example({"x": i}, {"y": i}) for i in range(3)]
        validation = [Example({"x": 99}, {"y": 99})]
        seen = []

        async def teacher(program, inputs, rollout):
            seen.append(inputs["x"])
            self.assertTrue(all(d.inputs != inputs for demos in program.values() for d in demos))
            return Prediction(
                {"y": inputs["x"]}, [Trace("first", Example(inputs, {"mid": "trace"}))]
            )

        async def student(program, inputs, rollout):
            self.assertEqual(inputs["x"], 99)
            return Prediction(
                {"y": 99 if any("mid" in d.outputs for d in program["first"]) else -1}
            )

        result = await DSPyBootstrap(
            ["first", "second"],
            teacher,
            student,
            lambda ex, pred: float(ex.outputs == pred.outputs),
        ).run(train, validation, candidates=0, max_bootstrapped=1, max_labeled=2)
        self.assertEqual(result["best"]["seed"], -1)
        self.assertEqual(result["teacher_calls"], 1)
        self.assertEqual(result["validation_calls"], 3)
        program = result["best"]["program"]
        self.assertIn("mid", program["first"][0].outputs)
        self.assertTrue(all("mid" not in d.outputs for d in program["second"]))
        self.assertNotIn(99, seen)
