# Self-Taught Optimizer (STOP)

`STOP(task, provider, execute, problems).run(initial=SEED_IMPROVER, rounds=3)` recursively evolves optimizer source. A candidate improver is scored by running it on downstream initial solutions and averaging their measured utility. Each successful outer iteration uses the newly evolved source as the optimizer of itself on the next iteration. This is not a fixed prompt-refinement loop.

Sources: [paper](https://arxiv.org/abs/2310.02304), official [outer loop](https://github.com/microsoft/stop/blob/main/run_improver.py), [seed improver](https://github.com/microsoft/stop/blob/main/tasks/meta_optimization/secret_seed_algorithm.py), and [meta utility](https://github.com/microsoft/stop/blob/main/tasks/meta_optimization/secret_utility.py).

The async `execute(improver_source, initial_source, capabilities)` adapter must run the supplied `async improve(initial, capabilities)` code in an isolated service. It exposes budgeted `suggest(source, guidance="")` and `evaluate(source)` endpoints through RPC. It must enforce time/memory/network restrictions; this package never executes generated code. The seed program samples candidates and picks maximum utility. `Problem(initial, utility_description, evaluate)` describes a downstream task, whose async evaluator also owns execution isolation.

`generations`/`evaluations` limit each outer improver invocation; `inner_generations`/`inner_evaluations` limit each downstream invocation. Final returned solutions and the proposed next improver are checked separately from their search budgets, as upstream. Explicit `ImproverExecutionError` or empty returned code yields zero meta utility/fallback; infrastructure failures propagate. The source's nonzero checked-utility acceptance is retained, without an invented elitist comparison. Count fields distinguish generation, utility and meta evaluations.

Failure rolls back the active optimizer executable to its predecessor while preserving the current source being optimized. A prior optimizer can then repair a successor that works downstream but fails when applied to itself.

Adaptations: async RPC capabilities replace unrestricted Python execution; JSON replaces code-fence parsing; a caller-defined problem collection replaces repeated evaluations of one benchmark. The default seed can be inspected without executing it. Configure Slick's template root to local `prompts/`. Checks use a scripted executor and prove recursion/budgets, not safe execution or benchmark performance: `rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_stop_optimizer`.
