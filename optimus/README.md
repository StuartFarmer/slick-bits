# OptiMUS

`OptiMUS(provider, execute, interface=...).run(problem, data, test_code=None, repairs=3, augmentations=0)` implements the original sequential formulation/code/test/repair method. Supply a structured description of the problem type, parameters, constraints, objective and input/output format. Actual instance data goes only to the executor, never the optimizer model. The interface documents the chosen solver and code/test contracts.

Sources: [original paper](https://arxiv.org/abs/2310.06116), official **optimus-v0.1** branch [`gpt4or.py`](https://github.com/teshnizi/OptiMUS/blob/optimus-v0.1/gpt4or.py) and [templates](https://github.com/teshnizi/OptiMUS/tree/optimus-v0.1/templates). Later v0.2/v0.3 agent/RAG algorithms are different papers and are not silently substituted.

The async `execute(code, tests, data)` must execute both artifacts in caller-owned isolation and return `Execution(status, output, feedback)`. Status is `passed`, `execution_error`, `test_failure`, or `invalid_test`. Execution errors and semantic test failures trigger distinct repair prompts; tests stay fixed. Invalid tests stop without declaring success. Human test code bypasses test generation. Optional augmentation rephrases the problem after an exhausted attempt and retries the pipeline. Inspect `result.execution.status`; returning an artifact alone is not a claim of correctness.

Adaptations: JSON replaces fenced code; bounded loops perform at most `repairs + 1` executions per variant and never leave an untested final repair. Rephrasing is integrated as bounded fallback rather than file-based preprocessing. No solver, direct generated-code execution, forced Gurobi choice, benchmark data or human-review UI is bundled. This is generic across solver-modelable problems, not a text-prompt optimizer for every application.

Set Slick's global template root to local `prompts/`. Returns the final result, every variant/attempt and execution count. Check: `rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_optimus`.
