# ADaPT

Problem-agnostic Slick implementation of *ADaPT: As-Needed Decomposition and
Planning with Language Models*. It attempts each task directly with a ReAct
executor, decomposes only after failure, and recursively executes the resulting
AND/OR plan. AND stops on failure; OR stops on success. Parentheses support mixed
logic without spending additional decomposition depth.

```python
from pathlib import Path

from slick import prompts

import adapt

# Configure once at application startup, before any generation.
prompts.TEMPLATE_ROOT = Path(adapt.__file__).resolve().parent / "prompts"

async def solve(task, provider, evaluate, actions, observation=""):
    agent = adapt.ADAPT(
        task,
        provider,
        evaluate,
        actions=actions,
        observation=observation,
    )
    return await agent.run(max_depth=3, max_executor_steps=20, max_calls=256)
```

Supply any task text, a configured Slick provider, action instructions, and an
async `evaluate(history: tuple[Step, ...], action: str) -> str`. The evaluator
returns the actual observation after applying the action **from exactly the
supplied history**. For a simulator, restore a snapshot or reset and replay that
history before applying the action. For reasoning or document tasks, compute the
next observation from that history directly. The agent never executes generated
code itself; the caller owns action validation, tools, and execution isolation.

`history` contains action/observation pairs, including earlier actions within the
current attempt. Successful attempts extend the checkpoint. Failed direct
attempts are discarded before decomposition, and failed OR alternatives cannot
leak actions or successful inner subgoals into the next alternative. Do not use
blind replay with irreversible external actions. The callback may leave its
physical environment at the last attempted action; `result.steps` is the accepted
checkpoint, and callers own any final restoration or commit.

`actions` describes available commands, syntax, atomic skills, and constraints.
`observation` describes the initial state. Optional `executor_examples` and
`planner_examples` provide domain demonstrations. `planner_provider` optionally
uses a different model for planning; the executor continues using `provider`.
No benchmark, model, tool, credentials, or provider setup is hardcoded.

The executor generates one textual action or thought per call. It reports its
own success with `think: Task completed!`, or failure with `think: Task failed!`.
Only these standalone thought markers finish an attempt; their appearance in an
action or observation does not imply success. Thoughts are never sent to the
evaluator. Bare commands are accepted, as well as `action: <command>`.

The planner uses the paper's textual protocol:

```text
Step 1: Obtain the input using source A
Step 2: Obtain the input using source B
Step 3: Produce the requested output from the obtained input
Execution Order: ((Step 1 OR Step 2) AND Step 3)
```

Parsing uses Python's AST without executing expressions. Every declared step
must be referenced exactly once; repeat a task with distinct step numbers when
needed. Invalid IDs, duplicate references, blank steps, unsupported expressions,
and malformed parentheses raise `ValueError`. AND takes precedence over OR;
explicit parentheses are recommended. Both prompt operations deliberately keep
textual output rather than introducing a JSON protocol.

Root depth is one. `max_depth=1` is executor-only; tasks at the depth limit still
get an execution attempt, but no planner call. `max_executor_steps` counts every
executor generation, including thoughts and completion markers. Hitting this
limit is an execution failure and permits decomposition. `max_calls` counts all
planner/executor generations, including unsuccessful generation attempts; its
exhaustion raises `RuntimeError`. Evaluations and any provider transport retries
have separate costs owned by the caller.

`Result.completed` is the LLM's success heuristic, **not a verified reward**.
`Result.steps` holds accepted actions and observations (possibly partial progress
on failure), `executions` records each finished executor attempt with its depth,
local steps, and stopping reason, and `calls` counts generations. No extra parent
execution or success judge runs after a successful plan. Provider, parser, and
evaluator exceptions propagate without retries or fallback. Partial
`agent.executions`, `agent.evaluations`, and `agent.calls` remain inspectable;
call records retain raw responses before parsing, and error records survive
failures. Runs reset records; use one run at a time per instance.

Slick's template root is process-global. Imports never change it. Configure the
absolute root above once; it works from other directories when this repository
is on `PYTHONPATH`. Different simultaneous template roots need separate processes.
This uses the adjacent Slick checkout declaring version 0.3.0, with no new
dependencies and no shared mutable Session: the controller owns prompt history.

## Official implementation and adaptations

The [project website](https://allenai.github.io/adaptllm/) identifies
[archiki/ADaPT](https://github.com/archiki/ADaPT) as the official code. This port
used commit `ecdc4ab0030b4be9be122622d8ea78f8c59c44c4`, specifically:

- [`run_alfworld.py`](https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/run_alfworld.py):
  `plan_and_run`, `alfworld_run`, `parse_expression`, `fetch_args`, and `plan_to_args`
  supplied the executor-first flow, depth boundary, checkpoint behavior, ordered
  short-circuit evaluation, and textual plan convention.
- [`run_webshop.py`](https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/run_webshop.py):
  checkpoint restoration, separate execution modules, and failed-attempt state
  communication to the planner informed the generic boundary.
- [`run_textcraft.py`](https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/run_textcraft.py):
  recursive task execution and observation handoff informed the controller.
  Its live inventory behavior differs from the restore-before-attempt convention
  selected here from ALFWorld/WebShop.

The upstream MIT notice is retained in [LICENSE.upstream](LICENSE.upstream).
The environment-specific scripts are references, not runtime dependencies: they
construct benchmarks and providers at module scope. The local controller adapts
their algorithm into ordinary async Python and external Slick templates.

Intentional differences: domain prompts and hardcoded models are replaced with
caller context; accepted history replaces domain-specific salient facts; exact
terminal markers replace substring checks; invalid plans are rejected instead of
silently falling back to AND; a total generation budget is added; failed compound
OR branches are fully isolated. Paper Algorithm 1 would generate an unusable plan
at maximum depth; this follows the official code's early return instead. Full
accepted history can grow with the call budget. There are no environment-specific
reward overrides or repeated-no-op patience rules.

This implements the algorithm, not the published benchmark results. LLM
self-assessment can be wrong, and problem-specific observations and atomic-skill
examples affect performance. No paid model calls or benchmark reproduction were
performed.

## Verification

```sh
rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_adapt
rtk proxy ../slick/.venv/bin/ruff check adapt tests/test_adapt.py
rtk proxy ../slick/.venv/bin/ruff format adapt tests/test_adapt.py --check
```

The shared scripted provider exercises real Slick rendering/postprocessing,
executor-first recursion, checkpoint isolation, mixed logic, stopping budgets,
raw error records, and template loading outside the repository. During development,
the official parser was also compared with this parser on 32 truth assignments
across homogeneous and mixed plans; outcomes and short-circuit visit orders matched.
