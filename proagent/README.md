# ProAgent

Task-independent adaptation of [ProAgent: Building Proactive Cooperative Agents
with Large Language Models](https://arxiv.org/abs/2308.11339) (Zhang et al., AAAI
2024). One agent predicts teammates' intentions, selects one complementary skill,
checks its feasibility, and keeps executing low-level actions until the skill is
complete or invalid. Failures trigger explanation and bounded replanning.

## Official implementation

Adapted from [PKU-Alignment/ProAgent](https://github.com/PKU-Alignment/ProAgent),
revision `8aa0e6563b368d2220e0ac3fff44d0d971b5715b`, linked by the
[authors' project page](https://pku-proagent.github.io/). The upstream MIT license
is included in [LICENSE](LICENSE); exact files and modifications are in [NOTICE](NOTICE).

The upstream `ProMediumLevelAgent.action()` supplies the persistent-skill,
completion-check, verifier, explanation, and replanning flow. Its
`generate_belief_prompt()` supplies historical predictions alongside actual
teammate behavior. `Module.get_cache()` supplies recent-K retrieval. The L3
planner and precondition explainer prompts inform the local templates.

The original code depends on Overcooked, its path planner, old model APIs, and
baseline-model infrastructure. Those components are replaced by caller-supplied
callbacks. They are not needed to implement the cooperative algorithm.

## Connect a task

Install the adjacent Slick checkout with `pip install -r requirements.txt` from
this folder, or use an environment containing that checkout. The API was checked
against the adjacent `slick-ai` 0.3.0 source. No additional dependency is required.

```python
from pathlib import Path

import proagent
from proagent import ProAgent, Skill
from slick import prompts

# Once at application startup, before making any prompt calls.
prompts.TEMPLATE_ROOT = Path(proagent.__file__).resolve().parent / "prompts"


async def coordinate(initial_state, provider, ground, verify, control, step):
    agent = ProAgent(
        task="Complete the shared task efficiently with the other worker.",
        role="worker A",
        teammates=("worker B",),
        provider=provider,
        skills=(
            Skill("prepare", "Prepare an available item; arguments: item_id (string)."),
            Skill("assemble", "Assemble an item whose parts are ready; arguments: item_id (string)."),
        ),
        knowledge="Supply the actual task rules, layout, constraints, and skill preconditions.",
        examples="Optional examples of observations and good cooperative decisions.",
        ground=ground,
        verify=verify,
        control=control,
        step=step,
    )
    result = await agent.run(initial_state, max_steps=100)
    return result, agent
```

The skill names above illustrate the interface: supply your own skills and
callbacks. `State` and `Action` may be any caller-defined Python types.

| Callback | Contract |
| --- | --- |
| `ground(state) -> Observation` | Synchronously describe the current state and report observed teammate events. Must be repeatable without consuming or advancing the environment. |
| `await verify(state, skill) -> Verification` | Authoritatively check completion and preconditions, including arguments, permissions, resources, and reachability. Return `"complete"`, `"ready"`, or `"invalid"` and diagnostics. |
| `await control(state, skill) -> Action` | Produce one low-level action for the skill from the fresh state. Do not execute it. The controller may recompute paths or use a learned policy. |
| `await step(state, action) -> State` | Execute exactly that action and return the new state, incorporating teammate/environment updates. Own external side effects and execution cleanup here. |

`SkillCall` contains a name and JSON-compatible argument dictionary. The agent
checks that generated skill names exist and that intentions name exactly the
configured teammates. The verifier owns task-specific argument and feasibility
checks. Neither generated skills nor model explanations are executed as Python.

For example, grounding may return:

```python
from proagent import Behavior, Observation, Verification

Observation(
    step=12,
    description="Part A is ready; worker B is near the assembly station.",
    behaviors=(Behavior(step=11, teammate="worker B", action="prepared part A"),),
    terminal=False,
)
Verification("invalid", "Cannot assemble: part B is still missing.")
```

Steps are environment timestamps supplied by grounding. Report actual observed
behavior, not a guessed intention. Delayed events keep their original step.
Identical `(step, teammate, action)` events are deduplicated. Use a finer step
identifier if two identical events at the same step must be distinguished.

## Algorithm and memory

`run()` calls `act()`, executes its returned action through `step()`, and observes
the new state. `act()` also works independently when the caller manages the
environment loop. It returns `Decision(status="action", action=..., skill=...)`,
or a terminal/exhausted decision without an action. Branch on status rather than
on the action value, since `None` may itself be a valid caller-defined action.

On each fresh state, an active skill is verified again and the controller produces
its next low-level action. No new planning call is needed while that skill remains
valid. Completion or failure clears it. A proposed skill that is already complete
consumes a planning attempt and triggers a request for useful unfinished work.

The default planner follows L3: a concise analysis, teammate intention predictions,
and exactly one skill. Knowledge and optional examples are persistent constructor
inputs. `memory` stores planning-time states, plans, verifier feedback, statuses,
and counts of issued actions. `recent_k=1` supplies the latest trajectory to each
planner call; zero omits trajectories. The count refers to trajectory entries,
not the upstream implementation's individual chat messages.

With `belief_revision=True` (default), **all** recorded predictions and observed
behavior are additionally rendered into the next planning prompt, as in the
official belief mechanism. Predictions stay unchanged as historical hypotheses;
the planner revises its next intentions in light of actual behavior. No fabricated
match labels or additional belief-model call is introduced. Multi-step skills and
delayed events are not automatically paired merely because their timestamps match.
Disabling belief revision omits this paired history; planning still predicts
intentions, so this switch is not the paper's L2 ablation.

Invalid skills are never authorized by an LLM. The authoritative check rejects
them, then `analyze_failure()` explains unsatisfied preconditions before `replan()`.
One round is the default, matching the authors' practical setting.
`verification_rounds=3` additionally calls `double_check()` and
`conclude_failure()`. Each operation owns a separate Jinja template.

## Budgets and failures

- `max_steps` bounds environment actions in `run()`. Planning and explanation
  calls do not advance the environment. Zero steps makes no calls.
- `max_plan_attempts=5` allows the initial plan plus four replacements per `act()`;
  this makes the upstream retry count explicit. Continuing an active skill does
  not consume a new planning attempt. Malformed JSON, wrong teammate keys,
  unknown skills, unusable plans, and controller failures consume attempts.
- Explanations occur only when another planning attempt remains. A failure of an
  already-active skill can require an explanation before the first new plan.
  Each explanation costs one or three additional model calls. Inspect `calls`
  for actual operation counts; internal provider transport retries are external.
- Exhaustion returns `"exhausted"` without executing a fallback action. This
  intentionally replaces upstream's Overcooked-specific `wait(1)`/random motion.
  The caller can decide how to recover in its own domain.
- A controller may raise `SkillFailure` when it cannot produce an action, such as
  when every route is blocked. It must not have executed an action before raising.
  Other callback errors, provider failures, and failed explanation calls propagate;
  there are no automatic transport or side-effect retries.
- Raw responses and prompts are retained in `calls` before structured parsing and
  domain checks. `attempts` records planning rejections; `events` records verifier
  and controller outcomes, including exceptions and cancellation.

The result contains final `state`, `decisions`, and `stop_reason` (`terminal`,
`exhausted`, or `budget`). Decisions may include a final non-action status.
`memory.actions` counts issued controller actions, not verified successful
environment transitions. Skill status reflects the latest verifier check; stopping
an episode alone does not establish that the active skill completed. Final
teammate observations are retained even when the last action ends the episode.

Use a fresh instance for each episode and call it serially. Memory is explicit;
no hidden Slick Session carries history. Providers own transport timeouts and
retries. Callers own callback deadlines, cancellation cleanup, and environment
isolation. Slick's template root is process-global: set it once, not concurrently
for different agents. Full belief history is kept in memory; long episodes may
need a caller-specific retrieval policy.

This implements the enhanced L3/belief-revision algorithm, not the paper's main
L2 benchmark configuration. Relevant-K embedding retrieval, task controllers,
Overcooked baselines, and benchmark reproduction are outside this module.

## Verification

From the repository root, using the Slick environment:

```sh
python -m unittest tests.test_proagent
```

Tests use the shared scripted provider and deterministic caller callbacks. They
check persistent skills, rejection/replanning, observed-belief context, bounded
calls, controller failures, error propagation, cancellation, memory, and all
prompt templates. These are algorithm/interface checks, not evidence of live-model
cooperation quality or reproduced Overcooked scores.
