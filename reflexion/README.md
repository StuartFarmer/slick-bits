# Reflexion

Problem-agnostic implementation of *Reflexion: Language Agents with Verbal
Reinforcement Learning*, Section 3: attempt → evaluate → reflect → remember → retry.
The actor improves through natural-language experience, without changing weights.

## Use

Use the existing `optimizer/.venv` environment or install the adjacent Slick
checkout with `python -m pip install -e ../slick` from the repository root.
No additional dependencies are needed.

Configure the prompt root once at application startup:

```python
from pathlib import Path

from slick import prompts
import reflexion
from reflexion import Evaluation, Reflexion

prompts.TEMPLATE_ROOT = Path(reflexion.__file__).resolve().parent / "prompts"


async def solve(task, input, provider, evaluate):
    agent = Reflexion(task, provider, evaluate)
    result = await agent.run(input, max_trials=4, memory_size=3)
    return result
```

`task` supplies the instructions, constraints, and required output format; `input`
is any text to work on. The provider is a caller-configured Slick provider. The
default actor generates arbitrary text, preserving its whitespace. Customize the
three local templates for prompt examples or domain instructions.

Supply an async evaluator accepting a `Trajectory` and returning `Evaluation`:

```python
async def evaluate(trajectory):
    passed = trajectory.output.strip() == "42"
    return Evaluation(passed=passed, feedback="Correct" if passed else "Incorrect")

result = await solve(
    "Answer with only an integer.", "What is 6 times 7?", provider, evaluate
)
print(result.trajectory.output)
print(result.stop_reason)
```

The example's answer key lives exclusively in the evaluator. For real tasks,
replace that callback with your tests, environment reward, heuristic, or LLM
judge. `Evaluation(passed, feedback="", reward=None)` supports a binary signal,
textual feedback, and an optional scalar reward. Only `passed` stops the loop;
the algorithm neither assumes a reward threshold nor selects a best-scoring trial.
The evaluator can close over private data, while deciding which feedback is safe
to expose to reflection. Evaluation accuracy determines the reliability of stopping.

## Environment actors

For interaction, tools, or an existing agent, inject
`actor(task, input, memory) -> Trajectory` as an async callback. `memory` is an
immutable tuple of the most recent reflections, ordered oldest first.

```python
from reflexion import Trajectory


async def actor(task, input, memory):
    # These are application-owned operations, not a required environment API.
    observation = await environment.reset(input)
    output, trace = await existing_agent.run_episode(
        task=task, observation=observation, reflections=memory
    )
    return Trajectory(output=output, trace=trace)

agent = Reflexion(task, provider, evaluate, actor=actor)
result = await agent.run(input, max_trials=12, memory_size=3)
```

The callback owns environment resets, within-episode action/observation history,
action limits, tools, and resource cleanup. Return actual observed history in
`trace`; the evaluator and reflector see both `output` and `trace`. The default
text actor leaves `trace` empty. This boundary supports ReAct-style episodes
without baking an environment or action grammar into Reflexion.

By default the same provider generates attempts and reflections. Pass
`reflection_provider=another_provider` to use a separate reflection model. With
an external actor, the supplied provider is used only for reflections unless
`reflection_provider` overrides it. An LLM evaluator likewise owns its provider.
The core never executes generated text. Execution isolation belongs to the
application's actor/evaluator.

## Algorithm and budgets

1. Start a fresh trial with the task, input, and bounded reflection memory.
2. Evaluate the complete trajectory and retain the result.
3. Return immediately on evaluator success.
4. If another trial is available, reflect on the trajectory, evaluation, and
   prior memory; append that reflection, dropping the oldest when full.
5. Retry, or return the last evaluated trajectory when the budget is exhausted.

`max_trials` includes the initial attempt. Zero returns `trajectory=None`, empty
history, and `stop_reason="budget"`, with no callbacks or model calls. A run of
N trials costs N actor calls, N evaluations, and at most N−1 reflections. With
the default actor that is at most 2N−1 model calls. No reflection is generated
after success or after the final failed trial.

`memory_size=3` follows the paper's typical reasoning/decision-making window;
use 1 for its programming memory setting. Zero discards reflections, although
the reflection operation still runs before retries. Caller settings are trusted.
The active memory is bounded by number of reflections, not tokens. Full trial
history and raw prompt records remain available for inspection and can grow.

Every `run()` resets state, including memory, so separate inputs do not inherit
lessons. Use one run at a time per instance. Prompt history is assembled explicitly;
there is no shared Slick Session. Providers must not add implicit conversation
history. A custom actor may use its own fresh Session per episode. Slick's prompt
root is process-global; use separate processes for simultaneous implementations
that require different roots. Importing this package does not change it, and an
absolute configured root works independently of the launch directory.

## Results and failures

`Result` contains `trajectory`, `trials`, `memory`, `stop_reason` (`"success"` or
`"budget"`), and `calls`. Each immutable `Trial` holds its trajectory and evaluation.
`calls` counts model requests made by this agent only; it excludes calls inside
actor/evaluator callbacks and transport retries hidden inside providers.

Blank model responses and tool requests in text-only prompt operations raise
`ValueError`. Provider and callback errors propagate without retries or fallback
answers. `agent.calls` records operation, prompt, raw response when available,
and generation errors, including rejected text before validation. Callback errors
propagate directly; they do not create model-call records. Partial `agent.trials`,
`agent.memory`, and the last completed `agent.trajectory` remain available. An
evaluation failure leaves its trajectory but does not append an evaluated trial.
Callers own logging persistence and retry policy.

## Official implementation reference

The paper links `noahshinn024/reflexion`; its current official repository is
[`noahshinn/reflexion`](https://github.com/noahshinn/reflexion). This implementation
uses the following upstream algorithm and prompt sources, inspected on 2026-09-14
at commit [`218cf0e`](https://github.com/noahshinn/reflexion/commit/218cf0ef1df84b05ce379dd4a8e47f17766733a0):

| Official source | Use here |
| --- | --- |
| [`programming_runs/reflexion.py`](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/programming_runs/reflexion.py) | Initial evaluation, failure → reflection → retry, early success, and no unused final reflection |
| [`alfworld_runs/generate_reflections.py`](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/alfworld_runs/generate_reflections.py) | Reflection uses the failed experience and previous plans to propose a concrete corrective plan |
| [`alfworld_runs/alfworld_trial.py`](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/alfworld_runs/alfworld_trial.py) | Fresh episodes and latest-three-reflection actor context |
| [`hotpotqa_runs/agents.py`](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/hotpotqa_runs/agents.py) | Reflection-only retry context and reset of within-trial history |

This is a deliberate generic Slick adaptation of those flows. The default actor
uses reflection-only context: failed trajectories inform the reflector, rather
than being replayed to the next actor. Upstream's programming variant additionally
passes the last implementation and test feedback to its actor. That task-specific
prompting, self-generated tests, held-out scoring, benchmarks, and experiment
infrastructure are left to the application. The local prompts are newly written
general instructions, not reproductions of benchmark few-shot prompts.

The supplied Algorithm 1 prints a loop condition using `or`; taken literally,
that can exceed the budget and continue after success. This implementation uses
bounded trials with early success, following the surrounding prose and official
programming loop. It physically bounds active reflection storage; upstream
AlfWorld retains all reflections but slices to three when constructing prompts.

Upstream is MIT licensed; its notice is retained in [LICENSE.upstream](LICENSE.upstream).

## Verification

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_reflexion
../slick/.venv/bin/ruff check reflexion tests/test_reflexion.py
../slick/.venv/bin/ruff format --check reflexion tests/test_reflexion.py
```

Tests use the shared scripted provider and real Slick decorators to check FIFO
memory, call ordering, budget accounting, early stopping, external trajectories,
separate reflection providers, run isolation, failure preservation, and template
rendering from another working directory. These are offline algorithm/interface
checks, not evidence of quality gains or reproduction of the paper's benchmarks.
