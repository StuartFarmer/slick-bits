# Plan of Thoughts

`PlanOfThoughts` implements the search core of Houjun Liu's *Plan of Thoughts:
Heuristic-Guided Problem Solving with Large Language Models*. Supply any task as
text, including its constraints, what one incremental step means, and the desired
final artifact. No Game of 24 rules, datasets, execution tools, or benchmarks are
built into the agent.

```python
from pathlib import Path

import pot
from slick import prompts


async def solve(task, provider, evaluate=None):
    prompts.TEMPLATE_ROOT = Path(pot.__file__).resolve().parent / "prompts"
    agent = pot.PlanOfThoughts(task, provider, evaluate)
    result = await agent.run(depth=5, max_calls=256, time_limit=120)
    return result


# Within your application's async entrypoint:
# result = await solve(your_task, your_slick_provider)
# if result.best is not None:
#     print(result.best.answer)
```

Configure Slick's process-global template root once at application startup. The
absolute path above works from any launch directory once the repository is on
Python's import path. Importing `pot` does not change the root. Separate processes
are needed for concurrent applications using different template roots.

## Algorithm

A state contains committed steps (`prefix`) and one uncommitted `current` thought.
The initial `think` proposes a first thought. After each search action a separate
LM call samples a usefulness observation: `sure`, `likely`, or `impossible`.

| Action | Transition |
| --- | --- |
| `continue` | Commit the current thought and propose the next step. |
| `think` | Replace the pending thought while preserving the committed prefix. |
| `rollback` | Discard the pending thought and pop the last committed step into its place. |

Each planning round simulates actions through a tree of action–observation
histories. An observation key includes the complete resulting textual state and
its sampled judgment. Equal text with different judgments has different branches.
Repeated histories reuse their statistics. Each action keeps a visit count and
mean return; UCT selects `Q + exploration * sqrt(log(N) / N_action)` after trying
unvisited actions. `impossible` prioritizes rollback/rethinking; other labels
prioritize continuation. This heuristic only orders initial exploration and ties;
it does not prune branches or count as a correctness reward.

A new leaf gets a sequential rollout: repeatedly choose one best next thought
until the trajectory reaches `depth` or a thought declares completion. Rollouts
do not sample intermediate judgments. The evaluator scores the resulting full
trajectory, and its expected reward is backed up through the visited actions.
After planning, the highest mean-return action is executed with a fresh sample,
and the matching observation subtree becomes the root for the next round.

The agent retains assessed candidates from both planning and execution, allowing
early success during a rollout. Complete candidates take precedence over partial
ones; within either group, greater success probability wins, with stable ties.
The final thought of a complete candidate must contain the standalone answer or
artifact. Partial candidates expose their latest step as `answer`; check
`best.terminal` before treating it as a complete deliverable.

## Providers and evaluation

```python
agent = pot.PlanOfThoughts(
    task=your_task,
    provider=thought_provider,
    judge_provider=larger_judge_provider,
    rollout_provider=greedy_thought_provider,
    evaluate=your_async_evaluator,  # optional
)
```

Both optional providers default to `provider`. The judge provider handles sampled
usefulness observations and, absent a callback, final evaluation. The rollout
provider handles sequential completion; configure its decoding to be greedy or
low-temperature. Providers own sampling settings. All model calls use independent
Slick prompt operations with local Jinja templates and explicit typed JSON output.
No mutable Session is shared across branches.

`evaluate(trajectory: tuple[str, ...]) -> float` is async and returns the
probability that the final artifact correctly solves the task, in `[0, 1]`.
A deterministic checker can return `1.0` or `0.0`. This is a success probability,
not arbitrary fitness. The callback replaces final judging only. It owns any
execution isolation, data access, or domain verification it needs. The agent
never executes generated artifacts.

## Budgets and results

`run()` accepts these search controls:

| Argument | Default | Meaning |
| --- | --- | --- |
| `depth` | `5` | Maximum thoughts in a trajectory, including the final artifact. |
| `simulations` | `16` | Simulations per planning round. |
| `max_actions` | `32` | Executed actions after the initial proposal. |
| `max_calls` | `256` | Total provider and external evaluator invocations. |
| `time_limit` | `None` | Overall wall-clock seconds, including generation and evaluation. |
| `exploration` | `sqrt(2)` | UCT exploration coefficient, in reward units. |
| `success_threshold` | `0.95` | Minimum probability for a complete candidate to stop search. |
| `reward_min`, `reward_max` | `0`, `1` | Failure and success rewards. |

Simulation action depth is bounded by `2 * depth` so rollback/rethink loops cannot
run indefinitely. Caller settings are trusted; choose positive depth, nonnegative
budgets, and `reward_max > reward_min`. Zero call budget makes no requests.

The result contains `best` (or `None` if no trajectory was assessed), the current
executed `state`, `solved`, stop `reason` (`solved`, `calls`, `time`, or `actions`),
and call, completed simulation, and executed action counts. `best` includes the
whole trajectory, completion flag, probability, expected reward, and `answer`.
`solved` reports the evaluator's judgment, not independently established truth.

Budget exhaustion returns the best candidate already assessed; it never makes an
extra finalization call. A deadline cooperatively cancels in-flight async work
and awaits cancellation cleanup; blocking callbacks or providers that suppress
cancellation cannot meet a strict wall-clock limit. External cancellation propagates.
Provider timeouts, malformed JSON, invalid generated fields, invalid evaluator
probabilities, and other errors propagate without implicit retries. Any retries
inside a provider or evaluator are outside the agent's invocation count.

`agent.calls` records prompts, raw responses, callback measurements, and errors,
including parsing failures. `agent.history` records executed actions and their
observations; `agent.root` exposes current tree statistics. Completed work and the
incumbent remain inspectable after an exception. Calls are sequential; use one run
at a time per instance. A new run resets all search state and records.

## Paper interpretation and limits

- The paper does not specify a latent utility transition model or likelihood
  linking hidden utility to observations. This implementation uses the observable
  text and sampled judgments directly in history-based PO-UCT, the search part of
  [POMCP](https://papers.nips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html).
  It does not invent utility particles or implement a general POMDP belief filter.
  Judgment-based action ordering is an explicit heuristic choice where the paper
  leaves initialization unspecified.
- Prefix edits are deterministic; newly proposed text and judgments are provider
  samples. Textually distinct outcomes create separate observation branches.
  Very diverse free-form generations can therefore limit history reuse.
- Slick's checked `Provider.acall` returns text, not token log-probabilities. The
  default JSON judge therefore supplies a self-reported confidence estimate,
  **not** the paper's measured posterior. Inject a calibrated evaluator to supply
  that signal. A greedy-configured rollout provider approximates Algorithm 1's
  argmax decoding; prompt wording alone cannot enforce it.
- Reward is `p * reward_max + (1 - p) * reward_min`, with discount one and no
  intermediate rewards. The paper's additional division by
  `reward_max + reward_min` is omitted: it is not expected-reward normalization
  and is undefined for symmetric positive/negative rewards. Default `0/1` rewards
  coincide with the printed equation.
- Task-specific few-shot prompts and the Game of 24 benchmark are intentionally
  absent. This is a reusable algorithm implementation, not a reproduction of the
  reported 89.4% success rate.

## Offline checks

Uses the existing Slick/Pydantic environment; no new dependencies.

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_pot
../slick/.venv/bin/ruff check pot tests/test_pot.py
```

Tests use the shared `tests.providers.ScriptedProvider` through real Slick
rendering and parsing. They cover transitions, alternative exploration, UCT
backups, subtree reuse, rollout provider separation, budgets, deadlines,
malformed generation, evaluator rejection, and rendering from another directory.
These are algorithm and interface checks, not paid model evaluation.
