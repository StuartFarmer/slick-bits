# PINSKY

Co-generates environments and parameterized agents using mutation, minimal
playability criteria, differential evolution, cross-environment transfer, and
oldest-first culling. This is a problem-agnostic implementation of **Co-Generation
of Game Levels and Game-Playing Agents** (Dharna, Togelius, and Soros, 2020).
[SOURCES.md](SOURCES.md) maps the algorithm to the pinned official implementation
and records deliberate adaptations.

Environments are serialized strings. Agents are flat NumPy parameter vectors;
your evaluator defines their architecture and how they act. This supports any
domain with an executable parameterized policy and meaningful weak/strong solver
checks. It does not turn DE into an optimizer for arbitrary prose agents.

## Connect a problem

```python
from pathlib import Path

import numpy as np
from slick import prompts
import pinsky

# Configure once at application startup; Slick's template root is process-global.
prompts.TEMPLATE_ROOT = Path(pinsky.__file__).resolve().parent / "prompts"


async def coevolve(task, provider, evaluate, random_solve, strong_solve,
                   seed_environment, initial_parameters):
    search = pinsky.PINSKY(
        task, provider, evaluate,
        random_solve=random_solve, strong_solve=strong_solve, seed=42,
    )
    return await search.run(
        seed_environment, np.asarray(initial_parameters, dtype=float),
        config=pinsky.Config(iterations=100, de_evaluations=200),
    )
```

Supply these application-owned boundaries:

- `task`: describe the environment format, legal edits, protected elements, and
  task family. Separate local Slick prompts perform remove, add, and move.
- `await evaluate(environment, parameters) -> Evaluation(score, solved)`:
  run the parameterized policy in that environment. Scores are maximized and
  must be finite; `solved` reports actual success independently of reward.
  Callbacks receive copies of parameters. Policy decoding, rollouts, random seeds,
  time limits, held-out data, and execution isolation belong here.
- `await random_solve(environment) -> bool` and
  `await strong_solve(environment) -> bool`: run weak and strong reference solvers.
  Admission requires **random fails AND strong succeeds**, regardless of the
  inherited policy's score or success. For the paper's games these are random
  action selection and MCTS with 40 ms planning per action. Use domain-appropriate
  solvers elsewhere; model claims of solvability do not establish playability.

To use a native generator, supply
`mutate=async_mutation` with signature
`async_mutation(environment, operation, rng) -> str`, where `operation` is
`"remove"`, `"add"`, or `"move"`. It performs **one** edit; PINSKY controls repetition.
This bypasses Slick generation, so `provider=None` is appropriate and template
configuration is unnecessary. No optimizer callback is needed: DE runs locally.
Dependencies are the existing Slick, Pydantic, and NumPy installation.

## Algorithm and budgets

One fresh `PINSKY` instance owns one run. The caller supplies the seed environment
and unoptimized parameters; the seed is evaluated without the admission gate.
The short run loop performs mutation/culling, optimization/reevaluation, transfer.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `iterations` | 5000 | Outer loop count |
| `mutation_timer` | 25 | Mutate at zero-based loops 0, 25, 50, … |
| `max_children` | 8 | Total attempted children per mutation phase |
| `mutation_rate` | 0.8 | Probability a child receives edits |
| `continuation_rate` | 0.5 | Probability of another edit after each edit |
| `operation_probabilities` | (0.25, 0.5, 0.25) | Remove, add, move |
| `max_environments` | 30 | Active pair capacity; retire oldest admissions |
| `population_size` | 50 | DE population; use at least 4 |
| `de_evaluations` | 1500 | Trial evaluations per active pair per loop |
| `scaling_factor`, `crossover_rate` | 0.6, 0.4 | DE/rand/1/bin settings |
| `lower_bound`, `upper_bound` | -5, 5 | Scalar parameter bounds |
| `transfer_timer` | 10 | Transfer after loops 10, 20, 30, … |

Parents are sampled uniformly with replacement from the phase-start population.
Accepted children copy their parent's parameters and remain alongside parents
until age culling. Rejected attempts still consume `max_children`. An edit gate
that does not fire creates a clone, which still receives both solver checks,
matching upstream. There is no novelty ranking or duplicate filter.

Each DE phase starts afresh: uniform `[-1, 1]` vectors plus the incumbent in slot
zero, clipped to the configured bounds. Three distinct non-target donors form
each trial, at least one coordinate crosses over, and selection is deferred until
all trials in that generation are evaluated. DE accepts ties. A final partial
generation consumes the exact remaining trial budget. Only the winning vector
carries into the next outer loop. Use nonempty flat vectors and usable settings;
configuration is trusted, following the Slick development contract.

`de_evaluations` excludes the `population_size` initial evaluations and one final
reevaluation per pair. Thus the default costs **1551 evaluations per active pair
per loop**, plus the one seed evaluation and `N²` evaluations per transfer phase.
This makes the official code's initialization overhead explicit despite the
paper table calling `nGames=1500` the evaluations per optimization step.
Both solver calls are counted separately; their internal rollouts are caller-owned.
Mutation chains use the paper's geometric stopping rule, so total prompt calls
have no deterministic cap. Set `continuation_rate=0` for exactly one edit when
mutation fires; keep it below 1 for eventual stopping.

All transfer measurements use frozen agent snapshots, including fresh incumbent
measurements. Only strictly better agents replace incumbents; ties retain the
incumbent. Agent source and destination can swap in the same transfer phase.
Scores are compared only within an environment, never across different tasks.

## Results and failures

`result.active` contains final active pairs; `result.retired` retains culled
pairs. Pair IDs and `parent_id` reconstruct environment lineages, including
parents that have retired. `born` is the zero-based admission loop. Retired
children culled before their first optimization can have `evaluation=None`.
`result.attempts` stores generated environments, operations, solver outcomes,
acceptance, and explicit rejection errors. `result.transfers` records source,
target, loop, and before/after score. There is no globally best agent across
incomparable environments. These are final/retirement snapshots, not every
historical policy or a durable checkpoint.

`result.generations` retains `(rendered_prompt, raw_response)` before parsing,
including malformed responses. Native mutation callbacks do not create these
records. Blank/invalid model outputs and caller-raised `CandidateRejected`
consume one child attempt and preserve incumbents. Solver failures return false;
raise `CandidateRejected` for malformed domain artifacts. Unexpected callback or
provider errors, nonfinite fitness, and cancellation propagate. Completed records
remain on `search.result`; there are no implicit retries or shared sessions.
Your adapter must enforce protected edits and any required execution isolation.

## Checks

```sh
rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_pinsky
```

Tests exercise numerical optimization, budgets, viability, inheritance, culling,
transfer snapshots, external prompt rendering, and failure records with the shared
scripted provider. These are algorithm/interface checks, not reproductions of
the paper's GVGAI results. No paid model calls or game infrastructure are needed.
