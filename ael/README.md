# AEL

Problem-agnostic implementation of **Algorithm Evolution Using Large Language
Model**, Liu, Tong, Yuan, and Zhang ([paper, 2023](https://arxiv.org/abs/2311.15249)).
It evolves reusable algorithms represented by a description and implementation
text. Supply a task, a Slick provider, and an async fitness evaluator; the optimizer
owns initialization, uniform parent selection, crossover, mutation, and elitism.

```python
from pathlib import Path

import ael
from slick import prompts
from ael import AEL, Proposal

# Application startup: independent of the launch directory.
prompts.TEMPLATE_ROOT = Path(ael.__file__).resolve().parent / "prompts"

async def evolve(task, provider, evaluate):
    agent = AEL(task=task, provider=provider, evaluate=evaluate)
    population = await agent.run()  # paper defaults
    return population[0], agent.history, agent.attempts
```

Describe the problem, implementation format, entry point, input/output names and
types, available libraries, and constraints in `task`. For example, ask for a
Python scheduling function, a numerical search procedure, or a routing policy.
The core has no TSP, Python execution, dataset, or solver dependency.

`evaluate(content: str) -> float` must be async. It receives the complete generated
implementation unchanged and returns aggregate fitness over a **fixed collection
of evaluation instances**, as in the paper. Lower fitness wins; pass
`maximize=True` to `AEL` for rewards. The evaluator owns syntax/interface checking,
execution isolation, time limits, and invalid-solution rejection. Keep held-out
instances outside the search. A finite score checks the measurement, not algorithm
correctness or generalization.

Optional existing algorithms can initialize the population:

```python
agent = AEL(task, provider, evaluate)
population = await agent.run(
    initial=[Proposal(description="Baseline method", content=baseline_source)],
    population_size=10,
    generations=10,
    parents=2,
    offspring=1,
    crossover=1.0,
    mutation=0.2,
    seed=0,
)
```

Seeds are evaluated in order until the population is full; remaining seeds are
unused. Rejected seeds do not fill a slot. LLM initialization fills missing slots,
with at most `init_attempts` calls (default `3 * population_size`), independently
of seed evaluations. A fully seeded population needs no initialization calls.
Failure to fill the population raises `RuntimeError` with attempts retained.

Each generation performs `population_size` parent selections, uniformly without
replacement within each selection, from the generation's starting population.
A crossover coin is drawn once per selection; when it fires, `offspring` separate
LLM calls generate children from those parents. Each child independently undergoes
mutation; only the final version is evaluated. Without crossover, a randomly
selected parent supplies each mutation input. Without either operation, the clone
reuses its measured fitness. This assumes comparable, stable fitness measurements.
After the full generation, parents and children compete for the best N slots;
stable ties favor incumbents. Results are sorted best first.

The defaults match Section IV-B: N=10, Ng=10, crossover=1, mutation=0.2, l=2,
s=1. With valid initialization and default direct-provider operation, these imply
110 crossover/initialization calls plus approximately 20 mutation calls, and
110 evaluations if all final candidates are valid. `seed` controls selection and
probability draws, not model generation or evaluator randomness.

`Proposal` contains nonblank `description` and `content`. Description whitespace
is trimmed; implementation whitespace is preserved. `Individual` adds `id` and
finite `fitness`. `agent.history` contains N-sized immutable population snapshots,
including initialization. `agent.attempts` records seed/generation IDs, operations,
parent IDs, proposals, measured fitness, and errors. A crossover immediately
mutated has a proposal but no fitness; its mutation records the crossover's ID.
Each run resets both collections. Distinct algorithms with equal fitness remain
eligible; scores are not rounded or used for deduplication.

Malformed generated proposals and evaluator `ValueError`/`TimeoutError` reject the
candidate. Evolution does not retry rejected offspring or fall back to an
unmutated crossover result. Nonfinite scores are rejected. Provider outages and
unexpected evaluator exceptions propagate after recording the error. Provider
transport retries belong to the caller; AEL adds none.

Calls normally use the constructor provider independently. Optional `session=`
uses a caller-owned Slick Session and its provider, which intentionally adds
conversation history and may use tools. Session parse failures propagate because
Slick retains a pending conversation for caller recovery. Raw exchanges remain
in `session.history`; direct-provider raw response logging belongs in the caller's
provider adapter. Attempt records contain parsed proposals/errors, not a complete
raw transcript. An agent instance supports one run at a time. Configure Slick's
process-global template root once; concurrent applications needing different
roots should use separate processes.

## Official source and adaptations

The author's [code collection](https://github.com/FeiLiu36/LLM4AlgorithmDesign)
links AEL to [FeiLiu36/EoH](https://github.com/FeiLiu36/EoH), now a later algorithm.
The original `FeiLiu36/AEL` homepage is unavailable. We inspected and used the
author's published [aell 0.0.1 source release](https://pypi.org/project/aell/0.0.1/)
as the implementation reference, including `src/aell/ael.py`,
`ec/interface_EC.py`, `ec/evolution.py`, `ec/selection.py`, and `ec/management.py`.
The source archive SHA-256 is
`50c5dd8c6ae459da39126d05786e6c5628dd0c6db8afd5df227e53f9730fed3e`.
It is a reference, not a runtime dependency; no upstream package needs installation.

| Source behavior | This implementation |
| --- | --- |
| Task/interface hints plus algorithm description and code | Caller task plus `Proposal(description, content)` |
| Initialization, parent-inspired generation, strategy/parameter revision | Separate local `initialization`, `crossover`, and `mutation` Jinja prompts |
| Seed code evaluation and deletion of worst individuals | Evaluated `initial` proposals and best-N survivor selection |
| Published release uses weighted parent sampling and immediate updates | Uniform sampling and generation snapshots follow the supplied paper |
| Release contains later e1/e2/m1/m2 operators | Paper's single crossover and mutation, applied sequentially |
| Tagged prose/code extraction and Python-specific prompts | Explicit structured JSON and caller-defined implementation format |
| Unbounded retries, score rounding, duplicate-score filtering | Bounded initialization, visible failures, unrounded fitness, stable ties |

Prompts contain parent descriptions and implementations only; measured fitness
and internal IDs are not sent as extra evolutionary feedback. The JSON output
format and generic wording are deliberate adaptations of the paper's tagged
TSP prompts. This implements the search algorithm; it does not reproduce the
TSP results or claim equivalent model behavior.

## Verification

From this directory, `python -m pip install -r requirements.txt` installs the
existing adjacent Slick checkout (checked against Slick 0.3.0). From the repository
root, run:

```sh
python -m unittest tests.test_ael tests.test_prompt_layout
```

Tests use the shared scripted provider and require no paid model calls. They cover
prompt parsing/rendering, code preservation, seeds, budgets, snapshot selection,
mutation lineage, multiple offspring, elitism, rejection, and exception ownership.
The previous domain-specific application remains in
`examples/legacy/2026-09-14-problem-specific.tar.gz`.
