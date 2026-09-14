# Evolution of Heuristics (EoH)

A problem-agnostic Slick implementation of **Evolution of Heuristics: Towards
Efficient Automatic Algorithm Design Using Large Language Model** (ICML 2024).
EoH evolves a population of reusable heuristics, each represented by a natural
language thought (`description`), an implementation (`content`), and measured
`fitness`. It designs algorithms; the caller defines the problem and evaluation.

## Use it on your problem

```python
from pathlib import Path

import eoh
from slick import prompts

# Set once at application startup; independent of the working directory.
prompts.TEMPLATE_ROOT = Path(eoh.__file__).resolve().parent / "prompts"

async def design(task, provider, evaluate):
    agent = eoh.EoH(task, provider, evaluate, maximize=False)
    population = await agent.run(
        population_size=20, generations=20, parents=5, seed=0,
    )
    return population[0], agent

# task: describe the problem, objective, required function/interface, input and
# output semantics, implementation language, dependencies, and constraints.
# evaluate: async evaluate(content: str) -> float, supplied by your application.
# best, agent = await design(task, provider, evaluate)
# print(best.description, best.content, best.fitness)
```

The evaluator receives the complete candidate content and returns an aggregate
fitness over your training instances. Use the same instance set and scoring rule
for comparable measurements, and evaluate generalization on held-out instances
separately. Lower fitness wins with `maximize=False`; the API defaults to
maximization. No benchmark, language, function signature, or objective is hardcoded.
Python heuristic code is the paper's choice; other implementation formats work
when specified in the task and understood by the evaluator.

The evaluator owns syntax/interface checks, feasibility checks, execution
isolation, dependencies, timeouts, and worker cleanup. The optimizer never
executes generated content. Raise `eoh.CandidateRejected("reason")` for an
infeasible candidate. Returning a nonfinite score or raising `TimeoutError` also
rejects it. Other evaluator exceptions, including `ValueError`, are recorded and
propagated so evaluator bugs are visible.

## Algorithm

1. Generate and evaluate N initial thought/implementation pairs. Stop once N are
   valid, with at most `init_attempts` calls (default `3 * population_size`).
   Exhausting initialization raises `RuntimeError`; partial work remains in
   `agent.attempts`.
2. Each generation takes a ranked population snapshot. Each selected operator
   generates N candidates from that snapshot, and every parsed candidate is
   evaluated once:

   | Operator | Parent count | Action |
   | --- | --- | --- |
   | E1 | p | Explore an idea as different as possible from the parents |
   | E2 | p | Identify a common idea and introduce new components |
   | M1 | 1 | Modify the structure to improve performance |
   | M2 | 1 | Adjust parameters/settings while preserving the structure |
   | M3 | 1 | Remove redundant components |

3. Select the best N from incumbents and all feasible offspring. Stable fitness
   ties favor incumbents. Selection happens after all operator batches, following
   the paper's Section 3.2. Calls are sequential; all use the same generation
   snapshot. The best fitness therefore cannot deteriorate.

Parents are sampled **with replacement**, matching official `prob_rank.py`,
with weight `1 / (r + N)` for one-based fitness rank r. Repeated parents are
possible, including when p exceeds N. Each method has a separate local Jinja
prompt, with explicit JSON schema, thought-first instructions, and complete
implementation output. `Proposal` validates nonblank fields and rejects extra
fields; implementation whitespace is preserved. No syntax or correctness claim
is made by schema validation.

With no failures, default N=20, G=20 and five operators use
`N + 5*N*G = 2,020` proposal calls, including initialization. Invalid evolutionary
attempts are not retried or replaced. `operators=("E1",)` or `("E1", "E2")` selects
the paper's operator ablations; these do not implement its code-only ablation.
Provider adapters may make their own transport retries, outside this count.

## Records and Slick ownership

- `agent.attempts`: every attempt's ID, generation, operator, parent IDs, raw
  response (when returned), parsed `Proposal` (when valid), fitness (when finite),
  status, and error. `accepted` means eligible for selection, not necessarily a
  survivor. Raw malformed JSON remains available.
- `agent.evaluations`: calls started to the evaluator, including rejected and
  failed evaluations. `len(agent.attempts)` counts proposal attempts.
- `agent.history`: immutable ranked population snapshots after initialization
  and each completed generation. The result is the final best-first population.

Each run resets these records and the seeded parent-selection RNG. Provider
sampling and evaluator randomness need their own reproducibility settings.

By default each generation call uses the injected provider independently.
Provider errors and generation timeouts consume an attempt. Optional
`run(session=your_session)` retains caller-owned conversational history and tools;
this is an extension that changes the LLM context relative to independent calls.
Session responses finish as text before schema parsing, so malformed proposals
can be rejected without leaving a pending conversation. Session transport errors
propagate, allowing the caller to resume the pending conversation. No new Session
is created by the optimizer. An agent/Session supports one run at a time.

Slick's template root is process-global. Configure it before rendering; concurrent
applications requiring different roots need separate processes. To render without
generation, use `await EoH.explore_diverse.render(agent, parents)` with the owner
explicitly bound. Provider setup, durable logging, and persistence belong to the
application.

## Official implementation and deliberate adaptations

Reference: [official FeiLiu36/EoH repository](https://github.com/FeiLiu36/EoH),
legacy release **v0.1**, commit
[`4b2338fc2317c743cec50c5c7a2e4684707af973`](https://github.com/FeiLiu36/EoH/tree/4b2338fc2317c743cec50c5c7a2e4684707af973).
The repository's current main branch has a newer API; this implementation uses
the legacy algorithm sources alongside the supplied paper, not that newer API.

| Official source inspected and used | Relationship to this implementation |
| --- | --- |
| [eoh_evolution.py](https://github.com/FeiLiu36/EoH/blob/4b2338fc2317c743cec50c5c7a2e4684707af973/eoh/src/eoh/methods/eoh/eoh_evolution.py) | INIT/E1/E2/M1/M2/M3 prompt semantics adapted to caller-defined tasks and Slick JSON output |
| [prob_rank.py](https://github.com/FeiLiu36/EoH/blob/4b2338fc2317c743cec50c5c7a2e4684707af973/eoh/src/eoh/methods/selection/prob_rank.py) | Same one-based rank weights and `random.choices` sampling with replacement |
| [eoh.py](https://github.com/FeiLiu36/EoH/blob/4b2338fc2317c743cec50c5c7a2e4684707af973/eoh/src/eoh/methods/eoh/eoh.py) | Population evolution reference; this version follows the paper's end-of-generation selection rather than upstream selection after every operator |
| [eoh_interface_EC.py](https://github.com/FeiLiu36/EoH/blob/4b2338fc2317c743cec50c5c7a2e4684707af973/eoh/src/eoh/methods/eoh/eoh_interface_EC.py) | Parent counts, per-operator candidate batches, and evaluated-fitness boundary; execution delegated to caller |
| [pop_greedy.py](https://github.com/FeiLiu36/EoH/blob/4b2338fc2317c743cec50c5c7a2e4684707af973/eoh/src/eoh/methods/management/pop_greedy.py) | Elitist selection; equal-fitness candidates are retained here to maintain the paper's N-sized population |

Additional explicit differences: initialization stops at N feasible candidates
rather than upstream's 2N initial proposals; bounded rejection replaces implicit
retries; fitness is not rounded; all five operators run by default. M3 retains
both thought and implementation as specified by the paper (the legacy M3 prompt
asks for revised code only). Structured JSON replaces upstream brace/regex
extraction. These are algorithm adaptations, not a claim of identical prompts,
random trajectories, benchmark results, or an exact reproduction.

Compared with the previous local implementation, parent sampling now uses
replacement, content whitespace is preserved, raw responses and evaluation counts
are retained, and evaluator `ValueError` now propagates. Use `CandidateRejected`
for intentional evaluation rejection.

## Verification

Requires the adjacent Slick checkout (`slick-ai` 0.3.0 inspected here), including
Pydantic and Jinja. From this directory, `python -m pip install -r requirements.txt`
installs that local dependency. With this repository importable, run:

```sh
python -m unittest tests.test_eoh tests.test_prompt_layout
```

Offline checks cover all six templates, typed parsing, full operator budgets,
rank weights and replacement, generation snapshots, maximization/minimization,
elitism and ties, infeasible candidates, failure accounting, raw output,
whitespace preservation, and caller-owned Sessions. These are algorithm and
integration checks; no paid LLM run or paper benchmark reproduction is claimed.
The previous problem-specific application remains in
`examples/legacy/2026-09-14-problem-specific.tar.gz`.
