# EoH-S

Evolves a small set of complementary heuristics for any task represented by text.
Inject a task description (including the desired artifact interface), a Slick
provider, and an async evaluator returning one **lower-is-better cost per instance**.
Code, prompts, rules, and other textual artifacts use the same algorithm.

```python
from pathlib import Path

import eoh_s
from slick import prompts
from eoh_s import EoHS, cpi

# Configure once at application startup; independent of the working directory.
prompts.TEMPLATE_ROOT = Path(eoh_s.__file__).resolve().parent / "prompts"

# Supply your provider and async evaluate(content) -> sequence of costs.
agent = EoHS(
    task="Your task, artifact format, interface, constraints, and instance context",
    provider=provider,
    evaluate=evaluate,
)
population = await agent.run(population_size=10, max_evaluations=2000, seed=0)
print(cpi(population))
artifacts = [individual.content for individual in population]
```

Keep evaluation instances and their order fixed throughout a run. The first valid
evaluation establishes the vector length; later vectors must match, be nonempty,
and contain only finite numbers. Normalize costs in your evaluator when instance
scales differ; negate rewards when maximizing. Instances, baselines, execution,
timeouts, and any required sandbox belong to the evaluator. This module never
executes generated artifacts or loads benchmark data.

`Proposal` contains a nonblank `description` and textual `content`. Prompt bodies
reject blank content without stripping artifact bytes. `Individual` additionally
contains `id`, immutable `scores`, and a mean-cost `fitness` property.

## Algorithm

1. Generate and evaluate N initial thought/artifact pairs.
2. For each of N offspring attempts, choose CS or LS with probability 0.5 from
   the population at the start of that generation. CS uses the pair with the
   greatest Manhattan distance across score vectors. LS sorts by mean cost and
   samples one parent with weight `1 / (N + zero_based_rank)`.
3. Combine parents and valid offspring. Greedy CPM starts with the best mean,
   then repeatedly selects the largest improvement in per-instance minima.
   Ties favor better mean cost, then original input order (incumbents first).
4. Stop at the evaluation or attempt cap; apply CPM to a partial final batch too.

`cpi(population)` is the mean of the per-instance minimum costs. It represents
evaluating all heuristics and taking the best outcome on each instance. It does
not learn which heuristic to dispatch before evaluation. Greedy CPM is an
approximation: adding candidates can change the first choice, so CPI across
successive fixed-size populations is not guaranteed to decrease.

Use a fresh agent per run. `history` holds immutable population snapshots;
`attempts` records operation, parents, raw response before parsing, proposal,
scores, and failures as available. `evaluations` counts actual evaluator calls,
including rejected or timed-out evaluations. Each decorated operation makes
one provider call with no conversational history or implicit retry. The caller
owns transport retries and durable logging.

The evaluation cap includes initialization. The separate attempt cap defaults
to `3 * max_evaluations`, preventing malformed output from looping forever.
Initialization also stops after `3 * population_size` attempts and raises if it
cannot fill the population. Invalid JSON, blank artifacts, provider errors, and
generation timeouts consume attempts. Invalid measured vectors, evaluator
`EvaluationError`, and evaluation timeouts consume evaluations too. Other
evaluator exceptions abort and are recorded; use `EvaluationError` for deliberate
candidate rejection. Failed offspring leave incumbents available for selection.
Population size one uses LS only. Caller configuration follows annotated types
without preflight range validation. Slick's template root is process-global;
configure it before calls and use separate processes for concurrent agents
requiring different template roots.

## Official implementation and deliberate adaptations

Source: Liu et al., *EoH-S: Evolution of Heuristic Set Using LLMs for Automated
Heuristic Design*, AAAI 2026. This implementation uses the algorithm and prompt
operations from the [official EoH-S repository](https://github.com/FeiLiu36/EoH-S),
inspected at commit `8310b056ab0d4ea83d194f1d9e8c7873c0a3b892`:

- [`population.py`](https://github.com/FeiLiu36/EoH-S/blob/8310b056ab0d4ea83d194f1d9e8c7873c0a3b892/code/llm4ad/method/eohs/population.py):
  adapted `survival_set` to lower-is-better costs, retaining greedy improvement
  and mean-based tie ordering; uses the `selection` rank-weight formula.
- [`eohs.py`](https://github.com/FeiLiu36/EoH-S/blob/8310b056ab0d4ea83d194f1d9e8c7873c0a3b892/code/llm4ad/method/eohs/eohs.py):
  initialization, reproduction, evaluation, and population replacement flow.
- [`prompt.py`](https://github.com/FeiLiu36/EoH-S/blob/8310b056ab0d4ea83d194f1d9e8c7873c0a3b892/code/llm4ad/method/eohs/prompt.py):
  INIT, complementary exploration, and local revision intent, deliberately
  rewritten as task-agnostic JSON contracts in three local Jinja files.

The official implementation is built on **LLM4AD**: Fei Liu et al.,
*LLM4AD: A Platform for Algorithm Design with Large Language Model*,
[arXiv:2412.17287](https://arxiv.org/abs/2412.17287), 2024;
[official platform repository](https://github.com/Optima-CityU/LLM4AD).
This port uses Slick for generation and stdlib numerical logic; it does not
require installing LLM4AD or its benchmark dependencies.

The released source samples inverse-distance-rank pairs, alternates CS/LS, and
uses population order for local selection. Here the supplied paper takes
precedence: choose the farthest pair, draw CS/LS with equal probability, and rank
LS parents by mean cost. Other deliberate differences: sequential snapshot
generations, explicit failure budgets independent of a profiler, no padding of
mismatched vectors, no rejection solely for equal score vectors, and no bundled
Python function execution. Hand-designed initialization and benchmark drivers
are not included. These are documented adaptations, not a bit-for-bit port or
a reproduction of the paper's reported performance.

## Verification

From the repository root, using the existing environment:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_eoh_s
../slick/.venv/bin/ruff check eoh_s tests/test_eoh_s.py
../slick/.venv/bin/ruff format --check eoh_s tests/test_eoh_s.py
```

From this folder, `python -m pip install -r requirements.txt` installs the adjacent
Slick checkout. Offline tests use the repository's shared scripted provider and
verify numerical selection, both operators, snapshot semantics, partial budgets,
rejections, error propagation, and templates from another launch directory.
No paid generation or optimization benchmark is claimed by these tests.

During implementation, greedy CPM also matched the pinned official
`survival_set` on 100 seeded random candidate pools after negating costs to match
the upstream reward convention. This compared the actual upstream method and
selected IDs, including ties; it does not establish end-to-end search equivalence.
