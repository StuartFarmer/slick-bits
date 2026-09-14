# Optimizing the Optimizer

Implements the paper's code-improvement dialogue with Slick and its CMSA
construction heuristics with ordinary Python. Both interfaces accept caller-owned
domain logic. [SOURCES.md](SOURCES.md) records the official implementation used,
formula discrepancies, licensing, and adaptations.

## Improve an existing optimizer

Supply the complete implementation, the target component, a task description,
a provider, and an async evaluator. The source can be in any language.

```python
from pathlib import Path
from slick import prompts
import optimizing_the_optimizer
from optimizing_the_optimizer import Evaluation, OptimizingTheOptimizer

# Configure once at application startup; independent of the launch directory.
prompts.TEMPLATE_ROOT = (
    Path(optimizing_the_optimizer.__file__).resolve().parent / "prompts"
)

async def evaluate(source: str) -> Evaluation:
    # Your service owns compilation, interface checks, isolated execution,
    # fixed evaluation instances/seeds, and resource limits.
    report = await evaluation_service.evaluate(source)
    return Evaluation(report.score if report.valid else None, report.feedback)

agent = OptimizingTheOptimizer(
    task="Describe your optimization problem, language, interface and scoring contract.",
    provider=provider,
    evaluate=evaluate,
    maximize=True,
)
best = await agent.run(
    Path("existing_optimizer.py").read_text(),
    target="generate_solution",
    rounds=2,
    performance=True,
)
print(best.code, best.score)
```

`evaluation_service` and `provider` above are application dependencies. No provider
is constructed and no candidate source is executed by this library.

The baseline is evaluated first. The first round discovers a heuristic using
underutilized algorithm state; later rounds refine the previous proposal using
evaluation feedback. Each valid heuristic optionally gets a separate implementation
efficiency proposal. These performance branches do not replace the heuristic
parent for the next round. All valid versions compete with the baseline for the
returned `Candidate(code, score, feedback)`; ties keep the earlier incumbent.
The next heuristic can build on a worse candidate, as in the paper's V1-to-V2
dialogue, while the best measured version remains available separately.

The three operations have separate decorated methods and local templates. Generated
JSON has `code` and `rationale`; postprocessing preserves source whitespace while
rejecting blank code. `Evaluation(None, feedback)` rejects a candidate; nonfinite
scores are also rejected. Invalid generation consumes its round and its raw output
and error are included in the next revision's dialogue. A performance proposal
gets one attempt; failures are recorded without an automatic repair loop. An
invalid baseline raises. Provider failures and evaluator exceptions propagate:
report expected compile/test failures using `Evaluation(None, feedback)`.

`rounds=R` makes R heuristic calls, plus at most R performance calls when enabled.
There are no hidden retries. `agent.attempts` records raw responses, source parents,
operations, errors and measurements; `agent.evaluations` includes the baseline and
every evaluator invocation. Both reset on each run. The complete dialogue is
explicitly rendered from these records; there is no implicit Session history or
tool execution. Keep rounds modest for large implementations, since full dialogue
uses context proportional to all previous proposals. A provider's transport retry
policy remains its caller's responsibility.

Use one instance for one run at a time. Configure Slick's process-global template
root before generation; applications needing different roots concurrently should
use separate processes. The supplied task and evaluator define allowed edits and
correctness. Prompt requests to preserve interfaces or behavior are not guarantees;
the evaluator must check them, including equivalence for performance proposals.

## Run the CMSA heuristics directly

`CMSA` implements the actual construct/merge/solve/adapt loop without model calls.
It supports finite component-subset formulations with feasible greedy construction:
components can represent assignments, selected items, edges, facilities, or any
other hashable identifiers. It does not require graph vertices.

```python
from optimizing_the_optimizer import CMSA

optimizer = CMSA(
    components=component_ids,           # stable sequence of unique hashable IDs
    costs=heuristic_costs,               # nonnegative cost per ID; smaller is preferred
    can_add=can_add,                     # (frozenset of chosen IDs, next ID) -> bool
    solve=solve_restricted,              # async (frozenset of allowed IDs, seconds)
    evaluate=score_solution,             # async (frozenset of selected IDs) -> float
    variant="v1",                       # "baseline", "v1", or "v2"
    maximize=True,
)
solution = await optimizer.run(
    iterations=100, constructions=10, age_max=10,
    determinism_rate=0.8, candidate_list_size=5,
    time_limit=60.0, solve_time_limit=2.0, seed=42,
)
```

The application supplies the variables and functions in this example:

| Input | Responsibility |
| --- | --- |
| `costs` | Static heuristic cost; use full graph degree to recover the MIS rule. Score direction does not change these costs. |
| `can_add(chosen, c)` | Decide whether adding an unused component is feasible. Construction stops when no component can be added. Every terminal construction must be a complete valid solution. |
| `solve(pool, seconds)` | Solve using only components in the merged pool. Return a feasible `frozenset`, including an empty one if valid, or `None` when no incumbent was found. Use an exact solver with a time limit to match CMSA; heuristic solvers are possible but change the method. |
| `evaluate(solution)` | Validate the complete constructed/solver solution and return its finite objective. Exceptions abort. |

All constructors start from the full component universe. Components are sorted
once by cost, with ties retaining caller order. Deterministic choice takes the
first feasible component. The baseline random branch samples the first k feasible
components; V1 samples all feasible components with weights
`1/(2 + age) + 1/(1 + cost)`. V2 normalizes those weights to `p`, computes
`H = -sum(p*log(p))`, and samples using `(p + H)/(1 + n*H)`.

Age -1 means absent from the pool. Construction merges a newly selected component
immediately by setting its age to zero; selecting an existing component leaves its
age unchanged. After a solver incumbent, all active ages increase, its selected
components reset to zero, and ages reaching `age_max` expire to -1. With `None`
from the solver, adaptation is skipped. The best constructed or solved solution
is retained independently of the pool, with stable ties.

`Solution.components` and `Solution.score` are returned, or `None` if the budget
ends before any complete solution is evaluated. `optimizer.age` exposes final
ages; `optimizer.history` records each completed iteration's pre-adaptation pool,
solver result, post-adaptation age snapshot, construction count and best solution.
State and RNG reset each run. A time limit is cooperative wall time checked between
constructions and before solving; each construction/evaluation can finish after
the deadline. The solver receives the smaller of its limit and remaining time
and must enforce that limit itself. No worker cancellation or execution service
is hidden inside the optimizer.

## Verification

From this folder, `python -m pip install -r requirements.txt` installs the adjacent
Slick checkout. From the repository root:

```sh
python -m unittest tests.test_optimizing_the_optimizer
```

The offline checks use the shared scripted provider and a tiny exact capacity
problem. They exercise dialogue, rejection and measurement accounting, all prompt
templates, probability formulas, seeded variants, pool aging and solver boundaries.
These are algorithm/contract checks, not reproduction of the paper's GPT-4o,
CPLEX or graph-benchmark results. No claim of improved optimization performance
is made without the caller's measurements.
