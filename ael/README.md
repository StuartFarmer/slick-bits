# AEL

`AEL` evolves arbitrary textual candidates using population selection, crossover,
mutation, and elitism. Task content and an async evaluator come from the caller.

```python
from pathlib import Path
from slick import prompts
from ael import AEL

prompts.TEMPLATE_ROOT = Path("ael/prompts").resolve()
agent = AEL(task="Your task and candidate requirements", provider=provider,
            evaluate=evaluate)
population = await agent.run(population_size=4, parents=2, generations=3)
best = population[0]
```

`evaluate(content: str) -> float` is async and must return finite fitness. Lower
fitness wins unless `maximize=True`. `Proposal` contains nonblank `description`
and `content`; ranked `Individual` results also contain `id` and `fitness`.

`run` exposes population size, generations, parent count, offspring count,
crossover/mutation probabilities, selection seed and `init_attempts`. By default,
initialization stops after three attempts per slot. Each generation samples only
its starting population; offspring compete with incumbents using stable ties.
Clones without variation incur no model call or reevaluation. Mutation operates
on the crossover result when both fire. Proposal schema validation failures and
evaluator `ValueError`/`TimeoutError` consume attempts. Provider errors, including
timeouts and ordinary `ValueError`, propagate immediately; other unexpected errors
also abort.
`agent.attempts` retains proposals, parent IDs, scores and errors in memory;
`agent.history` holds immutable population snapshots. Each run resets both.

Pass `session=` to use a caller-owned Slick Session; otherwise calls use the
constructor provider. Calls are sequential. Configure the process-global template
root once before running; separate concurrent applications with different roots
need separate processes. An agent instance supports one run at a time.

The caller owns evaluation, any required execution isolation, provider setup and
persistence. This agent never executes candidates. The previous domain-specific
application is preserved in `examples/legacy/2026-09-14-problem-specific.tar.gz`.
From this directory, install the existing local Slick dependency with
`python -m pip install -r requirements.txt`. From the repository root, run
`python -m unittest tests.test_ael` for offline verification.

`run` delegates initialization, offspring generation, assessment and survivor
selection to named phases. The three prompt methods own separate templates:
`initialization`, `crossover`, and `mutation`.

Caller inputs follow the annotated API without blanket runtime type checks.
Generated proposals retain their schema constraints, and evaluation requires finite scores.
