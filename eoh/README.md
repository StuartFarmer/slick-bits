# EoH

`EoH` evolves arbitrary textual ideas and candidates through five operators:
diverse exploration (E1), recombination of a shared idea (E2), structural revision
(M1), setting changes (M2), and simplification (M3).

```python
from pathlib import Path
from slick import prompts
from eoh import EoH

prompts.TEMPLATE_ROOT = Path("eoh/prompts").resolve()
agent = EoH(task="Your task and candidate requirements", provider=provider,
            evaluate=evaluate)
population = await agent.run(population_size=4, parents=2, generations=3)
best = population[0]
```

`evaluate(content: str) -> float` is async and must return finite fitness. Higher
fitness wins unless `maximize=False`. `Proposal` contains nonblank `description`
and `content`; ranked `Individual` results also contain `id` and `fitness`.

`run` exposes population size, generations, parent count, seed, operator subset
and initialization attempts. Initialization uses INIT and defaults to at most
three attempts per slot. Each generation makes exactly N attempts per selected
operator. Exploration chooses the requested number of parents; modifications
choose one. Sampling is without replacement, weighted by `1/(one_based_rank + N)`,
from the population at the start of the generation. Offspring compete with
incumbents; stable ties retain incumbents. Invalid JSON, invalid/nonfinite scores,
provider errors and timeouts consume their attempt without shrinking the
population. Unexpected exceptions abort.

`agent.attempts` retains proposals, parent IDs, scores and failures in memory;
`agent.history` holds immutable population snapshots. Each run resets both.
`run(session=...)` uses a caller-owned Slick Session; otherwise it uses the
constructor provider. Calls are sequential; an instance supports one run at a
time. Configure Slick's process-global template root before running. Applications
with different roots need separate processes when running concurrently.

The caller owns evaluation, any required execution isolation, provider setup and
persistence. This agent never executes candidates. The previous domain-specific
application is in `examples/legacy/2026-09-14-problem-specific.tar.gz`.
From this directory, `python -m pip install -r requirements.txt` installs the
existing local Slick dependency. From the root, run
`python -m unittest tests.test_eoh` for offline verification.

`run` delegates initialization, parent selection, offspring generation and
survivor selection to named phases. INIT, E1, E2, M1, M2 and M3 each have a
dedicated prompt method and template.

Caller inputs follow the annotated API without blanket runtime type checks.
Generated proposals retain their schema constraints, and evaluation requires finite scores.
