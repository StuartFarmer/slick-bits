# Darwin Gödel Machine

`DGM` evolves **executable agents** for a caller-defined task family. It samples
a parent from an archive, diagnoses a general improvement, runs that parent's
own code to implement the change, evaluates the child, and archives every
functional child. A worse child can become a useful stepping stone later.

The algorithm is adapted from the paper's [official implementation](https://github.com/jennyzzt/dgm).
See [SOURCES.md](SOURCES.md) for the pinned revision, source mapping, and differences.

## Use

Use the existing Slick environment from the repository root. Configure the local
template root once at application startup; importing `dgm` does not change it.

```python
from pathlib import Path

import dgm
from slick import prompts

prompts.TEMPLATE_ROOT = Path(dgm.__file__).resolve().parent / "prompts"


async def optimize(task, provider, execute, evaluate):
    search = dgm.DGM(task, provider, evaluate, execute=execute, seed=42)
    result = await search.run(
        config=dgm.Config(iterations=80, batch_size=2, max_generations=16)
    )
    return result.best.source, result
```

`task` describes your task family and runtime constraints. `provider` is a Slick
provider used for diagnosis and the frozen model's generation endpoint.
The implementation needs two application adapters:

- **`await execute(source, instruction, generate) -> str`** runs the supplied
  Python source in your isolated worker, then awaits its
  `agent(instruction, generate)` entry point. Expose the async `generate(str) -> str`
  callback through RPC. Return the agent's final string. During modification,
  that string must be the full replacement agent source. Execute the *supplied*
  source on every call: routing every request through a fixed agent would remove
  recursive self-improvement. The adapter owns timeouts, resource limits, cleanup,
  runtime dependencies, and any additional isolated tools.
- **`await evaluate(source) -> Evaluation`** runs the same supplied agent on your
  downstream tasks, returning `Evaluation(score, can_self_modify, feedback)`.
  Use a consistent higher-is-better score in `[0, 1]`; normalize other metrics in
  this adapter. `can_self_modify` must reflect a functional source-editing probe,
  not just successful compilation or a model's claim. Feedback contains training
  evaluation logs or summaries for the next diagnosis. Keep held-out answers out
  of that feedback. The evaluator owns its execution isolation, model-call
  budgets, task splits, caching, and optional staged evaluation.

The built-in `SEED_AGENT` is intentionally minimal:

```python
async def agent(instruction, generate):
    return await generate(instruction)
```

Supply `run(initial=your_source, ...)` to start from an existing agent with the
same entry point. Its source may define additional functions, prompts, and
workflows. The Python implementation is what evolves; downstream tasks can be
writing, reasoning, planning, coding, or any other task your evaluator measures.
The archive controller and evaluator remain outside the evolving source.

For downstream execution, a model endpoint can be built from the same external
template without sharing mutable sessions:

```python
async def generate(instruction):
    return await search.generate(instruction, provider=provider)

answer = await execute(result.best.source, "Your next task", generate)
```

Here `search` and `result` are the objects from your application. This downstream
endpoint needs its own caller-enforced budget; the search budget applies only to
self-modification. Separate providers for downstream evaluation are supported by
the evaluator adapter, as in the original experiments.

## Search and failure policy

Parent weights are `sigmoid(10 * (score - 0.5)) / (1 + archived_children)`.
Sampling is with replacement. Every admitted candidate has positive probability,
including zero-scoring candidates. Only admitted children affect child counts.

`iterations` counts attempted children, including rejected ones. Parents for each
batch are sampled before executing any child. Completed children are saved
immediately, so a later worker failure cannot lose them; they become selectable
in the next batch. Execution is serial in this implementation. A final partial
batch respects the exact attempt budget. One fresh `DGM` instance owns one run.

Each attempt makes one diagnosis call and allows at most `max_generations` model
calls by the executing parent. There are no implicit algorithm retries or shared
sessions. Transport retries, if configured, belong to the provider adapter.
The executor must terminate stuck or excessive generated computation even when
it makes no model calls.

Compilation is checked without executing the source. Blank, unchanged,
syntactically invalid, or nonfunctional children are rejected; score decreases
are retained. Invalid diagnosis JSON and `ExecutionError` consume an attempt.
Workers should raise `ExecutionError` for generated-program failures/timeouts,
and evaluators can raise `CandidateRejected` for an invalid candidate. A bad
initial agent aborts the run. Infrastructure/programming errors and cancellation
propagate; completed records remain available on `search.result`.

`result.archive` stores immutable source snapshots and parent IDs;
`result.attempts` records proposals, accepted/rejected children, modifier IDs, and
errors. `result.generations` retains raw prompts/responses before parsing or
validation, including the executor's final modification output. Records marked
`modify` are executable-agent outputs, **not additional foundation-model calls**.
`evaluation_calls` counts evaluator invocations, including the seed, but not the
individual tasks or model calls inside an evaluator. Persist these objects in
your application if needed. `result.best` returns the highest-scoring admitted
agent, keeping the earlier agent on ties.

`Config(mode="no_self_improve")` runs the fixed seed as the modifier while still
selecting and editing archived parents. `Config(mode="no_open_ended")` always
modifies the latest admitted agent, falling back to it after rejected children.
That ablation always uses one attempt per batch regardless of `batch_size`.
Historical records remain available, but are not selectable in that ablation.

## Verification

```sh
rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_dgm
```

Tests use the shared scripted provider and a scripted execution adapter. They
exercise recursive source routing, stepping stones, selection, ablations, budgets,
rejections, and prompt loading. They do not execute generated source, certify an
execution sandbox, demonstrate measured improvement, or reproduce the paper's
benchmark results. No new dependencies or paid model calls are needed.
