# Adaptive continuation

Problem-agnostic implementation of the controller algorithm in **Large Language
Models as Optimization Controllers: Adaptive Continuation for SIMP Topology
Optimization**, Yang, Wang and Wang (2026), [arXiv:2603.25099](https://arxiv.org/abs/2603.25099).

The agent controls an existing iterative optimizer. Supply the task, numeric
parameter definitions, an async solver step, and a measured initial checkpoint.
The agent does not generate candidate solutions or implement a physics solver.

```python
from pathlib import Path

from slick import prompts
from adaptive_continuation import AdaptiveContinuation, Measurement, Parameter
import adaptive_continuation

# Configure once at application startup; Slick's template root is process-global.
prompts.TEMPLATE_ROOT = Path(adaptive_continuation.__file__).parent / "prompts"

async def step(x, parameters):
    x = x - parameters["learning_rate"] * 2 * (x - 3)
    return Measurement(state=x, objective=(x - 3) ** 2, metrics={})

def schedule(iteration, budget, settings):
    return {"learning_rate": 0.1 * (1 - iteration / budget) + 0.01}

async def optimize(provider, mode="llm"):
    agent = AdaptiveContinuation(
        task="Minimize (x - 3)^2 using gradient descent; control learning_rate.",
        provider=provider,
        evaluate=step,
        parameters={"learning_rate": Parameter(0.001, 0.45, 0.1)},
        schedule=schedule,
    )
    return await agent.run(
        Measurement(0.0, 9.0, {}),
        iterations=30,
        settings={"call_every": 5},
        mode=mode,
        tail={"learning_rate": 0.05},
        tail_iterations=10,
    )
```

Run `asyncio.run(optimize(your_slick_provider))` with any configured Slick provider.
`asyncio.run(optimize(None, mode="schedule"))` exercises the numerical plumbing
without an API call. Provider creation, model/version selection, temperature,
transport retry limits, and execution isolation belong to the application.
The example is a small mathematical demonstration, not a paper reproduction.

## Solver contract

- `evaluate(state, parameters)` advances **one** optimizer iteration and returns
  `Measurement(state, objective, metrics, feasible)`. Evaluate the returned state
  under those parameters; do not attach a pre-update objective to a post-update
  state. Include the variables needed to resume the solver in `state`, including
  optimizer/RNG state where appropriate. For a remote solver, state can instead
  be an immutable checkpoint handle; the callback must restore it before stepping.
- The supplied initial measurement must use each parameter's `initial` value.
  The agent counts only its own step calls, not this initial external evaluation.
  Snapshot state is deep-copied; mutating the active state cannot corrupt it.
- Objectives and reported scalar metrics must be finite. Feasibility is reported
  by the evaluator; the optional `valid(measurement, parameters)` predicate adds
  snapshot eligibility requirements. Lower objectives win unless `maximize=True`.
  Objective values must be meaningful to compare across iterations.
- `Gate(metric, parameter, threshold, cap)` caps a parameter whenever the metric
  is **strictly greater** than the threshold. `threshold_key` optionally reads
  a tunable threshold from settings. Bounds, optional monotonicity, and gates
  apply in that order, with gates taking priority. Supply mutually compatible
  bounds/gates. No SIMP names or geometry are required by the generic agent.

## Control and failure behavior

The live model observes the current/best objectives, signed one/five-step relative
changes, a recent secant slope, stagnation, iterations since best, budget fraction,
custom metrics, current controls, settings, and hard gates. The five-step signal
uses the available shorter window at startup. The paper's compliance-derived
grayness proxy is exposed as `progress_proxy_slope`; it is not a measured metric
derivative. Stagnation counts steps without improvement in the best valid snapshot.

`control.j2` requests all parameter values, `restart`, and a decision note. Slick
parses an explicit typed output; Python rejects unknown/missing names, nonfinite
values, numeric strings, and Boolean numeric values. Finite out-of-range actions
are clamped. Restart requests are ignored until a valid snapshot exists. A restart
restores the state, retains the new control values, and rechecks the gates.

One model call follows each `call_every` completed iterations, **only when another
main iteration remains**. Parameters otherwise persist. Gates are also checked
before every main step, including between calls. No model history/session or
hidden generation retries are used. Generation/parse/semantic failures and
`ProviderError`/`OSError` use the supplied deterministic schedule; without one they
hold existing parameters. Custom provider adapters should wrap transport errors
as `ProviderError`. Programming errors and solver errors propagate. Cancellation
propagates. Failed solver attempts are counted on the agent's evaluation counters.

`calls` retains rendered prompts, raw responses (including invalid JSON), errors,
observations, requested actions, and applied actions. `Result.fallbacks` and
`primary_eligible` make degraded runs identifiable. `history` records measured
iterations and applied parameters; it does not copy every solver state. `best`
is the best valid **main-loop** snapshot; `final` is the actual final measurement.
The raw record distinguishes a model request from any deterministic fallback.

The optional fixed finishing phase restores `best`, or `initial` if there was no
valid snapshot. It runs exactly `tail_iterations` with the supplied `tail` values,
deliberately bypassing adaptive gates/monotonicity. It does not overwrite `best`.
`mode="fixed"` holds initial parameters and receives no tail. `mode="schedule"`
calls the supplied schedule before each step, without LLM decisions or state gates.

## SIMP preset

`simp.py` supplies Tables 2–3 bounds/defaults, the hard grayness gate, snapshot
validity (`penal >= 3` and `grayness < 0.25`), Equation (7)'s `grayness`, a fallback
schedule, Table 1's schedule-only ablation, and the fixed tail. The solver should
report `grayness`, `checkerboard`, and `volume_fraction` in `metrics`, and volume
constraint satisfaction in `feasible`. All density filtering, projection, FEA,
sensitivity propagation, OC updates, and mesh handling remain in that solver.

```python
from adaptive_continuation.simp import (
    GATES, PARAMETERS, SETTINGS, TAIL, TUNABLES, fallback, valid_snapshot,
)

def simp_agent(task, provider, solver_step):
    return AdaptiveContinuation(
        task, provider, solver_step,
        parameters=PARAMETERS, gates=GATES, valid=valid_snapshot,
        schedule=fallback,
        guidance=(prompts.TEMPLATE_ROOT / "simp_guidance.j2").read_text(),
    )

# agent = simp_agent(task, provider, solver_step)
# result = await agent.run(initial, settings=SETTINGS, tail=TAIL)
# -> 300 main steps, 59 model attempts at k=5, 40 tail steps.
```

For the paper's fixed baseline, use `dataclasses.replace` to change the `penal`
parameter's initial value to 3, then `mode="fixed"`. For the schedule ablation,
supply `schedule_only` and use `mode="schedule"` with `tail=TAIL`.

## Between-run tuning

`await agent.run_meta(compare, rounds=5, settings=SETTINGS, tunables=TUNABLES)`
runs `await compare(settings)` sequentially. Supply a callback that initializes
fresh solvers, runs the desired controllers, and returns a JSON-serializable
summary including objectives, feasibility, runtime and fallback counts. It may
cycle geometries, arbitrary tasks, or seeds. Include baseline results explicitly.
`reflect.j2` sees completed comparisons and proposes a partial configuration
update. Values are clamped to tunable bounds; integer fields are rounded.
Invalid updates retain the previous configuration and log the failure. Reflection
runs between comparisons only, so five comparisons use four meta calls.

The returned dictionary contains comparisons, reflection records, and the final
settings. Persist it using your application's existing storage. Configuration
updates replace the paper's regex source patching; no generated code is executed.

## Provenance and limits

Checked 2026-09-14: the [paper's code availability statement](https://arxiv.org/html/2603.25099v1)
still promises a future release and provides no official implementation URL.
Title/repository searches and the [author's public profile](https://github.com/nbbllxx0)
did not identify an official repository for this controller. A related official
[GPU solver repository](https://github.com/nbbllxx0/Fused-Gather-GEMM-Scatter-Kernels)
accompanies a different paper, arXiv:2604.18020; it is not used as this controller's
reference implementation. No upstream code is vendored or claimed as reused.

This implementation uses the supplied paper as the algorithm specification.
The prompts are reconstructed and deliberately generalized, not the unpublished
verbatim prompts. Absolute config updates replace source patches. Exact fallback
interpolation and phase-minimum semantics are unspecified; `simp.fallback`
documents the chosen interpretation. `schedule_only` follows Table 1's budget
fractions, rather than the conflicting claim that beta reaches 32 by iteration 80
of a 300-step run. The objective slope uses a recent secant. Gates are checked on
each applied step and after restores; the terminal no-op model call is omitted.
Thresholds are exposed to the model. These are intentional implementation choices.

The repository contains the adaptive algorithm and fixed/schedule comparisons,
not the original FEA benchmark suite, expert heuristic, or three-field baseline.
Neither exact compliance results nor generalization to new objectives is claimed.
No new dependencies beyond the existing Slick/Pydantic/Jinja installation.

## Checks

From the repository root, using the adjacent Slick environment:

```sh
../slick/.venv/bin/python -m unittest tests.test_adaptive_continuation
../slick/.venv/bin/ruff check adaptive_continuation tests/test_adaptive_continuation.py
```

Tests use the shared `tests/providers.py` provider with real Slick parsing and
external async steps. They check control timing, guards, restart/tail checkpoints,
failure accounting, bounded meta updates, and both templates from another working
directory. These checks establish algorithm/interface behavior, not LLM quality.
