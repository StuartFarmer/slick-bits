# EvoX

Evolve arbitrary text artifacts **and the Python search strategies that generate
them**. Supply a task description and an async evaluator; candidates can be
prompts, programs, plans, configurations, or another text representation.

```python
from pathlib import Path
from slick import prompts
import evox
from evox import Config, Evaluation, EvoX
from evox.runtime import run_python_strategy

# Configure once at application startup, before any prompt calls.
prompts.TEMPLATE_ROOT = Path(evox.__file__).resolve().parent / "prompts"

async def evaluate(candidate: str) -> Evaluation:
    score, feedback = await your_evaluator(candidate)
    return Evaluation(score, artifacts={"feedback": feedback})

agent = EvoX(
    task="Describe the artifact, objective, constraints, and required format here.",
    provider=provider,
    evaluate=evaluate,
    run_strategy=run_python_strategy,
    config=Config(iterations=100, window=10, maximize=True, seed=42),
    # Optional: already evaluated seeds, outside the new evaluation budget.
    initial_population=[("your seed artifact", Evaluation(0.5))],
    # Optional: strategy_provider=meta_provider, operator_provider=cheap_provider,
)
result = await agent.run()
if result.best is not None:
    print(result.best.text, result.best.score)
```

Install/use Slick 0.3.0 with Pydantic 2 and Jinja 3. The repository's
`optimizer/.venv/bin/python` already imports the adjacent Slick checkout. There
are no additional dependencies. Use a fresh agent per run. Imports do not change
Slick's process-global template root; configure it once using the module path
above, which works independently of the launch directory.

`evaluate` owns task-specific validation, datasets, and any candidate execution.
Return a finite score and JSON-compatible artifacts; include component metrics,
logs, or critiques when they can inform search. Set `maximize=False` for costs.
Raw scores remain in results; strategies also receive `quality`, which negates
scores for minimization so higher always wins. Candidate text is preserved
exactly, including whitespace.

`run_strategy(code, population, state, seed)` is an async execution callback
returning a selection dictionary. The bundled runner executes Python in a fresh
subprocess with a five-second timeout, temporary working directory, and empty
environment. It checks for mutation of the supplied population/state. **This is
process isolation, not a security sandbox:** generated code still has local
filesystem access. For untrusted workloads, provide a container or remote sandbox
with this same callback signature. The optimizer itself never executes generated
code, and the strategy worker never executes candidate text. The task evaluator
has its own independent execution boundary.

The strategy interface is:

```python
def select(population, state, rng):
    parent = max(population, key=lambda candidate: candidate["quality"])
    return {
        "parent_id": parent["id"],
        "operator": "refine",  # "free", "refine", or "diverge"
        "inspiration_ids": [],
    }
```

Selection programs receive all accepted candidates, their artifacts, population
statistics, parent reuse counts, recent outcomes, and a seeded `random.Random`.
They may invent selection logic rather than choosing from fixed algorithm names.
Each invocation starts fresh; derive strategy state from the provided database
and descriptor. Supply `initial_strategy=source` to replace the uniform seed
strategy; custom seeds undergo the same validation as generated strategies.

The run prepares task-specific refinement and structural-variation instructions,
then repeats Algorithm 1's three phases:

1. Generate/evaluate solutions under the current strategy for one window.
2. Record score improvement and the strategy's before/after population states.
3. On stagnation, select a score-biased parent strategy and inspirations from
   successful/similar-state deployments, mutate its code, validate it, and deploy
   it while retaining the entire solution population.

The default window is `max(1, iterations // 10)`. The last window is truncated to
the remaining budget. `stagnation_threshold` defaults to `1e-6`; a switch occurs
only when the window's best-quality improvement is strictly below that threshold.
No strategy is generated after the solution budget is exhausted.

`iterations` counts newly generated solution attempts, including blank output,
nonfinite measurements, and evaluator `ValueError` rejections. `evaluation_calls`
counts actual evaluator invocations, so blank artifacts consume a step without
calling the evaluator. Pre-evaluated seeds consume neither budget. With an empty
database, initialization repeats until an artifact is accepted or the budget
ends; the first accepted score establishes the first window's baseline. If no
candidate is accepted, `best` is `None`.

Operator preparation uses one separate model call. Each stagnation update permits
`strategy_attempts` model calls (default three), with rejection feedback supplied
to the next attempt. JSON parsing, Python syntax/interface checks, singleton and
current-population execution, and selection validation must pass before deployment.
Invalid IDs, operators, duplicate inspirations, parent-as-inspiration, and excess
inspirations are rejected. These checks establish interface validity, not fitness.
If all attempts fail, the active strategy stays in place. A deployed strategy
that later fails selection falls back to uniform selection for that step; the
failure is recorded and its window still counts against that strategy.

Malformed operator preparation and provider failures abort. Evaluator
`ValueError` is an intentional rejection; other evaluator errors and cancellation
propagate. The strategy callback reports rejected executions as `ValueError`;
other callback errors propagate. Transport retries belong to the provider.
All calls are stateless provider calls, with no shared conversational session.

Results retain all candidates (including rejections), accepted strategy source,
strategy attempts/errors, selection failures, window histories, and raw prompts
and responses captured before structured parsing. Persist `dataclasses.asdict(result)`
if desired. Results can grow with the budget and artifact size.

The default strategy reward follows official SkyDiscover code:
`Δ * (1 + log1p(max(0, start_quality))) / sqrt(actual_window_steps)`.
It remains defined for zero and negative scores. The paper's Equation 2 omits
the additive `1` and clamp. For an exact Eq. 2 experiment with nonnegative,
higher-is-better scores, pass
`score_window=lambda start, end, n: (end-start) * math.log1p(start) / math.sqrt(n)`.
This changes strategy ranking, not the downstream objective.

See [SOURCES.md](SOURCES.md) for pinned official implementation references and
the deliberate differences between this implementation, the paper, and upstream.
This is an algorithm implementation, not a reproduction of reported benchmark scores.

Run offline checks from the repository root:

```sh
optimizer/.venv/bin/python -m unittest tests.test_evox
../slick/.venv/bin/ruff check evox tests/test_evox.py
../slick/.venv/bin/ruff format --check evox tests/test_evox.py
```
