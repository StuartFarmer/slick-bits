# Zero-shot parent selection

Problem-independent Slick implementation of **Benchmarking Zero-Shot LLM-Generated
Parent Selection in Genetic Programming for Symbolic Regression**, by Hengzhe
Zhang, Qi Chen, Bing Xue, Wolfgang Banzhaf, and Mengjie Zhang.

There are two usable pieces:

- `ZeroShotSelection`: generate independent selection operators from one task
  description, reject functionally invalid outputs, then measure the valid batch.
- `kimi_selection` and `gpt_selection`: NumPy implementations of the representative
  rules in Algorithms 1 and 2, returning indices into any population.

The method synthesizes **parent-selection code**, not candidate solutions to the
task. Your existing search owns initialization, variation, fitness, and survival.
It supplies candidate measurements and uses the returned parent indices or objects.
No GP trees, regression datasets, model provider, or execution service are required
by the numerical selectors.

## Use the selectors directly

This is a complete example with arbitrary candidate labels and two fitness cases:

```python
import numpy as np
from zero_shot_selection import kimi_selection, gpt_selection

population = ["schedule A", "schedule B", "schedule C"]
residuals = np.array([[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]])
sizes = np.array([3.0, 1.0, 2.0])  # Any meaningful complexity/cost measure.

# Example preprocessing choice: column L2 normalization; variance of raw residuals.
# The abbreviated paper does not specify these transformations completely.
normalized = residuals / (np.linalg.norm(residuals, axis=0) + 1e-12)
indices = kimi_selection(
    normalized, residuals.var(axis=0), sizes, k=6, stage=0.5,
    rng=np.random.default_rng(42),
)
parents = [population[i] for i in indices]
assert len(parents) == 6

# Here behavior is measured output, not necessarily residuals. Unit-length rows
# make the dot products cosine similarities; this preprocessing is the caller's choice.
behavior = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
indices = gpt_selection(
    behavior, (residuals ** 2).mean(axis=1), sizes, heights=np.ones(3),
    k=6, stage=0.5, rng=np.random.default_rng(42),
)
assert len(indices) == 6
```

Rows are candidates and columns are shared cases or behavior dimensions. Kimi
receives normalized errors `R_n`, a per-case variance vector, and per-candidate
sizes. GPT receives behavior matrix `P`, mean errors, sizes, and heights. Lower
errors and complexity are preferred. Constant sizes/heights disable their relative
parsimony effects. A single scalar objective can be represented by one case; it
contains less behavioral information than a multi-case evaluation.

Supply finite, compatible arrays; use a nonempty population and nonempty case
axis when `k > 0`, a nonnegative integer `k`, and `stage` in `[0, 1]`. Use positive
`epsilon` and `0 < elite_downweight <= 1`. Inputs are trusted and never mutated.
`k=0` returns an empty integer array, including for an empty population. Sampling
allows repeated parents and `k` larger than the population. Pass a seeded NumPy
Generator for reproducibility; no global RNG state is used. Typical stage scheduling
is `generation / (generations - 1)` for multiple selection rounds.

Kimi retains the curriculum, standardized fitness/novelty blend, min-max size
penalty, and tournament width growing from 2 to 8. Its matrix work is O(nm), plus
O(kτ) tournament draws. GPT retains the clipped behavior similarities, stage
weights, elite fraction, temperature, and crowding discount. It explicitly uses
O(n²) similarity memory and O(n²d) work. GPT returns descending-score elites first,
followed by sampled indices; elite ties retain population order. Kimi breaks
tournament ties by the first drawn contender.

## Synthesize operators for your own task

Use `optimizer/.venv`, which has Slick and NumPy installed. Configure templates
once at application startup, before rendering or generation:

```python
from pathlib import Path
from slick import prompts
import zero_shot_selection

prompts.TEMPLATE_ROOT = Path(zero_shot_selection.__file__).resolve().parent / "prompts"

async def synthesize_for_task(task, provider, validate, evaluate=None):
    agent = zero_shot_selection.ZeroShotSelection(
        task, provider, validate, evaluate, validation_timeout=10.0,
    )
    return await agent.run(count=10, max_attempts=100)
```

The task should describe your candidates, their available measurements, score
direction, and relevant constraints. For example:

```text
Select parent schedules for a genetic algorithm minimizing lateness and schedule
complexity. Each population member has .case_values, a NumPy vector of nonnegative
lateness costs on the same delivery scenarios, and .size, its number of assignments.
```

The generated module defines
`custom_selection(population, k=100, status={})`, returning exactly `k` original
population objects with repetition allowed. `status["evolutionary_stage"]` is the
normalized stage. Population members and status must not be mutated. NumPy is
available to the generated code only if **your execution environment** provides it.
This interface does not require regression-specific `.y` or `.predicted_values`.

`validate(source) -> None` is a required async callback. It owns **isolated** code
execution and functional tests. Check imports/execution, exact output length,
membership by identity, default/keyword calling conventions, input non-mutation,
singleton and tied populations,
different stages, and counts including zero and larger than the population. Test
synthetic interface cases, not task fitness or held-out data. Convert candidate
runtime errors or failed checks to `CandidateRejected`; let infrastructure errors
propagate. A return value is not a verdict: success means returning normally,
and rejection means raising `CandidateRejected`.

The agent wraps validation in an asyncio timeout. Your callback must be
nonblocking, respond to cancellation, terminate its workers, and enforce execution
resource limits. An asyncio timeout cannot interrupt synchronous `exec`, kill
orphan processes, or establish a security sandbox. This package never executes or
imports generated source. AST compilation checks syntax and the declared entry
point only; it cannot prove the actual runtime interface or safety.

`evaluate(source) -> Mapping[str, float]` is an optional async callback that runs
your fixed evolutionary search with that selector and returns named measurements,
such as `train/problem-a/seed-0` and `test/problem-a/seed-0`. It owns execution
isolation, benchmark deadlines, dataset splits, seeds, and search budgets. Use the
same initialization, variation, fitness, top-1 survival, and budgets for each
selector and baseline. In the paper these were 200 individuals, 100 generations,
0.9 crossover, 0.1 mutation, 12 datasets and 5 split seeds. Keep held-out data out
of selection and training. The generic package does not implement these
problem-specific operators or aggregate away the per-run measurements.

## Zero-shot and failure contracts

Every attempt receives the same task-only prompt via an independent provider
call. No previous code, failure text, reference selector, or benchmark result is
included. Do not use a provider that injects conversation history. The bundled
Kimi/GPT selectors are utilities for callers, never synthesis examples.

`run()` first collects the requested number of syntactically and functionally valid
operators, then evaluates all of them in sample order. Poor fitness does not reject
or regenerate an operator. It returns `Result(source, attempt, metrics)` objects;
metrics are `None` without evaluation. Measurements must be a nonempty mapping of
finite numbers, including negative scores when meaningful. There is no ranking,
winner selection, reflection, repair prompt, or outer evolutionary loop.

Malformed JSON, invalid Python/interface, tool requests, `CandidateRejected`, and
validation timeouts consume one attempt and draw afresh. Provider failures and
unexpected validator failures propagate immediately. Benchmark failures or
nonfinite measurements abort evaluation without regenerating code. Transport
retries belong to the provider; the attempt count measures calls to that provider,
not internal requests. The default 100-attempt cap is a practical addition to the
paper's regenerate-until-ten protocol. Exhaustion raises `RuntimeError` and skips
benchmarking the partial batch.

`agent.attempts` retains prompts, raw responses before parsing, phase, accepted
source, measurements, and failure messages. `agent.operators` retains accepted
operators and completed measurements even if the run fails. Both reset on a new
run. Use one run at a time per instance. Imports leave Slick's global template root
unchanged; use separate processes for concurrent algorithms with different roots.

## Sources and interpretation limits

Source inspection performed on 2026-09-14:

- [Paper](https://arxiv.org/abs/2607.23505), supplied in full with the request:
  Section 3.3 and Figure 2 inform independent synthesis and validation; Algorithms
  1 and 2 inform the numerical rules.
- The paper's [operator-code repository](https://github.com/hengzhe-zhang/Zero-Shot-LLM-Benchmark)
  returned “Repository not found” on clone. Its generated source files could not
  be retrieved or used for equivalence tests.
- The authors' [official supplement](https://github.com/hengzhe-zhang/ppsn2026-zero-shot-llm-selection)
  was retrieved at commit
  [`7b8565bf823deb03516335e74621e35351e333dd`](https://github.com/hengzhe-zhang/ppsn2026-zero-shot-llm-selection/commit/7b8565bf823deb03516335e74621e35351e333dd).
  Its `supplementary_material.pdf` supplies Algorithm 2. The repository contains
  the PDF and README, without executable implementations.
- The request's [APET link](https://github.com/daankepel/APET) is a different prompt
  engineering method, already implemented in this workspace's `apet/` folder.

These are explicitly **pseudocode-based implementations**, not verified copies of
the unavailable generated operators. Algorithm 1 does not fully define error
normalization or the source of case variance, so both are caller inputs. It squares
the supplied `R_n` exactly as written; passing already-squared errors squares them
again. Algorithm 2 uses `clip(P @ P.T, -1, 1)` literally, with no implicit row
normalization. Its unspecified elite down-weight is exposed as `elite_downweight`,
defaulting to 0.5. Z-scores use population standard deviation and epsilon 1e-12.

The synthesis prompt intentionally replaces symbolic-regression wording with your
task, adds explicit membership/non-mutation requirements, and requests structured
JSON through Slick. It retains the 30-line preference, not a hard rejection rule
(the paper itself reports longer accepted operators). This is a domain-agnostic
adaptation, not an exact Figure 2 prompt reproduction.

## Verification

From the repository root:

```sh
rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_zero_shot_selection
rtk proxy ../slick/.venv/bin/ruff check zero_shot_selection tests/test_zero_shot_selection.py
rtk proxy ../slick/.venv/bin/ruff format --check zero_shot_selection tests/test_zero_shot_selection.py
```

Offline tests use the shared scripted provider through real Slick rendering and
typed parsing. They check independent calls, rejection budgets, timeout cleanup,
raw failure records, evaluation separation, alternate launch directories, hand
calculated selector cases, cardinality, ties, and input non-mutation. They do not
establish generated-code safety, measured optimization improvements, or reproduced
OpenML scores. No live model calls or benchmark replications have been run.
