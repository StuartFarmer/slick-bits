# QUBE with Slick

A runnable implementation of **QUBE: Enhancing Automatic Heuristic Design via
Quality-Uncertainty Balanced Evolution**, using the sibling
[Slick checkout](../../slick) for async two-parent prompts and model calls.
Python owns the evolutionary search; NumPy evaluates the heuristics.

## Run an offline demonstration

Requires Python 3.11–3.14. From `slick-bits`:

```bash
cd qube
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ..
qube/.venv/bin/python -m qube.run --output qube/runs/binpack-demo
qube/.venv/bin/python -m qube.run --task capset --samples 8 --output qube/runs/capset-demo
qube/.venv/bin/python -m qube.run --task tsp --samples 4 --instances 2 --output qube/runs/tsp-demo
```

The default provider cycles through fixed, trusted heuristics. It exercises Slick,
selection, evaluation, offspring attribution, and logging without credentials or
model calls. It does **not** discover new code or reproduce the paper's results.
For a small reset demonstration, add `--islands 4 --reset-interval 8 --samples 16`.
Every run requires a fresh output directory.

The existing `optimizer/.venv/bin/python` can also run these commands if that
environment has Slick and NumPy installed.

## Run a real model

Build the evaluator image once; Docker must be running:

```bash
docker build -t qube-evaluator:local qube
```

Slick's `LiteLLMAPI` supports local OpenAI-compatible servers and hosted providers.
For an already-running inference server, for example:

```bash
OPENAI_API_KEY=local qube/.venv/bin/python -m qube.run \
  --provider litellm --model openai/OpenCoder-8B-Instruct \
  --api-base http://localhost:8000/v1 \
  --samples 80000 --samplers 16 --evaluators 50 \
  --output qube/runs/weibull-search
```

Replace the model name and endpoint with those actually served. For hosted models,
use the appropriate LiteLLM model ID and provider credential environment variable;
omit `--api-base`. Generation uses temperature 1.0, top-p 0.95, a 4,096-token output
limit, and no automatic provider retries. It can incur provider charges.

Real candidates run in a fresh Docker container with no network or host mounts,
a read-only root filesystem, an unprivileged user, no capabilities, and limits on
memory, processes, files, and elapsed time. Both output streams are capped at 1 MiB
during execution, and Docker logging is disabled. The runner forcibly removes
timed-out or output-flooding containers. Syntax checks are not a sandbox. The host execution path accepts only
the exact built-in demo sources. Model output is never evaluated on the host by
the real-provider path.

The worker and candidate share a process inside the container; this isolates host
resources, but is not a tamper-proof benchmark judge. Do not substitute an
untrusted evaluator image. Heuristics should be deterministic; random seeds are
fixed for each evaluation, but arbitrary generated code can still use nondeterministic
inputs. Infrastructure failure during seed evaluation stops the run; subsequent
provider/evaluation failures are logged as rejected attempts.

## Problems and data

`--data` accepts a JSON list. The evaluated dataset is saved with the run.

| Task | Evolved function | Signature and maximized score |
|---|---|---|
| `binpack` | `priority(item, bins)` | Bins used per instance; `1 - sum(used)/sum(lower_bounds)` |
| `capset` | `priority(element, n)` | Cap-set size per dimension; mean size |
| `tsp` | `update_dist(distance_matrix, current_route)` | Tour length per instance; negative mean excess ratio if optima are provided, otherwise negative mean length |

**Bin packing:** without `--data`, generates five Weibull(shape=3, scale=45)
instances with 1,000 items and bin capacity 100, clipped/rounded to integers 1–100.
Use `--items 5000` or `--items 10000` for the larger settings. The evaluator considers
all feasible bins, including unused bins, and breaks priority ties by first index.

```json
[{"id": "tiny", "capacity": 10, "items": [6, 4, 6, 4], "lower_bound": 2}]
```

Missing lower bounds are computed using Martello–Toth L2, following the
[estimated-waste formulation in Korf (2002)](https://cdn.aaai.org/AAAI/2002/AAAI02-110.pdf).
Supplied bounds are trusted reference data after basic range validation.
For OR-Library files, pass the original text file, e.g. `--data /path/to/binpack3.txt`.
The loader retains the file's best-known packing count as `reference_bins` and
computes L2 separately: a best-known packing is not automatically a lower bound.
Use `--k 0.0001` for Weibull data loaded from a file; binpack `--data` otherwise
selects OR hyperparameters.

**Cap sets:** default dimension is eight (`--dimension 8`); JSON input is a list
such as `[8]`. Greedy construction blocks every vector completing a zero-sum
triple with two distinct selected vectors. Dimensions 1–10 are supported; the
candidate space grows as `3**n`.

**TSP:** without `--data`, generates five uniform `[0,1]^2` instances with 20 cities.
Set `--cities 50` or `--cities 100`, and `--instances 1000` for the paper's test-set
size. Guided search performs 100 rounds of symmetric 2-opt followed by additive
distance updates. The retained best tour is always measured on original distances.

```json
[{"cities": [[0,0],[1,0],[1,1],[0,1]], "optimum": 4.0}]
```

Supply independently computed, positive `optimum` lengths for **every** TSP instance
to report excess ratios. Concorde is not bundled or run automatically. When optima
are absent, the output explicitly labels the metric `negative_mean_tour_length`;
it is not comparable to the paper's excess-ratio table.

## Mapping to the paper

- Each island clusters programs by the complete deterministic outcome tuple,
  not aggregate score. Exact duplicate sources in a cluster are stored once.
- Equation 1 uses the mean score of valid offspring, credited to both distinct
  parent clusters, independent of the offspring's destination cluster.
- Equation 2 is `Q + k * sqrt(log(t)/N)`. Parent selection takes the top two UIQ
  clusters; programs within each cluster use the paper's length softmax.
- Islands are sampled uniformly. Resets rank each island by its maximum cluster
  UIQ, replace the bottom half, and choose a uniform random program from the best
  cluster of a uniform random surviving island. The global best survives resets.
- The last 500 attempted children define the rolling window. Recent best score
  and token-level Levenshtein change are computed over valid children in it;
  change is distance to the nearest parent divided by child token count.

The paper does not fully specify cold starts, invalid-child scores, tied medians,
or pending asynchronous work. This implementation makes those conventions explicit:

- Unvisited clusters have infinite UIQ when `k > 0`. Before a valid offspring is
  available, the quality prior is the cluster's own score. With `k=0`, there is no
  infinite exploration bonus.
- `N` counts attempted children using a cluster, reserved before their calls;
  failed calls and invalid children consume visits/budget but do not enter the
  offspring mean. Two examples from the only cluster credit it once per child.
- `t` is completed attempts plus one. Equal UIQ values are randomly ordered;
  exactly `floor(n/2)` islands reset, including tied medians.
- Async samplers run in one process; evaluations run in isolated child processes.
  At reset boundaries all pending children finish before island replacement.
  Multiple children per prompt are separate Slick calls with identical parents.
  With multiple workers, completion order affects evolution even with a fixed seed.

The supplied appendix contains truncated Python and inconsistent cap-set signatures.
The prompts use complete functions with the evaluator's actual signatures. TSP uses
a complete symmetric 2-opt neighborhood and validates symmetric updates rather than
copying the appendix's broken pseudocode verbatim. The [authors' public repository](https://github.com/zzjchen/QUBE_code)
contained only a placeholder README when checked; no implementation was copied.

| Parameter | OR | Weibull | Cap set | TSP |
|---|---:|---:|---:|---:|
| Islands | 10 | 10 | 10 | 1 |
| `k` | 0.0008 | 0.0001 | 32 | 0.00001 |
| Reset interval | 32768 | 32768 | 262144 | disabled |
| Samples per prompt | 4 | 4 | 4 | 1 |
| Evaluation timeout (s) | 30 | 60 | 90 | 90 |
| Program temperature | 1 | 1 | 1 | 1 |
| Paper sample budget | 80000 | 80000 | 2000000 | 2000 |

These task-specific defaults are built in. The CLI defaults to **12 samples,
2 samplers, and 2 evaluators** for a small run; pass the paper's budgets and
`--samplers 16 --evaluators 50` explicitly. CLI options expose `k`, temperatures,
timeouts, island count, and reset interval. Single-island runs never reset.

## Artifacts and checks

Each output directory contains `config.json`, `instances.json`, `history.jsonl`
(candidate code, parents, outcomes, failures, and rolling metrics), `best.py`,
`best.json`, and `summary.json`. `resets.json` is written at reset boundaries.
History is flushed and improvements saved during the run. There is no resume
support, multi-run benchmark aggregator, or FunSearch/EoH baseline implementation.
The sample budget includes failed attempts and excludes seed evaluation.

```bash
qube/.venv/bin/python -m unittest qube.test_qube
QUBE_TEST_DOCKER=1 qube/.venv/bin/python -m unittest qube.test_qube
```

The opt-in Docker check verifies evaluation, read-only filesystem, no outbound
network, invalid-result rejection, bounded stdout/stderr, timeout, and container cleanup. The offline
checks cover UIQ arithmetic/selection, offspring credit, reset behavior, input
validation, tiny optima, cap-set validity, Slick prompting, and failure accounting.
Published performance has not been reproduced; these checks establish implementation
behavior, not research efficacy.

## Prompt templates

Model instructions live in this implementation's own `prompts/` directory.
The CLI sets Slick's absolute template root once at startup. Programmatic callers
configure it before rendering or running the algorithm:

```python
from pathlib import Path

from slick import prompts
import qube.run as operations

prompts.TEMPLATE_ROOT = Path(operations.__file__).resolve().parent / "prompts"
```

The root is process-global; run implementations with different roots in separate
processes. Prompt wording, output parsing, and caller-owned validation are unchanged.
