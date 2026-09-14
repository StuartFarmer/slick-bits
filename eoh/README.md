# Evolution of Heuristics with Slick

Implementation of Liu et al., **Evolution of Heuristics: Towards Efficient
Automatic Algorithm Design Using Large Language Model**, ICML 2024, using the
local [Slick checkout](../../slick). It evolves a natural-language idea and Python
code together, evaluates each heuristic, and retains the best population.

## Run

Model instructions live in this implementation's `prompts/propose.j2`. The CLI
sets Slick's template root at startup, independent of the working directory.
Programmatic callers configure it once before rendering or generation:

```python
from pathlib import Path

from slick import prompts
import evolve

prompts.TEMPLATE_ROOT = Path(evolve.__file__).resolve().parent / "prompts"
```

The root is process-global; run implementations with different roots in separate
processes when they generate concurrently.

Python 3.10+; the only added dependency beyond Slick is NumPy.

```bash
cd ~/Developer/AI/slick-bits/eoh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python test_eoh.py

# Canned proposals through Slick's real render/parse path; no credentials needed.
.venv/bin/python evolve.py --problem binpacking --output runs/binpacking
.venv/bin/python evolve.py --problem tsp --output runs/tsp
.venv/bin/python evolve.py --problem flowshop --output runs/flowshop

# Real generation through Slick and your authenticated Codex CLI.
.venv/bin/python evolve.py --provider codex --problem binpacking --output runs/discovery
```

`demo` is the default provider. It cycles through fixed baseline functions and
does **not** discover new heuristics. Each smoke run initializes two individuals
and performs one generation of ten offspring. Output directories must be new;
existing results are never overwritten by the CLI.

The workspace's existing `../optimizer/.venv/bin/python` also has these
dependencies and was used for local verification.

For OpenAI, install Slick's optional SDK, supply credentials through the
environment, and choose the model explicitly:

```bash
.venv/bin/python -m pip install 'openai>=2,<3'
.venv/bin/python evolve.py --provider openai --model YOUR_MODEL_ID \
  --problem binpacking --output runs/api-discovery
```

Evolution executes generated Python automatically. Evaluations run in disposable
subprocesses with a wall timeout, a temporary working directory, and a reduced
environment without inherited API credentials. POSIX workers also have CPU and
file-size limits. **This is not a security sandbox:** candidates still have the
caller's filesystem and OS permissions. Run real searches in a disposable,
restricted environment suitable for executing generated code. Syntax/interface
checks validate the callable contract, not trustworthiness.

## Algorithm

`@prompt(template="propose.j2", output_type=Proposal)` asks for a short `thought`, followed by executable
`code`. `Heuristic` adds fitness and a run-local ID. Every prompt includes the
problem's function contract and the selected parents' thoughts, code and fitness.

| Operator | Action | Parents |
| --- | --- | --- |
| INIT | Design a new heuristic | None |
| E1 | Explore an idea different from the parents | p |
| E2 | Identify a shared idea and introduce new components | p |
| M1 | Modify algorithmic logic | 1 |
| M2 | Change numerical parameters, preserving structure | 1 |
| M3 | Remove redundant components | 1 |

The initial population is generated entirely by the provider. Initialization
retries invalid candidates up to `--init-attempts` (default `3*N`) and fails
explicitly if it cannot fill the population. No handcrafted heuristic is silently
substituted in real runs.

Each generation makes exactly `N` attempts per selected operator. Invalid JSON,
syntax, outputs, exceptions, timeouts and nonfinite fitness do not enter the
population, but consume their attempt. Parents are sampled with weights
`1/(rank + N)` using **one-based ranks**, without replacement within an exploration
prompt. All parents come from the population at the start of the generation.
Offspring and incumbents compete for `N` places; stable ties retain incumbents.
Duplicate code is allowed, as the paper's basic selection procedure does not
specify deduplication.

Calls and evaluations run sequentially. This keeps the generation semantics while
avoiding uncontrolled provider concurrency. A successful ordinary run uses
`N + generations * N * number_of_operators` model calls, plus any initialization
retries. Thus the paper's bin-packing settings use **2,020** calls, including the
20 initialization calls; 2,000 is the evolutionary portion alone.

## Problems and fitness

All internal fitness values are maximized.

| Problem | Generated function | Fitness |
| --- | --- | --- |
| Bin packing | `score(item, bins)` | Mean `lower_bound / bins_used` |
| TSP | `update_edge_distance(edge_distance, local_opt_tour, edge_n_used)` | Negative mean percentage gap to supplied optima; negative mean tour length when none are supplied |
| Flow shop | `get_matrix_and_jobs(current_sequence, time_matrix, m, n)` | Negative mean makespan |

Bin packing scores only feasible **open** bins, including exact fits. It opens a
new bin only when none fit. Highest score wins; first index breaks ties. This
implements the appendix's instruction to avoid unused bins. All scores must be
finite and have exactly the input vector's shape. The default lower bound is the
integer-size Martello–Toth L2 bound, including the volume bound. A dataset may
supply its own `lower_bound`. See the L2 characterization in
[Fekete and Schepers, §3.1](https://www.ibr.cs.tu-bs.de/users/fekete/hp/publications/PDF/1998-New_Classes_of_Lower_Bounds_for_Bin_Packing_Problems.pdf).
Report excess bins as `100*(bins/lower_bound - 1)`;
that reporting statistic is different from the optimized mean ratio.

TSP starts from nearest-neighbor construction, then descends with relocate and
2-opt. At a local optimum it updates the guiding matrix, takes one best improving
move on that matrix, and descends on the original distances. The heuristic sees a
closed 1-D tour and a symmetric count matrix, initially zero, incremented for all
edges in each tour presented for perturbation. Original instance matrices must
be symmetric and nonnegative with zero diagonal. Guiding matrices may be
asymmetric or negative, but must have the correct shape and finite values.

Flow shop starts with NEH insertion construction. It alternates swap/relocate
descent with a best improving move involving the heuristic's selected jobs on
the updated processing matrix. The selected job array must contain nonempty,
unique, in-range integer job IDs. Priority order determines move enumeration;
the best improvement wins. Guiding processing times must be finite and
nonnegative. Both solvers preserve the best sequence under the **original**
objective and pass copies of inputs to candidate functions.

These are explicit GLS adaptations of the paper's two-phase description, not
copies of the authors' optimized search kernels. Neighborhoods are exhaustive;
tour scans cost O(n³), scheduling scans O(n³m). `--seconds` bounds each GLS instance
cooperatively, and `--eval-timeout` bounds the entire worker process, including
candidate calls. `--iterations` bounds perturbation rounds. A timed-out worker
rejects that candidate. Each function also receives a small contract probe before
evaluation; stochastic heuristics consume random draws during that probe.

## Data and larger searches

```bash
.venv/bin/python evolve.py --provider codex --preset paper --problem binpacking \
  --data packing-train.json --output runs/packing-paper-settings

.venv/bin/python evolve.py --provider codex --preset paper --problem tsp \
  --data tsp-train-with-optima.json --output runs/tsp-paper-settings

.venv/bin/python evolve.py --provider codex --preset paper --problem flowshop \
  --output runs/flowshop-paper-settings
```

`paper` selects 20 generations, p=5, N=20 for packing and N=10 otherwise; five
5,000-item packing instances, 64 TSP100 instances, or 64 50-job flow shops;
1,000 GLS perturbation rounds and 60 wall seconds per instance. Explicit flags
override these defaults. It does not select the original GPT-3.5-turbo model or
recover the original data. TSP's paper preset requires supplied optima.

Without `--data`, seeded **synthetic** instances are generated once and saved:
packing uses rounded `Weibull(shape=0.5) * capacity/10`, clipped to `[1, capacity]`;
TSP uses Euclidean distances between points uniform in `[0,1]²`; flow shop uses
processing times uniform in `[0,1]`. Packing's distribution parameters are an
explicit approximation, not a reconstruction of the published Weibull dataset.
`--machines 0` samples 2–20 machines per instance (paper default); positive values
give a fixed machine count. `--seed` controls datasets, selection and each worker's
NumPy seed. Wall-time-limited searches and real LLM responses are not guaranteed
bitwise reproducible.

`--data` accepts JSON with `problem` and a nonempty `instances` list. Examples:

```json
{"problem": "binpacking", "instances": [
  {"items": [6, 4, 6, 4], "capacity": 10, "lower_bound": 2}
]}
```

```json
{"problem": "tsp", "instances": [
  {"distances": [[0, 1, 1], [1, 0, 1], [1, 1, 0]], "optimum": 3}
]}
```

```json
{"problem": "flowshop", "instances": [
  {"times": [[2, 1], [1, 3], [3, 2]]}
]}
```

Supply optima for every TSP instance or none; mixed metrics are rejected. No
Concorde, TSPLIB, or Taillard loader is bundled: convert external instances to
this format, preserving their distance/rounding conventions.

Reevaluate saved code on held-out data without a model call:

```bash
.venv/bin/python evolve.py --problem binpacking --candidate runs/discovery/best.py \
  --data packing-test.json --output runs/held-out
```

Operator subsets permit EoH-e1 (`--operators E1`), EoH-e2 (`--operators E1 E2`) and
EoC (`--operators E1 --code-only`). Set generation counts explicitly to match query
budgets. Separate runs generate separate initial populations; these flags alone
do not reproduce the paper's controlled ablation with shared initialization.

## Results and verification

Each run saves `config.json`, the actual `dataset.json`, baseline evaluations,
`attempts.jsonl` with rendered prompts, parent IDs, code, fitness or failure,
`history.json` with every generation, `population.json`, and `best.py`/`best.txt`.
Attempts stream to disk; population JSON updates use atomic replacement. Interrupted
runs retain completed records but automatic resume is not implemented.

Packing baselines include first fit, best fit, the published FunSearch function,
and Figure 6's EoH function with an algebraically equivalent stable exponential.
TSP has a simple usage-adjusted edge penalty baseline. Flow shop uses Figure 10's
random machine/job perturbation, adjusted to choose at least one job for small n.
All baseline and candidate functions use the same evaluators and seeds. The
neural comparison solvers and published TSP winning function are not bundled.

The offline checks exercise actual Slick rendering/parsing, exact generation
budgets, parent provenance, elitism after failed offspring, bounded initialization,
candidate timeouts, shape/NaN rejection, immutable objectives, known small optimal
solutions, actual escapes from local optima, selected-job restrictions, fitness
signs, and L2 validity against exact small bin packings. The
three CLI smoke runs verify execution only. Paid searches, full benchmark tables,
published performance claims, and statistical ablations have not been reproduced.

The supplied paper is the algorithm specification. The authors'
[repository](https://github.com/FeiLiu36/EoH) was inspected on 2026-09-14; its current
API differs from its legacy paper-era interface. This implementation depends on
Slick, not that repository's package.
