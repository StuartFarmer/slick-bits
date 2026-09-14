# Optimizing the optimizer with Slick

A Python implementation of **CMSA for maximum independent set**, the paper's
age-aware V1 and entropy-adjusted V2 constructions, and a
[Slick](../../slick) dialogue for generating and revising construction code.
All five benchmark variants run locally without an LLM or a CPLEX license.

## Run

Model instructions live in this implementation's `prompts/propose.j2`. The CLI
sets Slick's template root at startup, independent of the working directory.
Programmatic callers configure it once before rendering or generation:

```python
from pathlib import Path

from slick import prompts
import improve

prompts.TEMPLATE_ROOT = Path(improve.__file__).resolve().parent / "prompts"
```

The root is process-global; run implementations with different roots in separate
processes when they generate concurrently.

```bash
cd ~/Developer/AI/slick-bits/optimizer
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python test_optimizer.py

# Six graphs: three families, two density parameters, five algorithms.
.venv/bin/python experiment.py --output runs/comparison

# Exercise Slick with a clearly labeled canned response, without credentials.
.venv/bin/python improve.py --output runs/dialogue
```

Python 3.10+ is required. Dependencies use the local editable `../../slick`
checkout, SciPy's bundled HiGHS solver, and NetworkX graph generators.
The local `.venv` is already installed if using the workspace in which this
implementation was created.

## Generate a heuristic, give feedback, compare it

```bash
# Uses your existing Codex CLI authentication; makes a real model call.
.venv/bin/python improve.py --provider codex --output runs/discovery

# The same directory restores the complete source and previous dialogue.
.venv/bin/python improve.py --provider codex --output runs/discovery \
  --feedback 'The oldest vertices dominate selection. Revisit how age affects probability.'

# Request the paper's second kind of improvement: implementation efficiency.
.venv/bin/python improve.py --provider codex --output runs/discovery \
  --mode performance --feedback 'Preserve the previous heuristic and random choices.'

# After reviewing the generated Python, explicitly execute it for comparison.
.venv/bin/python experiment.py --constructor runs/discovery/candidate-003.py \
  --sizes 100 --densities 0.1 0.5 --instances 3 --output runs/candidate-comparison
```

`@prompt(template="propose.j2", output_type=Proposal)` renders the complete baseline CMSA implementation,
the improvement request, previous proposals, and researcher feedback. Slick parses
the answer into code, rationale, and proposed checks. A fresh dialogue excludes
the published V1/V2 formulas so that they are not supplied as the answer to a
discovery request. `--source path.py` accepts another complete Python baseline;
the supplied function must use this project's construction interface.

Each turn saves the rendered prompt, candidate source, and `dialogue.json`.
Syntax/interface failures remain in the dialogue for the next feedback turn.
The application does not execute generated code automatically. Syntax checking
is not a security sandbox; `experiment.py --constructor` imports and runs that
file as ordinary trusted Python, with the same permissions as the caller.
Candidates must honor the supplied deadline. Use one writer per dialogue directory.

OpenAI is also supported; install its optional SDK and supply an explicit model:

```bash
.venv/bin/python -m pip install 'openai>=2,<3'
export OPENAI_API_KEY=...  # or supply it through your normal environment
.venv/bin/python improve.py --provider openai --model YOUR_MODEL_ID \
  --output runs/api-discovery
```

The offline demo always returns a wrapper around the existing V1 implementation;
it demonstrates plumbing, not model discovery. Real providers were not called
during verification. Slick's OpenAI provider uses its own API defaults and an
8192-token output budget here, not the paper's exact sampling configuration.

## Algorithms and interpretation

Each iteration constructs `n_solutions` independent sets, merges their vertices
into the reduced problem, solves a binary MILP, and adapts ages. The MILP maximizes
`sum(x_v)` with `x_u + x_v <= 1` for every reduced edge and `x_v in {0,1}`.
Construction and solver incumbents both update the best solution seen.

| Variant | Random selection | Availability representation |
| --- | --- | --- |
| `cmsa` | Uniform among the first `candidate_list_size` feasible vertices by static degree | Filtered list |
| `v1` | Proportional to `1/(2+age[v]) + 1/(1+degree[v])` | Filtered list |
| `v2` | V1 probabilities plus entropy, then renormalized | Filtered list |
| `v1-perf` | Identical to V1 | Python integer bitset plus ordered list |
| `v2-perf` | Identical to V2 | Python integer bitset plus ordered list |

The authors' [repository](https://github.com/camilochs/optimizing-the-optimizer)
links to a [C++ archive](https://imp-opt-algo-llms.surge.sh/static/cmsa-mis-llm.zip),
inspected on 2026-09-13. Its behavior resolves several ambiguities in the supplied
paper:

- All deterministic branches select the first available vertex in **static
  increasing degree order**. We follow that code, rather than the paper's
  `argmin(P_w)` equation. Vertex ID breaks degree ties for reproducibility here.
- Random V1/V2 selection considers all currently available vertices. Only the
  baseline uses a restricted candidate list.
- Ages start at `-1`; newly merged vertices get age zero. Reconstructing an already
  present vertex does not reset its age. After a feasible solver result, selected
  vertices reset to zero, other pool ages increment, and ages **at least**
  `age_max` are removed (set to `-1`). No solver incumbent means no adaptation.
- For V2, with `m` available vertices, normalization is
  `P_H(v) = (P_w(v) + H) / (1 + m*H)`, where
  `H = -sum(P_w(v)*log(P_w(v)))`. This follows the archive's renormalization;
  the displayed denominator in the paper is ambiguous. A singleton has `H=0`.

The PERF implementations preserve the corresponding Python constructor's
candidate ordering, probabilities, and random draws. They do not emulate C++
cache alignment, prefetching, or its fixed 32768-bit capacity. No speedup or
memory reduction is claimed. Availability scans can still be quadratic.

## Benchmarks and input

```bash
.venv/bin/python experiment.py --sizes 500 1000 2000 3000 \
  --densities 0.1 0.2 0.5 1 --instances 30 \
  --time-limit 10 --solver-time-limit 1 --output runs/larger-comparison

.venv/bin/python experiment.py --graph my-graph.txt --time-limit 5 \
  --output runs/input-comparison
```

The large command creates 1440 **new synthetic** instances. Its ten-second budget
is an example, not the paper's schedule. The defaults are a short smoke benchmark.
`--graph` can be repeated. Files may contain the vertex count on the first line
and zero-based `u v` edge pairs, or DIMACS `p edge n m` / one-based `e u v` lines.
Isolated vertices are retained; malformed endpoints, self loops, duplicate edges,
and inconsistent DIMACS counts are rejected.

Generator parameters are explicit approximations because the original instances
and their density mapping were not supplied:

- ER: edge probability `p=d`.
- BA: attachment count `m=round(d*n/2)`, clipped to `[1,n-1]`.
- WS: even neighborhood size near `d*(n-1)`, clipped to the valid range, and
  rewiring probability `0.1`.

The density parameter is not an equal realized density across these families.
Each graph gets a stable seed derived from its specification and `--seed`.
Algorithms share that graph and seed, with shuffled execution order and equal
budgets. Tune `--n-solutions`, `--age-max`, `--determinism-rate`, and
`--candidate-list-size` explicitly; defaults are **not irace-tuned**.
`--iterations` adds a cap useful for small repeatability checks.

Every output directory contains the actual graph files, `metadata.json`, streamed
`results.jsonl` records with vertex sets and convergence history, and `summary.json`
with mean scores and tie-aware paired ranks (rank 1 is best). Candidate source is
copied into the output. Existing benchmark directories are never overwritten.

Budgets use elapsed wall time and record CPU time separately. This differs from
the paper's CPU budgets of 150/300/450/600 seconds. Deadlines are cooperative:
graph/model preparation and solver shutdown can slightly exceed them. A solver
timeout keeps the best construction and any feasible MILP incumbent. Reduced
optimality is not global optimality unless the pool contains the full graph.
HiGHS replaces CPLEX; CPLEX warm starts, abort callbacks, and emphasis options
are not reproduced. See [SciPy's MILP contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

Checks compare the MILP with brute force on empty, complete, path, cycle, star,
and seeded random graphs; verify construction feasibility/maximality, ages,
probabilities, PERF equivalence, and timeout handling; and exercise Slick's real
render/parse path and saved feedback using a canned provider.
The included short benchmark verifies execution, not the paper's superiority
claim. Published rankings, irace tuning, Friedman/Nemenyi significance tests,
and the exact figures have not been reproduced.
