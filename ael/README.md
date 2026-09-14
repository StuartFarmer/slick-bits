# AEL with Slick and OpenRouter

Implementation of **Algorithm Evolution Using Large Language Model** by Fei Liu,
Xialiang Tong, Mingxuan Yuan, and Qingfu Zhang, using the paper supplied with this
project. Slick renders the initialization, crossover, and mutation prompts;
ordinary Python owns selection, evaluation, and survival.

## Run

Model instructions live in this implementation's `prompts/propose.j2`. The CLI
sets Slick's template root at startup, independent of the working directory.
Programmatic callers configure it once before rendering or generation:

```python
from pathlib import Path

from slick import prompts
import ael

prompts.TEMPLATE_ROOT = Path(ael.__file__).resolve().parent / "prompts"
```

The root is process-global; run implementations with different roots in separate
processes when they generate concurrently.

```bash
cd ~/Developer/AI/slick-bits/ael
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python test_ael.py
.venv/bin/python ael.py demo --output runs/demo
```

Python 3.10+ and the local editable `../../slick` checkout are required.
The offline demo uses canned nearest-neighbor and Figure 6 implementations.
It exercises the actual Slick prompt path and evolution loop; it does **not**
demonstrate LLM discovery or reproduce the paper's performance claims.

For real evolution, start Docker and build the evaluation image:

```bash
docker build -t slick-ael:local .
.venv/bin/python test_ael.py --docker

# 64 fixed, uniformly distributed TSP50 instances, with exact reference lengths.
.venv/bin/python ael.py dataset --output runs/train50.json

# Set OPENROUTER_API_KEY through your normal environment, then select a model.
.venv/bin/python ael.py evolve --model YOUR_OPENROUTER_MODEL_ID \
  --dataset runs/train50.json --output runs/evolution
```

The `dataset` command solves every reference before evolution. It uses SciPy's
HiGHS MILP solver with degree constraints and repeated subtour elimination cuts.
A timeout or unproven optimum fails reference generation; no incumbent is
silently labeled optimal. See the [SciPy MILP status contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).
`--solver-timeout 60` is the per-instance default. Increase it for harder cases.
For a quick real-model smoke run, use `dataset --sizes 8 --instances 4` and
`evolve --population-size 2 --generations 1`.

## What follows the paper

| Setting | Default |
| --- | --- |
| Population size, N | 10 |
| Generations after initialization, Ng | 10 |
| Crossover probability | 1.0 |
| Mutation probability | 0.2 |
| Parents, l | 2, uniform sampling without replacement |
| Offspring per iteration, s | 1 |
| Evaluation instances | 64 TSP50 instances, fixed throughout evolution |
| Fitness | Mean of `100 * (tour_length / reference_length - 1)` |

Individuals hold a natural-language description, Python source, and fitness.
Each generation performs N selections from the **unchanged starting population**.
Crossover generates s children; each child can be replaced by an LLM mutation.
Only then are offspring evaluated. At the generation boundary, parents and
valid children are sorted by ascending fitness and the best N survive.
No handcrafted algorithm seeds a real run's initial population.

The four-argument `select_next_node` interface, description delimiters, and
three prompt intents follow Figures 3–4. Optional additional parameters may have
defaults. Node zero is both the start and destination; candidates choose every
remaining node, including the last, and the harness closes the tour. Figure 6
is included as a comparison with an explicit singleton guard to avoid NumPy's
empty-mean warnings. It is not supplied in real discovery prompts.

The paper leaves some execution details unspecified. This implementation uses:

- Uniform coordinates in `[0,1)^2`, sorted unvisited IDs, and node zero as start.
- One prompt per child when `--offspring` exceeds one. If crossover is skipped,
  each child copies a uniformly chosen selected parent before possible mutation.
- Rejection of malformed responses, exceptions, timeouts, and invalid tours.
  Initialization has at most `3*N` attempts; failed offspring leave existing
  parents eligible for survival. Provider failures abort and preserve artifacts.
- A seeded Python evolution RNG and NumPy dataset RNG. Each evaluation resets
  the worker RNGs. Model sampling itself is not reproducible or configured by
  this application; the provider controls its defaults.
- HiGHS instead of Gurobi for generated optimal references. OpenRouter supplies
  the chosen model; the historical GPT-3.5/GPT-4 experiments are not replicated.

With defaults, a successful run makes 110 initialization/crossover calls plus
about 20 mutation calls, potentially more for invalid initialization attempts.
This follows the pseudocode rather than the prose's approximate 100 interactions.

## Artifacts and evaluation on new sizes

Each new run directory saves the full dataset and configuration, every rendered
prompt and raw response, candidate Python files, parent IDs and operations in
`events.jsonl`, and evaluation JSON containing tours, lengths, runtime, or errors.
`history.json` contains generation best/mean scores and every survivor's fitness;
`population.json` and `best.py` are updated each generation. `result.json` saves
the final and initial populations plus the greedy and direct-LLM training gaps.
Outputs are never overwritten. A failed run retains partial artifacts; automated
resume is not implemented.

```bash
# A separate seed prevents reuse of training instances.
.venv/bin/python ael.py dataset --sizes 20 50 100 --instances 64 --seed 1 \
  --solver-timeout 300 --output runs/heldout.json
.venv/bin/python ael.py benchmark --result runs/evolution/result.json \
  --dataset runs/heldout.json --evaluation-timeout 120 --output runs/comparison
```

The benchmark evaluates greedy, Figure 6, the evolved winner, and **every initial
LLM algorithm** on each supplied size. It saves individual tours, mean lengths,
mean gaps, and errors in `results.json`. The initial algorithms support direct-LLM
mean/best comparisons; choose the best using training fitness when reporting a
held-out comparison. POMO training and neural checkpoints are outside this
implementation. No superiority or statistical significance is claimed.

TSP200/500/1000 work through the same interfaces. Exact MILP reference generation
can be expensive at those sizes; supply LKH3/Gurobi results in the dataset format
below. The published Figure 6 heuristic itself has cubic total tour-construction
work, so increase the candidate timeout for large batches.

```json
{
  "reference": "LKH3 (external reference, not a proof of optimality)",
  "instances": [
    {
      "coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]],
      "reference_length": 4.0
    }
  ]
}
```

Lengths must correspond to the supplied coordinates and **unrounded Euclidean
distances**, including any scaling used by an external solver. External lengths
are accepted as supplied; they are not independently certified. For plumbing or
heuristic comparisons, `dataset --reference greedy` is available and explicitly
labels every score as a gap to greedy, not optimality. Negative gaps then mean
improvement over greedy.

## Candidate execution

Real runs use a fresh Docker container per evaluated algorithm, with no network,
a read-only filesystem, no capabilities, no host credentials, and limits of
512 MB, one CPU, and 64 processes. Only the worker script is mounted, read-only;
code and coordinates enter through stdin. The host checks the returned tour
permutations and recomputes lengths from the original data. An evaluation timeout
covers the **whole batch**, including container startup, and removes the container.
The greedy preflight verifies evaluation works before any model call.

`--trusted-code` is an explicit escape hatch for execution inside an independently
isolated environment. It runs ordinary Python locally in a temporary working
directory with a timeout; it is **not a security sandbox**. The offline demo and
default tests use this path only with source fixtures maintained in this project.
Do not use it for unreviewed model output on your host.

Checks cover exact solver agreement with brute force (including disconnected
subtours), fitness aggregation, invalid node IDs and signatures, evaluation
deadlines, real Slick rendering/parsing, malformed-response recovery, generation
selection boundaries, elitism, multiple offspring, skipped operators, and saved
artifacts. The optional Docker check exercises the actual container boundary.
