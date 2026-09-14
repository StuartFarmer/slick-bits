# LLM4MOEA

Problem-agnostic implementation of **Large Language Model for Multi-objective
Evolutionary Optimization** (Liu et al.), based on the
[paper](https://arxiv.org/abs/2310.12541) and
[official implementation](https://github.com/FeiLiu36/LLM4MOEA).
`MOEAD` implements both MOEA/D-LLM and MOEA/D-LO, with no benchmark or provider
hardcoded. It requires Slick and the Python standard library.

## Use

Supply real-valued bounds and `async evaluate(x) -> Sequence[float]`.
All objectives are minimized; negate objectives you want to maximize.

```python
import asyncio
from llm4moea import MOEAD

async def evaluate(x):
    # Replace this example with your own measurements or simulation.
    return sum(v * v for v in x), sum((v - 1) ** 2 for v in x)

async def main():
    optimizer = MOEAD(
        task="Find trade-offs between two costs.",
        bounds=[(-2.0, 2.0)] * 5,
        objectives=2,
        evaluate=evaluate,
    )
    result = await optimizer.run(operator="lo", max_evaluations=1000, seed=42)
    for individual in result.archive:
        print(individual.x, individual.objectives)

asyncio.run(main())
```

For the LLM operator, inject an existing Slick provider and configure the template
root once at application startup, before generation:

```python
from pathlib import Path
import llm4moea
from slick import prompts

prompts.TEMPLATE_ROOT = Path(llm4moea.__file__).resolve().parent / "prompts"

async def search_with_llm(provider, evaluate):
    optimizer = llm4moea.MOEAD(
        "Minimize measured cost and latency.",
        bounds=[(0.0, 10.0), (1.0, 100.0)],
        objectives=2,
        evaluate=evaluate,
        provider=provider,
    )
    result = await optimizer.run(operator="llm", max_evaluations=1000, seed=42)
    return result, optimizer.attempts
```

This preserves the paper's **bounded real-vector** representation. Arbitrary
objective functions and any number of objectives work without changing the
optimizer. For discrete choices or artifacts, decode the numeric vector in your
evaluator and retain that mapping when reading results. This is not a text/code
evolution algorithm. Bounds alone are enforced here; problem constraints,
objective scaling, execution isolation, and deadlines belong to the evaluator.
Fixed coordinates (`lower == upper`) are supported. No generated code is executed.

## Algorithm and budgets

1. Generate a complete Das–Dennis simplex lattice and find nearest-weight
   neighborhoods. Initialize one uniform random solution per weight vector.
2. Select parents without replacement from the neighborhood with probability
   0.9, otherwise from the whole population. Sort them worst-to-best by the
   current subproblem's Chebyshev value `max(w[j] * (f[j] - ideal[j]))`.
3. Generate offspring with the chosen operator, clamp to bounds, and apply
   polynomial mutation. Evaluate each offspring and immediately update the ideal
   point, population, and external nondominated archive.

Default settings are population target 50, neighborhood 10, 10 parents, 1,000
total evaluator calls, and at most two replacements per offspring. A complete
lattice can be smaller than the requested population: target 50 gives 50 points
for two objectives and 45 for three. Use `weights=das_dennis(m, partitions)` or
your own simplex weights for explicit control. Supplied weights determine the
population size. Exact zero weight components are retained.

Provide usable settings: at least two objectives, nonempty finite bounds with
lower <= upper, enough population/neighbors for the parent count, positive
offspring and attempt counts, and a budget covering initialization. Configuration
is trusted, following the Slick house style; there are no preflight validators.

- **LO:** one offspring by default. Compute cubic rank weights with ranks
  `1, (l-1)/l, ..., 1/l`, add independent `N(0, 0.5²)` noise per parent, and
  combine the parents in original coordinate units. Noisy weights are neither
  clipped nor renormalized. The same weights apply to every dimension. Update
  each coordinate with probability 0.1; otherwise copy the current subproblem's
  incumbent. Try the shuffled local/global selection pool for replacement;
  accept equal or better scalar values, as in the official MATLAB code.
- **LLM:** two offspring by default. Present normalized coordinates and scalar
  values in descending order. Preserve the paper's `<start>...<end>` text format.
  Parse finite vectors of the correct dimension, clip normalized coordinates to
  `[0,1]`, and map back to original units. Evaluate all usable requested offspring.
  Visit subproblems in shuffled order and replace up to two strictly improved
  neighbors in distance order, following the official Python replacement rule.
- **Mutation:** distribution index 20; each coordinate is selected with
  probability `1/d`. `mutation_probability` gates mutation per offspring and
  defaults to 1, matching the LO implementation; set it to 0 to disable mutation.
  `mutation_eta`, `noise`, `dimension_probability`, `neighbor_probability`,
  parent/offspring counts, and neighborhood size are explicit run parameters.

`Result` contains immutable population/archive snapshots, the ideal point,
evaluator-call count, and model-call count. The archive contains all evaluated
nondominated solutions, including candidates that never entered the population;
exact duplicate individuals are removed. It is unbounded and scanned linearly,
so large fronts can become expensive. No benchmark metrics are computed.

`optimizer.measurements` records every evaluator invocation, including failures.
`optimizer.attempts` records raw model text, parent coordinates/scalar values,
parsed offspring before mutation, rejected blocks, and failure status. These
input/output records can also support subsequent operator fitting; the shipped
LO uses the published coefficients and requires no fitting or model access.

- Initialization counts against the budget and aborts on its first rejected
  evaluation, preserving partial records. It never evaluates beyond the budget.
- `CandidateRejected` or an evaluator `TimeoutError` rejects that offspring and
  consumes an evaluation. Wrong-length or nonfinite objective measurements are
  also rejected. Other evaluator exceptions propagate after recording the error.
- If a model response contains no usable points, retry the same prompt, at most
  `max_attempts=3` calls per subproblem. Exhaustion raises `RuntimeError`. Partial
  valid batches are used; malformed blocks are recorded, excess points ignored.
  Repeated points are allowed, as in the official code; the prompt requests novelty.
- Provider errors and timeouts propagate. The caller owns transport retries and
  deadlines. Each attempt invokes Slick with exactly one `provider=` argument;
  there is no Session or conversation history and no fallback operator.
- The final batch is shortened to the remaining evaluation budget. Rejected
  generations cost model calls, not objective evaluations. No objective caching
  or hidden extra evaluations occur.

Each run resets records and its local seeded RNG. LO is reproducible for a
deterministic evaluator; a seed does not control external model randomness.
Use one active run per instance. Slick's template root is process-global:
configure once, and use separate processes for applications needing different roots.

## Official source and resolved differences

Inspected and used at commit `7fb29810abd891643914a0a7452c27956d38bddb`:

- [MOEADLO10.m](https://github.com/FeiLiu36/LLM4MOEA/blob/7fb29810abd891643914a0a7452c27956d38bddb/MOEAD-LO/MOEADLO10.m)
  and [OperatorLO.m](https://github.com/FeiLiu36/LLM4MOEA/blob/7fb29810abd891643914a0a7452c27956d38bddb/MOEAD-LO/OperatorLO.m):
  rank ordering, polynomial coefficients, additive Gaussian weight noise,
  coordinate mask, mutation, and limited replacement.
- [moead_LLM.py](https://github.com/FeiLiu36/LLM4MOEA/blob/7fb29810abd891643914a0a7452c27956d38bddb/MOEAD-LLM/pymoo/algorithms/moo/moead_LLM.py)
  and [gpt.py](https://github.com/FeiLiu36/LLM4MOEA/blob/7fb29810abd891643914a0a7452c27956d38bddb/MOEAD-LLM/pymoo/operators/crossover/gpt.py):
  neighborhood selection, normalized model inputs, tagged outputs, bound repair,
  independent calls, and Python replacement rules.
- [LinearReg.py](https://github.com/FeiLiu36/LLM4MOEA/blob/7fb29810abd891643914a0a7452c27956d38bddb/LLM2LO/LinearReg.py)
  and [PolyReg.py](https://github.com/FeiLiu36/LLM4MOEA/blob/7fb29810abd891643914a0a7452c27956d38bddb/LLM2LO/PolyReg.py):
  interpretation of dimension-wise input/output samples and the learned operator.

The paper and code do not define one identical executable configuration:

| Issue | This implementation |
| --- | --- |
| Equation (7) says softmax; MATLAB divides cubic values by their sum | `normalization="official"` defaults to the code; `"paper"` implements softmax. Neither renormalizes after adding noise. |
| Paper's prompt minimizes a scalar; current official prompt lists only two objectives and requests nondominance | Use the scalar prompt from Section IV-B, generalized with caller task and explicit normalized bounds. No four-decimal rounding. |
| Paper generates/evaluates sets of offspring; official Python randomly keeps one | Evaluate the requested usable set, with an exact evaluation cap. LO defaults to the code's one offspring, LLM to the paper's two. |
| Paper gives conflicting mutation probabilities; MATLAB uses `proM=1`, `disM=20`; Python defaults to a 0.9 per-offspring gate | Use `1/d`, index 20, and an explicit per-offspring gate defaulting to 1. Set `mutation_probability=0.9` for the Python gate. Python's per-coordinate cap at 0.5 is not reproduced for d=1. |
| MATLAB's mutation parentheses cancel the distribution-index powers | Use standard polynomial mutation, as implemented in the official repository's [pymoo PM](https://github.com/FeiLiu36/LLM4MOEA/blob/7fb29810abd891643914a0a7452c27956d38bddb/MOEAD-LLM/pymoo/operators/mutation/pm.py). For x=0.5 on [0,1], u=0.25, eta=20, the port gives 0.4675318; the MATLAB expression gives 0.25. This is a deliberate correction, not numerical equivalence. |
| Paper suggests T=N/10; MATLAB hardcodes 10; Python demo uses local probability 0.7 | Default T=10 and probability 0.9; both are configurable. |
| Official implementations lack a complete external population here | Maintain the nondominated external population required by Algorithm 1. |
| Official Python uses PBI above two objectives | Use the paper's Chebyshev decomposition for every objective count. |
| PlatEMO uses its own uniform-point generator and numerical RNG | Use a complete Das–Dennis lattice and Python's local RNG; do not promise identical trajectories. |

The port uses no vendored pymoo or MATLAB runtime. Attribution and upstream notices
are retained in [NOTICE.md](NOTICE.md). It is an algorithm implementation, not a
reproduction of the paper's HV/IGD tables, regression dataset, or live-model results.

## Checks

From the repository root, with Slick available:

```sh
python -m unittest tests.test_llm4moea
```

Tests use the shared `tests/providers.py` scripted provider for offline generation.
They cover numerical rules, objective directions, archive membership, exact budgets,
finite-score rejection, bounded generation failure, raw records, arbitrary bounds,
fixed coordinates, three objectives, custom weights, repeated runs, and template
loading from another directory. No paid model calls are needed.
