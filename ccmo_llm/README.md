# CCMO-LLM

Problem-agnostic implementation of [Large Language Model-Aided Evolutionary
Search for Constrained Multiobjective Optimization](https://arxiv.org/abs/2405.05767)
(Wang et al., 2024), built with Slick. Supply bounded real decision variables,
an async evaluator, and a provider. No benchmark, provider, or execution service
is built into the optimizer.

```python
from pathlib import Path

import ccmo_llm
from ccmo_llm import CCMOLLM, Evaluation, Vector
from slick import prompts


async def evaluate(x: Vector) -> Evaluation:
    # Replace with your simulator, API, experiment, or decoded candidate evaluation.
    return Evaluation(
        objectives=(x[0] ** 2 + x[1] ** 2, (x[0] - 1) ** 2 + (x[1] - 1) ** 2),
        inequalities=(0.5 - x[0] - x[1],),  # g(x) <= 0
        equalities=(),                     # abs(h(x)) <= equality_tolerance
    )


async def optimize(provider):
    # Configure once at application startup, before rendering or model calls.
    prompts.TEMPLATE_ROOT = Path(ccmo_llm.__file__).resolve().parent / "prompts"
    search = CCMOLLM(
        task="Minimize both costs subject to x[0] + x[1] >= 0.5.",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        objectives=2,
        evaluate=evaluate,
        provider=provider,
    )
    result = await search.run(population_size=100, max_evaluations=10_000, seed=7)
    return result.front
```

Run the caller's coroutine with its normal async entrypoint. Import from the
repository root, or put the repository on `PYTHONPATH`. The absolute template
root also works when the caller launches from another directory. Slick's root
is process-global; configure it outside the agent, without changing it during
concurrent generation. No Session is used: every offspring request is independent.
Dependencies are Slick and the Python standard library.

All objectives are **minimized**; negate a metric you want to maximize. The
evaluator receives an immutable tuple in original units and returns objective
values and raw constraint residuals. It owns domain decoding, execution isolation,
and timeouts. Numerical encodings can represent other domains, but native text,
permutation, or categorical operators are outside this implementation. Infeasible
points should return their residuals, not raise `CandidateRejected`.

The result contains the constrained `population`, unconstrained `auxiliary`,
feasible nondominated `front` from the final constrained population, evaluator
call count, and model call count. The front is empty if no final point is feasible;
it is not an all-time archive. Inspect `search.attempts` for raw model responses
and rejection reasons, `measurements` for every evaluator call, and `generations`
for survival snapshots. These are in-memory records; persistence belongs to the
caller. Runs reset state; use separate instances for concurrent runs.

## Algorithm and official source

No author-released CCMO-LLM repository was located, and the supplied paper does
not link one. The LLM operator is implemented from sections 3.2–3.3 and Figure 2.
The numerical backbone is adapted from the official
[BIMK/PlatEMO](https://github.com/BIMK/PlatEMO) platform used in the experiments,
pinned to commit `d25e65d1ffba58dbf4d7e1b5259786187d12968a`:

- [CCMO.m](https://github.com/BIMK/PlatEMO/blob/d25e65d1ffba58dbf4d7e1b5259786187d12968a/PlatEMO/Algorithms/Multi-objective%20optimization/CCMO/CCMO.m): two independent populations of size N, binary mating tournaments, and shared offspring.
- [CalFitness.m](https://github.com/BIMK/PlatEMO/blob/d25e65d1ffba58dbf4d7e1b5259786187d12968a/PlatEMO/Algorithms/Multi-objective%20optimization/CCMO/CalFitness.m): dominance strength plus kth-neighbor density in unnormalized objective space, with k=floor(sqrt(population count)).
- [EnvironmentalSelection.m](https://github.com/BIMK/PlatEMO/blob/d25e65d1ffba58dbf4d7e1b5259786187d12968a/PlatEMO/Algorithms/Multi-objective%20optimization/CCMO/EnvironmentalSelection.m): SPEA2 survival and lexicographic distance truncation, retaining union fitness for mating.
- [OperatorGAhalf.m](https://github.com/BIMK/PlatEMO/blob/d25e65d1ffba58dbf4d7e1b5259786187d12968a/PlatEMO/Algorithms/Utility%20functions/OperatorGAhalf.m): real-valued simulated binary crossover, one child per pair, clipping, and the reference mutation arithmetic; defaults proC=1, disC=20, proM=1, disM=20.

For N=100, initialize 100 constrained and 100 auxiliary individuals. Each
generation each population supplies 45 GA offspring and five LLM offspring,
using ten randomly sampled parent entries without replacement for its prompt.
Each population selects N survivors from its own incumbents plus all 100
offspring. The constrained population compares total violation first and Pareto
dominance at equal violation; the auxiliary population ignores violations during
mating and survival. Both LLM prompts include actual constraint violations, as
described in the paper. No objective scalarization or benchmark knowledge is used.

The paper's “fine-tuning” is in-context prompting here, not weight training.
One local `propose.j2` contains its task, feasible/infeasible observations,
two-parent generation instructions, and `<start>...<end>` output format. GA
initialization, crossover, and mutation are numerical operations, not additional
LLM operations.

## Explicit adaptations and failure policy

- **Budget:** count both initial populations and every subsequent evaluator call;
  never exceed `max_evaluations`. A budget exhausted before initialization completes
  raises. This resolves Algorithm 1's uncounted initialization and inclusive loop.
  Full generations use N offspring; a final partial generation divides its remaining
  quota between both populations, with the extra slot going to the constrained one.
- **Rounding:** each population's LLM quota is `int(quota * llm_fraction + 0.5)`.
  Use N divisible by 20 for the exact paper ratio; tiny populations or partial
  generations can have no LLM offspring. `llm_fraction=0` permits an offline CCMO
  baseline with no provider. The other defaults reproduce the stated 90/10 split.
- **Equality tolerance:** use `sum(max(0,g)) + sum(max(0,abs(h)-delta))`, with
  `delta=1e-4` by default. The printed `abs(h-delta)` would penalize h=0 and is
  inconsistent with a symmetric equality relaxation.
- **Reference mutation:** use GAhalf's polynomial mutation with distribution index
  20 and probability 1/d per coordinate, verified against numerical reference values.
  Fixed coordinates skip mutation to avoid division by zero. Python RNG ordering
  differs from MATLAB's vectorized draws; matching seeds do not imply matching runs.
- **Prompt adaptation:** dimensions, bounds, task, objective count, and requested
  batch size are caller-dependent. Bounds and multiobjective trade-offs are stated
  explicitly. The delimiter format and four-part prompt structure are preserved.
- **Generated output:** validate exact batch size, coordinate count, finite values,
  bounds, and distinct vectors before any evaluation. Reject the whole batch on
  failure and retry the same prompt up to `max_attempts` (default three). Exhaustion
  raises with raw responses retained; no silent GA replacement or clipping of LLM
  outputs. Novelty is checked against current populations, sampled parents, and
  already generated offspring, not every historical point.
- **Evaluation failures:** nonfinite/wrong-sized objectives, nonfinite residuals,
  `CandidateRejected`, and evaluator `TimeoutError` consume an evaluation and
  discard the offspring. Failure on an initial point aborts. Unexpected evaluator
  errors and provider errors are logged and propagated. Transport retries, if any,
  belong to the supplied provider and are outside the algorithm's model-call count.

## Checks

From the repository root, using an environment containing Slick:

```sh
python -m unittest tests.test_ccmo_llm -v
```

Tests use the repository's shared scripted provider. They check reference fitness,
constraint handling, truncation, GA arithmetic, shared offspring, split and budget,
rejection records, evaluator errors, seeded repeatability, fixed bounds, and template
rendering from another directory. They establish offline algorithm/interface
behavior, not replication of the reported IGD/HV results or LLM convergence gains.

## Attribution

The numerical adaptation uses PlatEMO code, copyright (c) 2026 BIMK Group,
provided for research purposes. See [NOTICE](NOTICE) for the upstream notice.
Publications using this implementation should acknowledge PlatEMO and cite:

- Zeyi Wang, Songbai Liu, Jianyong Chen, and Kay Chen Tan. *Large Language
  Model-Aided Evolutionary Search for Constrained Multiobjective Optimization*, 2024.
- Y. Tian, T. Zhang, J. Xiao, X. Zhang, and Y. Jin. *A Coevolutionary
  Framework for Constrained Multiobjective Optimization Problems*, IEEE Transactions
  on Evolutionary Computation 25(1), 102–116, 2021.
- Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin. *PlatEMO: A MATLAB Platform
  for Evolutionary Multi-Objective Optimization*, IEEE Computational Intelligence
  Magazine 12(4), 73–87, 2017.
