# MetaGen

Problem-agnostic implementations of the population Random Search (§4.1) and
Simulated Annealing (§4.2) algorithms in
[MetaGen: A framework for metaheuristic development and hyperparameter optimization
in machine and deep learning](https://doi.org/10.1016/j.neucom.2025.130046).
Supply a domain and an async scalar fitness evaluator; lower is better. To
maximize a utility, return its negative.

The actual `Domain`, `Solution`, initialization, mutation, and connector machinery
come from the [official implementation](https://github.com/Data-Science-Big-Data-Research-Lab/MetaGen),
pinned at `74f104e991cf7f307999adc47786722c15b04700`. The folder is named `meta_gen`
to avoid shadowing the dependency's `metagen` package. No domain representation is
reimplemented here.

Install from this repository's root with Python 3.10+:

```sh
python -m pip install -r meta_gen/requirements.txt
python -m unittest tests.test_meta_gen -v
```

The dependency declares NumPy and SciPy. No model, provider, prompts, session, Ray,
TensorBoard, or training library is needed for these numerical search loops.
They use ordinary async Python, consistent with the repository's numerical
optimizers; Slick generation would change the algorithm. Callers may use Slick
inside their own evaluator.

## Point it at a problem

```python
import asyncio
import random

from meta_gen import Domain, RandomSearch, SimulatedAnnealing, Solution

domain = Domain()
domain.define_integer("count", 1, 20)
domain.define_real("scale", 0.0, 1.0)
domain.define_categorical("mode", ["fast", "thorough"])


async def evaluate(candidate: Solution) -> float:
    # Replace this objective with your simulation, model evaluation, API, etc.
    return (candidate["count"] - 7) ** 2 + (candidate["scale"] - 0.3) ** 2


async def main():
    random.seed(42)
    search = RandomSearch("Minimize my cost", domain, evaluate)
    best = await search.run(population_size=30, iterations=20)
    print(best["count"], best["scale"], best.fitness, search.evaluations)

    annealing = SimulatedAnnealing("Minimize my cost", domain, evaluate)
    best = await annealing.run(
        iterations=100, alteration_limit=1, initial_temp=5.0, cooling_rate=0.95
    )
    print(best, annealing.evaluations)


asyncio.run(main())
```

`task` records caller context; numerical proposals are determined by the domain,
not by interpreting this string. The evaluator receives an independent deep copy
of each official `Solution`. Basic values are accessed with `candidate["name"]`;
group access yields a mapping whose basic members are MetaGen value objects
(use `.get()` when a downstream API needs a primitive). Groups within structures
are `Solution` objects with primitive bracket access. Basic structure elements
are value objects as well. Keep any code execution, model construction, data
splits, resource cleanup, or execution sandbox in the evaluator.

## Structured search spaces

All six paper variable types are supplied by the official framework: integer,
real, categorical, group, static structure, and dynamic structure. For example,
append a variable-length sequence of arbitrary stages to the domain above:

```python
domain.define_dynamic_structure("stages", 1, 6)
domain.define_group("stage")
domain.define_integer_in_group("stage", "capacity", 1, 100)
domain.define_categorical_in_group("stage", "operation", ["a", "b", "c"])
domain.define_real_in_group("stage", "weight", 0.0, 1.0)
domain.set_structure_to_variable("stages", "stage")

domain.define_static_structure("offsets", 3)
domain.set_structure_to_real("offsets", -1.0, 1.0)

# Inside evaluate(candidate):
# for stage in candidate["stages"]:
#     use(stage["capacity"], stage["operation"], stage["weight"])
# offsets = [value.get() for value in candidate["offsets"]]
```

Binding the group to the structure removes its standalone variable by default.
Both optimizers obtain the solution type through `domain.get_connector()`, so
custom solution representations can be registered using official `BaseConnector`.
Domain definition checks remain upstream; caller configuration is trusted here.

## Algorithm contract and source differences

| Operation | Behavior here |
| --- | --- |
| Random Search initialization | Generate and evaluate `population_size` solutions. Save the best copy. |
| Random Search iteration | Mutate and evaluate every population member, accepting even worse mutations in that member. Update the separate best only on strict improvement. |
| Annealing initialization | Generate and evaluate one solution. |
| Annealing iteration | Mutate a copy of the current solution. Accept lower/equal fitness, or a worse move with probability `exp((current - neighbor) / temperature)`. Multiply temperature by `cooling_rate`. |
| Return | An independent copy of the best observed solution, with its measured `.fitness`. |
| History | Best fitness after initialization and each completed iteration. |
| Evaluation count | RS: `population_size * (iterations + 1)`; SA: `iterations + 1`. Includes attempted evaluator calls on failure. No hidden retries or warmup evaluations. |

The inspected official sources are
[Random Search](https://github.com/Data-Science-Big-Data-Research-Lab/MetaGen/blob/74f104e991cf7f307999adc47786722c15b04700/src/metagen/metaheuristics/rs/random_search.py),
[SA](https://github.com/Data-Science-Big-Data-Research-Lab/MetaGen/blob/74f104e991cf7f307999adc47786722c15b04700/src/metagen/metaheuristics/sa/sa.py),
[solution operations](https://github.com/Data-Science-Big-Data-Research-Lab/MetaGen/blob/74f104e991cf7f307999adc47786722c15b04700/src/metagen/framework/solution/base_solution.py),
and [domain methods](https://github.com/Data-Science-Big-Data-Research-Lab/MetaGen/blob/74f104e991cf7f307999adc47786722c15b04700/src/metagen/framework/facades.py).

- RS follows the paper's mutate-all loop. The pinned official RS instead preserves
  an elite in its population and evaluates only `N - 1` mutations per round.
- SA evaluates exactly the requested number of neighbors, correcting the paper's
  `while iteration <= n_iterations` extra step. It returns the best seen solution
  rather than the paper snippet's terminal accepted state; that state remains
  available as `agent.current`. `agent.accepted` records each acceptance decision.
  There is no release-specific warmup or multi-neighbor extension.
- SA avoids evaluating an overflowing positive exponential for improvements and
  treats temperature underflow to zero as greedy search. Ties are accepted without
  consuming an acceptance random draw.
- Initialization passes the domain and connector; the paper's `Solution()` snippet
  omits a required argument. Mutation uses the real implementation: by default a
  random **subset** of top-level variables, rather than exactly one variable as
  suggested by the paper's prose.
- `alteration_limit` is passed unchanged to upstream. It is an absolute numeric
  radius, not a normalized fraction; integer moves are converted to integer bounds,
  categorical moves ignore it, and structural resizing follows upstream rules.
  Use a suitable radius for your units, or `None` for full-range moves. Mixed-scale
  or specialized constraints can use custom connector types.

## Limits and verification

Evaluations are sequential. Exceptions, cancellation, and nonfinite measured
fitness abort the run and propagate; they are not silently replaced with penalties.
If an infeasible candidate deserves a penalty, the evaluator should return a
deliberate finite cost. Use one agent per active run; a later `run()` reinitializes
its state.

The official types use Python's global `random` state and mutate a set of selected
variable names. Seed `random` and set `PYTHONHASHSEED` **before process startup** for
repeatable runs across processes. Other code consuming global randomness,
including your evaluator, changes trajectories; use separate processes for
independent runs. No reproducibility claim is made for arbitrary custom connectors.

Upstream representation behavior is preserved, including its limitations: dynamic
initialization uses an exclusive upper length bound (mutation can reach the upper
bound), empty structures can fail during mutation, and numeric step/radius handling
is upstream's rather than a new lattice implementation. This adapter does not
repair or replace the official framework.

This folder focuses on the two algorithms explicitly developed in the paper.
TPE, Tabu, GA, steady-state GA, Memetic, and CVOA are **not reimplemented here**;
they remain in the installed official `metagen.metaheuristics` package with its
own synchronous interfaces and optional dependencies. Reporting and distributed
execution are also left upstream.

The tests check evaluation counts, mutation population behavior, best snapshots,
annealing acceptance and temperature underflow, evaluator isolation and errors,
and all six domain types with custom connector construction. These are algorithm
and integration checks, not reproduction of the paper's optimization benchmarks
or evidence of superior performance.

The upstream source headers and repository license specify GPL-3.0-or-later,
despite an inconsistent MIT classifier in its packaging metadata. This adaptation
retains attribution and the [GPL license text](LICENSE).
