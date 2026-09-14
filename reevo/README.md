# ReEvo with Slick

An implementation of **ReEvo: Large Language Models as Hyper-Heuristics with
Reflective Evolution**, by Ye et al. The search follows Section 4 and the
Appendix B prompts in the supplied paper; the
[authors' implementation](https://github.com/ai4co/reevo/blob/main/reevo.py)
clarifies population replacement and post-crossover elite updates.

Slick renders the generator and reflector prompts. Python owns the population,
fitness evaluation, selection, memory, and stopping conditions. The runnable
example evolves NumPy heuristic measures for a small TSP Ant Colony Optimization
(ACO) solver. It supports both white-box and black-box prompting.

## Run

From `slick-bits/`, using the neighboring Slick checkout:

```sh
python3 -m venv reevo/.venv
source reevo/.venv/bin/activate
pip install -e ../slick 'numpy>=1.26,<3'
python -m reevo --max-evaluations 12 --initial-size 4 --population-size 4 \
  --nodes 12 --ants 8 --aco-iterations 5 --output runs/reevo-demo
python -m reevo.test_reevo
```

The default `demo` provider returns canned heuristic formulas and reflections
through the actual Slick prompts. The search and TSP evaluations are real;
this demonstrates the implementation, not LLM discovery or paper-level results.
Use `--black-box` to hide the COP name and attribute meanings from both LLM roles.

For real generation, install Slick's optional LiteLLM dependency and configure
credentials for the selected provider:

```sh
pip install -e '../slick[litellm]'
python -m reevo --provider litellm --model YOUR_PROVIDER/YOUR_MODEL \
  --reflector-model YOUR_PROVIDER/YOUR_REFLECTOR --output runs/reevo-live
```

The reflector defaults to the generator model. Temperature settings apply
to LiteLLM. CLI help lists all settings: `python -m reevo --help`.

**Execution boundary:** the TSP example executes generated Python in a child
process with a fresh temporary directory, stripped credential environment,
bounded output files, and a wall-clock timeout. This is **not a security sandbox**;
it does not prevent filesystem reads/writes or network access. Run live searches
inside your own isolated environment. The generic search accepts an evaluator
callback, so it can use an existing remote or sandboxed evaluator instead.
The bundled subprocess evaluator targets macOS/Linux.

## What is implemented

1. Evaluate an optional seed; generate and evaluate the initial population.
2. Sample valid parent pairs uniformly, excluding equal objective values.
3. Reflect on each worse/better pair, then generate crossover offspring.
4. Evaluate offspring and update the all-time elite.
5. Summarize prior memory plus new insights into long-term reflection.
6. Generate mutations of that elite, evaluate them, and replace the population
   with crossover offspring plus mutations. Include the elite in future selection.

| Setting | Search default |
| --- | --- |
| Initial generations | 30, plus an optional seed evaluation |
| Population size / crossover offspring | 10 |
| Evaluation cap | 100, including the seed and invalid candidates |
| Crossover / mutation rate | 1.0 / 0.5 |
| Generator / reflector temperature | 1.0 via LiteLLM |
| Initial generator temperature | 1.3 via LiteLLM |

Offspring counts are `floor(population_size * rate)`. Each batch is truncated to
the remaining evaluation budget. A population with no valid individuals or no
unequal-fitness parent pair stops explicitly. Provider failures abort and are
recorded; syntax, signature, runtime, shape, nonfinite-score and timeout failures
mark a candidate invalid. Invalid candidates cannot become parents or elites.

`--no-short-reflection`, `--no-long-reflection`, `--no-crossover` and
`--no-mutation` expose component ablations. Without crossover there are no new
pair reflections; mutation uses the existing guidance. Without short reflections,
long-term memory receives no new insights. For independent LLM sampling, set
`initial_size` equal to `max_evaluations` and call `run()` without a seed.

## Use with another evaluator

```python
from reevo.core import Config, ReEvo, Task

task = Task(
    description="Your optimization problem, or a black-box description.",
    function_description="Inputs, outputs, and constraints for your heuristic.",
    signature="def heuristic(values):",
)

# evaluate is async: evaluate(source_code: str) -> finite float.
# It should measure mean performance on a fixed validation dataset.
search = ReEvo(task, evaluate, generator_provider,
               reflector=reflector_provider, initializer=initializer_provider,
               config=Config(max_evaluations=100, maximize=False))
result = await search.run(seed_code=optional_seed_code)
if result.best is not None:
    print(result.best.code, result.best.objective)
```

Providers implement Slick's `acall` contract. The initializer defaults to the
generator; supply a separate provider to control its temperature. Code may use
imports, helpers and ordinary Python syntax. The function's name, argument kinds,
names and defaults must match the task signature; annotations are optional.
Evaluation determines behavioral validity. Use a new `ReEvo` instance per run.

## Evaluation and saved artifacts

The example generates Euclidean TSP instances from uniform unit-square points.
Every candidate gets identical validation instances and ACO random seeds.
The objective is the mean best closed-tour length. ACO uses pheromone and heuristic
exponents of 1, evaporation of 0.1, all-ant inverse-length deposits, and uniform
fallback when all feasible heuristic weights are zero. Inputs are copied before
calling generated heuristics. Black-box inputs flatten the distance matrix to
`(n_edges, 1)`; output is `(n_edges,)`, matching the Appendix B interface.

CLI defaults use a small example: 20 nodes, 3 validation instances, 8 ants and
10 ACO iterations. After search, the selected heuristic and an inverse-distance
baseline are evaluated on the same 8 held-out instances from a different seed.
Held-out scores never enter prompts or selection and do not count toward the
search budget. These are raw tour lengths, not optimality gaps.

The output directory must be new. It contains `best.py`, individual candidate
files, and `result.json` with objectives, errors, parent IDs, best-so-far history,
short/long reflections, rendered prompts, raw replies, and run settings. Partial
results are saved on Python exceptions and cancellation; abrupt process death
is not recoverable. No resume facility is implemented.

This implements the core method and TSP example, not the six-problem benchmark
suite, trained neural models, Appendix E heuristic collection or fitness-landscape
experiments. Calls are sequential. Prompts use a stable function name instead of
`_v0/_v1/_v2` renaming. Long-term carried memory is capped at 49 words. Selection
follows the paper's valid/unequal rule without the reference repository's extra
black-box filter for beating the seed. The ACO example is a compact Ant System,
not a port of DeepACO; reproducing the published tables needs the original
solvers, datasets and evaluation configurations.

## Prompt templates

Model instructions live in this implementation's own `prompts/` directory.
The CLI sets Slick's absolute template root once at startup. Programmatic callers
configure it before rendering or running the algorithm:

```python
from pathlib import Path

from slick import prompts
import reevo.prompts as operations

prompts.TEMPLATE_ROOT = Path(operations.__file__).resolve().parent / "prompts"
```

The root is process-global; run implementations with different roots in separate
processes. Prompt wording, output parsing, and caller-owned validation are unchanged.
