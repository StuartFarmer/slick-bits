# EvoPROMPT

Problem-agnostic implementation of **Connecting Large Language Models with
Evolutionary Algorithms Yields Powerful Prompt Optimizers** (ICLR 2024).
`EvoPrompt` evolves caller-supplied task instructions using a genetic algorithm
(GA) or differential evolution (DE). The class owns its task context, decorated
generation methods, and async selection loop. Supply a Slick provider and your
own async evaluator; there is no built-in dataset, evaluator, or runner.
`run` composes initialization, a GA or DE generation phase, and score snapshots.

```python
from pathlib import Path

from slick import prompts

import evoprompt
from evoprompt import EvoPrompt

# Configure Slick's process-global template root once before rendering.
prompts.TEMPLATE_ROOT = Path(evoprompt.__file__).parent / "prompts"

agent = EvoPrompt(task, provider, evaluate)
result = await agent.run(initial_prompts, algorithm="de", iterations=10)
best_instruction = result["best"]["prompt"]
```

`task` describes the problem and output requirements. `evaluate(instruction)`
must asynchronously return finite, nonnegative fitness, with higher scores
preferred. Set `cache=False` when repeated evaluation is intentionally noisy.
Generated instructions retain the textual `<prompt>...</prompt>` protocol:
decorated methods check that exactly one nonempty final prompt is present before
evaluation. Invalid output raises `ValueError` containing the raw response.
Provider and evaluator failures propagate without automatic retries.

- GA requires at least two members, samples parents with replacement using
  fitness-proportional roulette, and retains the best N parents and offspring.
  All-zero fitness falls back to uniform sampling. Stable ties prefer incumbents.
- DE requires at least three members, selects distinct donor indices excluding
  the target, and uses the current generation's best prompt. Each trial replaces
  its own target only on strict improvement. Every trial reads the same generation.
- A larger `population_size` fills missing members with checked variations of
  the supplied initial prompts. `seed` controls the local random generator.

The result includes the best and initial-best prompts, final population,
per-generation score history, generation/evaluation counts, and cache hits.
Pass `session=your_session` to `run` for an intentional sequential conversation;
otherwise every generation uses the injected provider independently. Different
applications need separate processes to use different template roots concurrently.
Each run resets the agent's search state; use one run at a time per instance.
Caller types and configuration are trusted. Supply a nonempty seed list and a
usable population size (GA: at least two; DE: at least three). When there are
more seeds than `population_size`, all are scored and the best N retained.
To minimize a loss or use signed rewards, map them to a nonnegative, increasing
fitness in your evaluator. Keep that mapping fixed throughout the search.

## Operations and budgets

Each operation has its own decorated method and local template:

- Initialization fills missing slots using `variation`.
- GA runs `crossover` then `mutate` for each offspring.
- DE runs `difference`, `mutate_difference`, `combine` with the best prompt,
  then `crossover` with the target. The two intermediate difference reports
  are nonempty prose; complete instructions use the tagged text contract.

Only completed offspring are evaluated. For N members and T iterations, GA
uses `2*N*T` generation operations and DE uses `4*N*T`, plus one per missing
initial member. With caching disabled and exactly N supplied seeds, both make
`N*(T+1)` evaluator calls. With caching enabled, exact duplicate instructions
reuse their scores. The optimizer never sees held-out test data: the evaluator
owns the development set, target model, metric, and any execution isolation.

`optimizer_calls` and `evaluations` count attempts **before** invocation, including
failed calls. Invalid output stops the run; there is no repair or retry loop.
The rejected raw response is included in the validation exception. Counters
remain inspectable on the agent after an exception. Transport retries belong
to the caller's provider. A supplied Session may make multiple model/tool turns
per generation operation; `optimizer_calls` does not count those internal turns.

Render an operation without generation using explicit owner binding:

```python
text = await EvoPrompt.crossover.render(agent, "first instruction", "second instruction")
```

`ga_offspring` and `de_offspring` are now ordinary async orchestration methods,
not decorated render boundaries. Use the individual operation methods to render.

## Official implementation and adaptations

Inspected and adapted the authors' [official repository](https://github.com/beeevita/EvoPrompt)
at commit `94caff336555df99acc5c338c7930c40cc550ad9`:

- [`evoluter.py`](https://github.com/beeevita/EvoPrompt/blob/94caff336555df99acc5c338c7930c40cc550ad9/evoluter.py):
  `GAEvoluter`, `DEEvoluter`, initial-population scoring, and fitness caching.
- [`data/template_ga.py`](https://github.com/beeevita/EvoPrompt/blob/94caff336555df99acc5c338c7930c40cc550ad9/data/template_ga.py):
  crossover followed by mutation.
- [`data/template_de.py`](https://github.com/beeevita/EvoPrompt/blob/94caff336555df99acc5c338c7930c40cc550ad9/data/template_de.py):
  the `v1` difference-only mutation, best-prompt combination, and target crossover.
- [`utils.py`](https://github.com/beeevita/EvoPrompt/blob/94caff336555df99acc5c338c7930c40cc550ad9/utils.py):
  final `<prompt>...</prompt>` extraction, made strict here.

The local operation templates adapt the official instructions; the accompanying
[MIT license](LICENSE) preserves attribution. No upstream runtime or benchmark
dependencies are needed.

This is an algorithm port, not an exact reproduction of the official experiment:

- Slick's separate-operation convention replaces one compound generation call
  with two for GA and four for DE. Task context replaces the official fixed
  sentiment/simplification demonstrations. This changes the generation
  distribution and cost; no equivalent quality claim is made.
- GA implements the paper's direct fitness-proportional parent sampling. The
  released code first samples a weighted temporary population, then selects
  pairs from it. This port preserves N slots even for duplicate prompts; the
  released GA uses a set union that can shrink the population. Ties are stable.
- DE preserves the existing local generation snapshot: all donors and the best
  prompt are fixed for that generation. The released code updates its best-so-far
  donor within a generation. Here only the two actual difference donors are
  sampled, excluding the target; upstream samples a third donor then overwrites
  it with the best prompt. Thus this version works with N=3 instead of N>=4.
- Python's local RNG replaces the upstream Python/NumPy global RNGs. The seed
  reproduces local selection for identical scores, not stochastic model output.
  Optional fitness caching also applies to GA offspring here; upstream reevaluates
  generated offspring. `cache=False` requests fresh measurements.

Run offline checks from the repository root with Slick installed:
`rtk proxy optimizer/.venv/bin/python -m unittest tests.test_evoprompt tests.test_prompt_layout`.
These deterministic checks establish search mechanics and Slick integration;
they do not reproduce benchmark scores or demonstrate prompt-quality gains.
The previous benchmark CLI and demo dataset are preserved in
`examples/legacy/2026-09-14-problem-specific.tar.gz`; historical `runs/` remain in place.
