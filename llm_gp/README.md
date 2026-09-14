# LLM_GP

`LLMGP(task, provider, evaluate, ...)` evolves code or arbitrary text through model
initialization, crossover, and mutation. `evaluate(content)` is an async callback
returning a finite score; lower is better unless `maximize=True`. Put the language,
interface, allowed primitives, constraints, and objective in `task`. The optimizer
preserves candidate whitespace and never executes candidate code. Your evaluator
owns execution isolation, timeouts, domain validation, and held-out data.

```python
from pathlib import Path
from slick import prompts
import llm_gp

async def optimize(task, provider, evaluate):
    prompts.TEMPLATE_ROOT = Path(llm_gp.__file__).resolve().parent / "prompts"
    agent = llm_gp.LLMGP(task, provider, evaluate, population_size=10, generations=30)
    return await agent.run()
```

Use a fresh instance for each run. Caller inputs follow the annotated types; native
errors surface when an operation uses an incompatible value. An optional `run(session=session)` carries sequential
conversation history; otherwise generation calls use the injected provider.
The application configures the process-global template root before generation.

`llm-gp-mu-xo` uses two-member tournaments with distinct competitors. It evaluates
a full population of offspring, adds the best old individual, then retains the
best `population_size` candidates. `llm-gp` asks the model
to select parents, replace the population from parents plus offspring, and
designate a final candidate. Results retain that designation separately from
the measured best seen. Generations include initialization.

`max_calls` bounds decorated generation requests, including malformed responses.
Session tool turns and provider transport retries can make additional model calls;
the caller owns those budgets. `evaluations` counts every candidate assessment,
including cache hits; carrying an elite into replacement adds no assessment.
For a completed run with no rejected candidates, it equals `population_size * generations`.
Evaluation is cached by exact candidate text,
so the callback must provide stable scores for the run. Invalid output consumes
a call: initialization tries again, variation retains parents, parent selection
samples with replacement, and replacement/final selection use score ordering.
Evaluator value/arithmetic errors and nonfinite scores reject a candidate and are recorded;
other evaluation errors, provider failures, and cancellation propagate. Budget exhaustion returns partial
results. Transport retries and raw response logging belong to the caller.

The flow follows the two demonstrated LLM genetic-programming variants from Hemberg,
Moskal, and O'Reilly's [*Evolving code with a large language model*](https://arxiv.org/abs/2401.07102)
(2024), sections 3 and 5 and Appendices 2–3.
The official implementation is [ALFA-group/Tutorial_GP-LLM](https://github.com/ALFA-group/Tutorial_GP-LLM),
inspected at commit `e3b3c52bfc21b3081ddd780bc2373ae05a699757`.
This implementation adapts the tournament and elitist replacement decisions from
[`evolutionary_algorithm.py`](https://github.com/ALFA-group/Tutorial_GP-LLM/blob/e3b3c52bfc21b3081ddd780bc2373ae05a699757/alfa_ec_llm/algorithms/evolutionary_algorithm.py),
and the generation loop and variation fallbacks from
[`tutorial_llm_gp.py`](https://github.com/ALFA-group/Tutorial_GP-LLM/blob/e3b3c52bfc21b3081ddd780bc2373ae05a699757/alfa_ec_llm/algorithms/tutorial_llm_gp.py).
The upstream MIT notice is retained in [LICENSE](LICENSE).

The pinned repository implements the Mu/XO variant; the additional model selection,
replacement, and designation operators here implement the paper's descriptions.
Generic task prompts deliberately replace the symbolic-regression prompts from
[`symbolic_regression.py`](https://github.com/ALFA-group/Tutorial_GP-LLM/blob/e3b3c52bfc21b3081ddd780bc2373ae05a699757/alfa_ec_llm/problem_environments/symbolic_regression.py).
Compared with upstream, each pair gets fresh tournaments, crossover and mutation
run pair by pair, and few-shot examples come from the current population instead
of the entire fitness cache. Model selection/replacement use explicit scored IDs.
Invalid initial candidates are retried within `max_calls` because no universal
default candidate exists. Invalid measured candidates are rejected and another
candidate is attempted. These adaptations change RNG and failure trajectories;
this is not a bit-for-bit reproduction. Numerical fitness stays with the evaluator;
the hypothetical LLM execution and fitness operators of Algorithm 1 are omitted,
as in the paper's demonstrated variants.

The traditional GP/random baselines,
AST interpreter, datasets, CLI, and accounting harness are in the
[legacy archive](../examples/legacy/README.md).

Run offline checks from the repository root:

```sh
../slick/.venv/bin/python -m unittest tests.test_llm_gp tests.test_prompt_layout
```

This uses the workspace's adjacent Slick installation (checked against 0.3.0).
Alternatively, install Slick in your environment with `pip install -e ../slick`
from the repository root and use `python` for the command above.
All tests use the single provider in `tests/providers.py`. Offline tests establish
algorithm and Slick integration behavior; no live model runs or paper benchmark
results have been reproduced.

`run()` coordinates initialization, evolution, and final designation. Named phase
methods handle parent selection, variation, assessment, and generation snapshots.
Initialization, crossover, mutation, parent selection, replacement, and final
designation each have their own prompt method and template. Templates contain no
conditionals; Python chooses the operation.
