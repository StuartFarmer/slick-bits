# LLM_GP

`LLMGP(task, provider, evaluate, ...)` evolves arbitrary text through model
initialization, crossover, and mutation. `evaluate(content)` is an async callback
returning a finite score; lower is better unless `maximize=True`.

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

`llm-gp-mu-xo` uses two-member tournaments and one elite. `llm-gp` asks the model
to select parents, replace the population from parents plus offspring, and
designate a final candidate. Results retain that designation separately from
the measured best seen. Generations include initialization.

`max_calls` bounds generation requests. `evaluations` counts every assessment,
including cached candidates and elites. Evaluation is cached by candidate text,
so the callback must provide stable scores for the run. Invalid output consumes
a call: initialization tries again, variation retains parents, parent selection
samples with replacement, and replacement/final selection use score ordering.
Evaluator value/arithmetic errors and nonfinite scores reject a candidate and are recorded;
other evaluation errors, provider failures, and cancellation propagate. Budget exhaustion returns partial
results. Transport retries and raw response logging belong to the caller.

The flow follows the two LLM genetic-programming variants from Hemberg, Moskal,
and O'Reilly's *Evolving code with a large language model*. Generic task prompts
replace the symbolic-regression instructions. The traditional GP/random baselines,
AST interpreter, datasets, CLI, and accounting harness are in the
[legacy archive](../examples/legacy/README.md).

Run offline checks from the repository root:

```sh
python -m unittest tests.test_llm_gp
```

All tests use the single provider in `tests/providers.py`.

`run()` coordinates initialization, evolution, and final designation. Named phase
methods handle parent selection, variation, assessment, and generation snapshots.
Initialization, crossover, mutation, parent selection, replacement, and final
designation each have their own prompt method and template. Templates contain no
conditionals; Python chooses the operation.
