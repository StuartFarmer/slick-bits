# SEED

Task-agnostic implementation of **SEED: Domain-Specific Data Curation With Large
Language Models**, using Slick for generation. The algorithm combines evolved
code, semantic cache reuse, a distilled model, and LLM fallback; it chooses their
configuration and order against a caller-defined validation metric.

Based on the [paper](https://arxiv.org/abs/2310.00749) and the verified
[official implementation](https://github.com/Magolor/SEED). See
[SOURCES.md](SOURCES.md) for the pinned source, algorithm mapping, and differences.

## Use

Configure the process-global template root once in your application. Importing
`seed` does not change it. The existing `optimizer/.venv` contains Slick and NumPy;
SciPy is needed only for `PRX` and `DIV` batching.

```python
from pathlib import Path
from slick import prompts
import seed
from seed import SEED, Config, Example, Prediction

prompts.TEMPLATE_ROOT = Path(seed.__file__).resolve().parent / "prompts"

async def exact_accuracy(examples, predictions):
    return sum(
        p is not None and p.value == e.output
        for e, p in zip(examples, predictions)
    ) / len(examples)

async def curate(task, provider, validation, inputs, *, examples=(),
                 embed=None, execute_code=None, train_model=None, tools=()):
    compiler = SEED(
        task=task,
        provider=provider,
        evaluate=exact_accuracy,
        embed=embed,
        execute_code=execute_code,
        train_model=train_model,
        tools=tools,
        config=Config(gap=0.05, batch_size=16),
    )
    result = await compiler.run(validation, examples=examples)
    predictions = await compiler.predict(inputs)
    return [prediction.value for prediction in predictions], result, compiler
```

`task` describes the problem, serialized input format, required JSON output, domain
knowledge, and permitted code dependencies. Inputs are arbitrary strings, including
JSON-serialized records. Outputs can be strings, numbers, booleans, arrays, objects,
or null. There are no benchmark-specific fields or task categories in the agent.

Pass development examples as `Example(input, output)`. `validation` is a nonempty,
representative sequence of held-out examples. If you have no labels, call
`compiler.label(validation_inputs)` first and build validation `Example`s from
those predictions; the resulting metric measures agreement with pseudo-labels,
not ground-truth accuracy. `run()` removes these inputs from the training cache.
Use `run(..., unlabeled=training_inputs)` to obtain additional LLM training labels.

## Problem-specific primitives

| Argument | Contract |
| --- | --- |
| `evaluate(examples, predictions)` | Async; finite higher-is-better score in `[0, 1]`, e.g. accuracy or F1. In code verification it receives only answered examples; optimizer evaluation uses the full validation set. |
| `embed(inputs)` | Async; one finite, nonzero vector per serialized input. Use a frozen text encoder. Enables CacheReuse, nearest examples, and embedding-based batching. |
| `execute_code(source, inputs)` | Async; execute generated code in a caller-owned isolated worker and return one `Prediction` or `None` per input. Enables CodeGen. |
| `train_model(examples)` | Async; fit on the provided development/LLM labels and return an async predictor `predict(inputs) -> Sequence[Prediction | None]`. Enables ModelGen after `min_train` labels. Return a fresh model snapshot when refitting. |
| `validate_output(value)` | Optional synchronous validation/transformation of generated LLM answers; raise on invalid task outputs. |
| `matches(expected, predicted)` | Optional per-case correctness test for code coverage; defaults to canonical JSON equality, distinguishing booleans from integers. |
| `tools` | Caller-supplied Slick tools, available only to the query session. |

Generated code defines `solve(input: str)` and returns `{"value": answer}` or `None`
to abstain. The isolated execution adapter unwraps the result into
`Prediction(result["value"])`. `Prediction(None)` is a legitimate null answer;
bare `None` always means abstention. AST checking establishes syntax/interface
only. A timeout bounds awaiting a worker; the worker must terminate its own process
and clean up resources on cancellation. No generated source executes in this package.

The fitting primitive can use the paper's T5 encoder plus classification/regression
head, or a sequence-to-sequence model for arbitrary JSON outputs. Training libraries,
checkpoint downloads, and hardware allocation belong to the caller.
`classification_confidence(probabilities)` implements `(K * max(p) - 1)/(K - 1)`;
`generation_confidence(token_log_probabilities)` implements inverse perplexity.
Regression needs a caller-calibrated confidence measure; the paper supplies none.

## Algorithm and controls

- `SeedOptimizer` supports `specialized` (default), `generic`, and `exhaustive`
  search. It caches each immutable module configuration's validation predictions,
  evaluates whole pipelines with your metric, and selects minimum estimated cost
  within `gap` of the best effectiveness on the retained frontier.
- Specialized search visits LLM, CodeGen, ModelGen, CacheReuse and orders modules
  by `(1 - fallback_probability) / cost`. Generic search enumerates positions,
  retaining skylines within each used-module subset. Exhaustive mode retains all
  subplans and is useful for checking small grids.
- `Module(key, family, cost, predict, threshold=None)` lets you use the optimizer
  directly with your own implementations. Keys identify immutable configurations;
  change them whenever predictions or costs change. Families are `LLM`, `CodeGen`,
  `ModelGen`, `CacheReuse`; each plan uses at most one configuration per family.
- CodeGen generates advice, source, and repairs with separate local templates.
  Every parent branches on up to `max_errors` individual failures and the full
  failure set. It filters costly/dominated candidates and ranks by precision and
  coverage. `code_branches`, `preserved_branches`, `evolution_iterations`, and
  `max_codegen_calls` bound this process. It generates development cases if none
  are supplied. `ensemble` is `vote` by default, or `weighted`/`sequential`.
- CacheReuse uses normalized cosine distance and an exact-key shortcut. Threshold
  grids are `distance_thresholds` and `confidence_thresholds`. CodeGen and LLM
  measurements persist across reoptimization; adaptive module keys are versioned.
- `batching` supports `RND`, `DIV`, `PRX`, `FAR`, and `CLS`, preserving caller order
  in returned results. `sample_modes` searches `random`, `balanced`, or `nearest`
  few-shot selection. `nearest` and nonrandom batching require `embed`.
- During `predict()`, new LLM answers enter the cache. Before a subsequent batch,
  `reoptimize_every` new labels trigger refitting and optimization. Cache/model
  snapshots refresh at that boundary. Threshold search centers on the previous
  selection with `local_radius` neighboring grid positions. `reoptimize()` forces
  a refresh immediately, including after the final batch of a workload.

Every plan retains an LLM fallback, even when its reach is zero. This deliberate
adaptation avoids silently inventing a task-specific default for unseen records.
Missing optional primitives disable their modules. LLM-only operation works with
just `task`, `provider`, `evaluate`, and validation examples.

## Evidence and failure policy

One instance owns one task, fixed validation split, and sequential stream. Use a
fresh instance for another experiment. Validation labels never enter code prompts,
training, or the reusable label cache; they are used only to score plans. Keep a
separate test set for reporting generalization after repeated validation search.

`history` records optimization results; `codes` exposes the final sources and
development profiles; `attempts` preserves raw Slick session snapshots even on
rejection. `code_evaluations`, `runtime_failures`, and `routing` record verification
and execution activity. `llm_records` counts attempted annotation/query records
across compilation, validation, and inference; it is not a provider-request count.
Batching and tools change the number of physical requests.

Each prompt call owns a fresh session. Query sessions have at most 12 tool turns;
code prompts have no supplied tools. Generation parsing errors and explicit
`CandidateRejected` failures consume code-generation budget. Code execution
rejections/timeouts abstain. Provider failures, invalid final answers, invalid
measurements, and unexpected evaluator errors propagate; there are no hidden retries.

The effectiveness gap is an empirical validation constraint. Generic pruning's
optimality requires the paper's descendant-dominance assumption. Specialized gap
pruning, a finite beam, and local threshold search are approximations. Equation 1
assumes independent fallback probabilities; its estimated cost can differ from
observed routing when modules' failures are correlated. Costs exclude compilation
and training and must share units. Defaults treat one LLM-processed record as cost 1.

## Checks

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_seed
../slick/.venv/bin/ruff check seed tests/test_seed.py
../slick/.venv/bin/ruff format --check seed tests/test_seed.py
```

These deterministic checks exercise algorithm decisions and Slick boundaries.
No paid LLM calls or model training were run, and no paper benchmark result is claimed.
