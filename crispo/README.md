# CriSPO

Task-agnostic Slick implementation of [CriSPO](https://arxiv.org/abs/2410.02748),
including multi-metric Automatic Suffix Tuning (AST).

`CriSPO` evaluates a seed template on training and development examples. For each
new template, it samples training predictions for one multi-aspect critique and
suggestion call. The receptive optimizer sees the top-K training templates,
their measured scores, and their critiques, ordered weakest to strongest.
It synthesizes suggestions before returning a new template. Development scores
select the returned prompt; development inputs, references and scores never
enter the critique or revision prompts. Test evaluation belongs to the caller.

## Use with your task

Supply an async task-model call and an async corpus evaluator. The evaluator
returns named numerical metrics; **higher is better for every metric**. Negate
losses. Metric names should describe their meaning to the optimizer; put any
additional rubric in `task`. No benchmark, model, retrieval system or metric
library is built into the algorithm.

```python
from pathlib import Path

import crispo
from crispo import CriSPO, Example, fill_prompt
from slick import prompts

# Configure once at application startup, before any model calls.
prompts.TEMPLATE_ROOT = Path(crispo.__file__).resolve().parent / "prompts"

# Your respond(filled_prompt) -> str calls the task model.
# Your evaluate(examples, predictions) -> dict[str, float] computes corpus metrics.
# optimizer_provider is a Slick Provider. It also critiques by default;
# pass critique_provider=... to use a separate critique model.
agent = CriSPO(
    task=task_description,
    provider=optimizer_provider,
    respond=respond,
    evaluate=evaluate,
    primary_metric="quality",
)

# Each split contains Example({"INSERT_INPUT_HERE": input_text}, reference_text).
# train and dev must be separate, nonempty datasets.
result = await agent.run(
    initial_prompt=seed_template,  # Includes INSERT_INPUT_HERE exactly once.
    train=train,
    dev=dev,
    iterations=20,
    history_size=5,
    critique_size=10,
    seed=0,
)
best_template = result["best"].prompt
answer = await respond(fill_prompt(best_template, {"INSERT_INPUT_HERE": new_input}))
```

`respond` receives only a filled string, never an `Example` or its reference.
`evaluate` receives the examples and predictions of one split in their original
order, and returns aggregate metrics. It owns metric computation, output parsing,
and any required execution isolation. Provider construction, sampling settings,
transport retries and persistence also remain with the caller.

For ICL, RAG, or additional input fields, declare all required tokens via
`placeholders=("INSERT_INPUT_HERE", "INSERT_CONTEXT_HERE", "INSERT_EXAMPLES_HERE")`
and populate those keys in each `Example.inputs`. Tokens follow `INSERT_*_HERE`
with uppercase letters, digits and underscores. The optimizer may reposition
each token, but must retain each exactly once. `fill_prompt` uses literal,
single-pass substitution: braces or placeholder-looking text inside the supplied
data are preserved. Missing input keys raise `KeyError`. Demonstration selection,
retrieval and formatting belong to the caller; avoid selecting demonstrations
from held-out data.

## Automatic Suffix Tuning

Add these options to `run` to tune a suffix after primary-metric optimization:

```python
initial_suffix="Check the output against the supplied constraints.",
suffix_iterations=20,
suffix_metrics=("faithfulness",),
```

The evaluator must return `quality` and `faithfulness` on both splits for this
example. AST freezes the dev-selected main template and appends each candidate
postscript with two newlines. Only postscripts are generated. The primary metric
is always included once, alongside the supplied secondary metrics.

AST uses the official implementation's **zero-based competition ranks**:
scores `[10, 10, 9]` have ranks `[0, 0, 2]`. Average these ranks across metrics;
negate the result for the optimizer's higher-is-better score. Ranks are recomputed
against the entire current suffix archive, separately for train and dev.
This balances metric scales; it is not a guarantee against metric regressions.
Stable ties prefer the earlier candidate. `initial_suffix=""` explicitly includes
the unchanged main prompt as a suffix-stage candidate; otherwise selection is
among the supplied suffix seed and generated suffixes.

## Results, budgets and failure behavior

`result["best"]` is a `Candidate` containing the editable `text`, full `prompt`,
`train_scores`, `dev_scores`, and textual `critique`. `main_best` records the winner
before AST. `history` and `suffix_history` retain the measured candidates;
`generations` retains raw critique and revision responses, including rejected
responses. The same state is available on the agent after an exception. A new
`run` resets the agent's state; don't run the same instance concurrently.

The seed is evaluated even for zero iterations. Each iteration spends one
optimizer call. A unique candidate costs `len(train) + len(dev)` task calls,
two evaluator calls and one critique call (including the final candidate).
Exact duplicates consume their iteration and skip those costs. AST has its own
seed and iteration budget. Counters `optimizer_calls`, `critique_calls`,
`response_calls`, `evaluations`, and `duplicates` are included in the result and
remain on the agent. Call counters advance before the corresponding invocation.

Blank/malformed response tags, missing/repeated/unknown generated placeholders,
nonfinite measurements, and provider/evaluator errors propagate immediately.
There is no automatic retry, repair, silent candidate fallback, or shared Session.
Caller inputs and configuration are trusted. Generation checks concern model
outputs, not configuration preflight. Slick's template root is process-global;
configure it once and don't switch it concurrently between algorithms.

## Official sources and adaptations

Inspected and used [Amazon's official repository](https://github.com/amazon-science/CriSPO)
at commit `411864aa9b264c6d93c55f2b71a6735d16134dfe`:

- [Trainer](https://github.com/amazon-science/CriSPO/blob/411864aa9b264c6d93c55f2b71a6735d16134dfe/crispo/trainer/trainer.py): training trajectory, critique sampling, dev selection, deduplication and dynamic AST ranks.
- [Multi-aspect critique](https://github.com/amazon-science/CriSPO/blob/411864aa9b264c6d93c55f2b71a6735d16134dfe/experiments/summarization/crispo/critique_prompt.py): discover dimensions, identify differences, attribute prompt flaws and suggest edits.
- [Receptive optimizer](https://github.com/amazon-science/CriSPO/blob/411864aa9b264c6d93c55f2b71a6735d16134dfe/crispo/optimizer/crispo_meta_prompt.py): enriched trajectory, suggestion synthesis and tagged instruction generation.
- [AST optimizer](https://github.com/amazon-science/CriSPO/blob/411864aa9b264c6d93c55f2b71a6735d16134dfe/crispo/optimizer/ast/crispo_meta_prompt.py) and [suffix critique](https://github.com/amazon-science/CriSPO/blob/411864aa9b264c6d93c55f2b71a6735d16134dfe/crispo/task/ast/critique_suffix.py): fixed main instruction and postscript-only revision.

This is a deliberate task-agnostic port. Four local Jinja templates adapt the
official prompts while preserving `<critique>`, `<suggestion>`, `<instruction>`
and `<postscript>` text boundaries. Task-neutral observations replace summary
examples; metrics replace hard-coded faithfulness instructions. The critic
discovers dimensions without predefined aspect examples. The revision prompt
receives task context and the enriched trajectory, without the official trainer's
additional few-shot example selection.

The search uses one seed and one proposal per iteration, as in the paper's core
loop, instead of the official trainer's batched proposals and retry mechanism.
Every candidate is evaluated on dev instead of using periodic/thresholded dev
evaluation. Invalid generations raise rather than being repaired or resampled;
placeholder insertion and summary-output wrappers are intentionally removed.
No official dependency stack, dataset loaders, paid model runs or paper benchmark
results are included. The local Slick checkout used for verification declares
`slick-ai` 0.3.0.

The adapted prompt templates retain Amazon's **CC-BY-NC-4.0** license and copyright
notice. Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
See [LICENSE.upstream](LICENSE.upstream) for the license and warranty disclaimer;
the modifications are described above. The port does not imply Amazon endorsement.

Check from the repository root:

```sh
rtk proxy optimizer/.venv/bin/python -m unittest tests.test_crispo
```

The shared scripted provider verifies algorithm decisions, input separation,
failure accounting and templates from multiple working directories. These are
deterministic implementation checks, not a reproduction of the paper's scores.
