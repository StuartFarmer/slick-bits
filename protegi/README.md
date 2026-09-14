# ProTeGi

Task-agnostic [ProTeGi / Automatic Prompt Optimization with “Gradient Descent”
and Beam Search](https://arxiv.org/abs/2305.03495), implemented as one Slick
optimizer with local gradient, revision, and paraphrase templates.

## Use with your task

Supply a task description, an optimization provider, and an async evaluator:

```python
from pathlib import Path

import protegi
from slick import prompts
from protegi import Evaluation, ProTeGi

# Configure once at application startup; independent of the working directory.
prompts.TEMPLATE_ROOT = Path(protegi.__file__).resolve().parent / "prompts"

async def optimize(task, initial_prompt, training_examples, provider, predict,
                   validation_examples=None):
    # This example uses exact match. Replace scoring and error descriptions
    # with your task's metric, judge, parser, or execution harness.
    async def evaluate(instruction, batch):
        predictions = [await predict(instruction, row["input"]) for row in batch]
        errors = tuple(
            f"Input: {row['input']!r}\nExpected: {row['target']!r}\nActual: {output!r}"
            for row, output in zip(batch, predictions)
            if output != row["target"]
        )
        return Evaluation(1 - len(errors) / len(batch), errors)

    return await ProTeGi(task, provider, evaluate).run(
        [initial_prompt], training_examples,
        selection="ucb", iterations=6, beam_size=4,
        selection_budget=800, samples_per_eval=5, exploration=2.0,
        validation_examples=validation_examples,
    )

# result = await optimize(task, initial_prompt, training_examples,
#                         optimizer_provider, predict, validation_examples)
# optimized_prompt = result["best"]["prompt"]
```

`predict` is your async `(instruction, input) -> output` function. It can call a
separate model, render a fixed wrapper with few-shot examples, or invoke an
existing application. `provider` is a Slick provider used only to generate
textual gradients, edits, and paraphrases. The evaluator owns execution isolation,
output interpretation, labels, and scoring. No dataset, provider vendor, output
schema, or classification task is built into the optimizer. Examples can be any
Python objects; the optimizer only samples them and passes them to the evaluator.

The evaluator returns `Evaluation(score, errors)`: a finite higher-is-better
metric and textual mistake descriptions for exactly the supplied batch. Include
input, expected behavior, and actual behavior in errors when available. For a loss,
return its negative. UCB rewards should preferably have a consistent bounded scale
such as `[0, 1]`; tune `exploration` for other scales. UCB and elimination selectors
average batch scores weighted by example count. For non-additive metrics such as
F1, this estimates batch rewards, **not pooled F1**. Final ranking calls your exact
metric on the entire final dataset.

## Algorithm and controls

Each iteration samples a training minibatch shared by the current beam. For each
prompt it samples groups of error descriptions, generates textual gradients,
applies each gradient, and generates semantic variations of edits and incumbents.
It deduplicates and randomly caps successors per parent, retains incumbents in the
candidate pool, and selects the next beam. Parent retention is eligibility, not a
guarantee that an incumbent survives selection.

| Setting | Default | Meaning |
| --- | --- | --- |
| `iterations` | 3 | Number of expand/select rounds; use 6 for the paper setup |
| `beam_size` | 4 | Maximum survivors per round |
| `minibatch_size` | 64 | Training examples used to find errors |
| `gradient_batches` | 1 | Error groups sampled per parent |
| `errors_per_gradient` | 4 | Maximum errors per group |
| `gradients_per_batch` | 4 | Textual reasons generated per group |
| `steps_per_gradient` | 1 | Edits per reason |
| `paraphrases` | 2 | Variations per edit and incumbent |
| `max_expansion` | 8 | Maximum successors retained per parent |
| `selection` | `s-sr` | Selector; existing default preserved |
| `selection_budget` | 100 | Selection example evaluations per round |
| `samples_per_eval` | 5 | Examples per UCB arm pull |
| `exploration` | 2.0 | UCB exploration coefficient |
| `prompts_per_round` | 4 | Candidate subset size for legacy `s-sr` only |
| `seed` | 0 | Local sampling seed; does not seed the model |

Supported selectors:

- `ucb`: sample each arm, then maximize its mean plus
  `c * sqrt(log(t) / N)`; retain the highest empirical means.
- `ucb-e`: use `c * sqrt(c / N)` as the exploration bonus.
- `sr`: successive rejects with the paper's harmonic cumulative sample schedule;
  eliminate one worst arm per phase until the beam remains.
- `sh`: allocate samples across halving phases and discard the bottom half,
  stopping at the beam width.
- `uniform`: score all candidates on one common random batch.
- `s-sr`: preserve the prior port's official stochastic variant, comparing a random
  candidate subset on a common fresh batch and rejecting its worst member.

The five new selectors never exceed `selection_budget`. Sample sizes are capped
by available data; different pulls can reuse examples. At least one observation
per candidate must fit when selection is needed, or a `RuntimeError` reports
budget exhaustion. If a later elimination phase cannot fit a complete comparison,
selection finishes using the accumulated estimates. Unused budget is possible.
New selectors break score ties by stable candidate order. Legacy `s-sr` breaks
rejection ties by the first minimum in its randomly ordered subset.

Legacy `s-sr` retains its **nominal** budget:
`ceil(budget / ((candidate_count - beam_size) * prompts_per_round))` examples per
comparison. Rounding can exceed that budget. Minibatch evaluation and final
ranking are outside all selection budgets. No selection calls are needed when
the candidate pool already fits the beam.

`validation_examples`, when supplied, are used only for final ranking. Otherwise
final ranking uses all training examples. Validation selects the returned prompt;
use a separate untouched test set for reporting generalization. `iterations=0`
just ranks the initial prompts. The default three iterations is a local convenience,
not a reproduction of the paper's six-round experiment.

## Results and failures

The returned dictionary contains `best` and `population` entries with `prompt` and
`score`, per-round `history` with beams and rejected prompts, and `measurements`
with prompt, score, phase (`gradient`, `selection`, `final`), and batch size.
`evaluations` counts evaluator attempts, `example_evaluations` counts examples
submitted, and `optimizer_calls` counts generation attempts. They do not measure
hidden provider retries, evaluator caching, or monetary cost.

Errors propagate immediately without optimizer retries. Counters remain on the
agent after a failure. Tagged outputs must contain exactly the requested number
of nonempty `<START>...<END>` items; paraphrases must be nonempty. Rejection errors
include the raw generated text. Configure provider-level logging if you also need
a complete generation transcript. No mutable conversation session is shared.
Use a separate optimizer instance per concurrent run, and configure Slick's
process-global template root once; separate processes are needed for concurrent
applications requiring different roots.

## Official implementation and adaptations

Reference implementation inspected on 2026-09-14:
[Microsoft LMOps / prompt_optimization](https://github.com/microsoft/LMOps/tree/main/prompt_optimization).
The implementation uses its expansion procedure and bandit design as references:

- [optimizers.py](https://github.com/microsoft/LMOps/blob/main/prompt_optimization/optimizers.py):
  error groups, tagged gradients and edits, semantic variations including the
  incumbent, random expansion cap, and original-prompt retention.
- [evaluators.py](https://github.com/microsoft/LMOps/blob/main/prompt_optimization/evaluators.py):
  UCB/UCB-E weighted rewards, stochastic rejection, and uniform evaluation.

The generic prompt wording deliberately replaces the source's zero-shot binary
classifier wording. The entire supplied instruction is editable; keep immutable
formatting and examples in your evaluator's wrapper. The optional source
error-rejection filter and benchmark infrastructure are omitted. Like the source,
empty error groups still permit gradient generation.

Numerical corrections are intentional: UCB samples every arm before applying a
bonus and weights actual batch sizes. SR uses candidate count in the harmonic
schedule and cumulative estimates. SH drops the bottom half as described in the
paper; upstream instead filters below the phase mean. Both stop at the requested
beam width and obey the explicit budget. These choices, final validation ranking,
and generic prompts mean this is an algorithm implementation, not a bit-for-bit
port or reproduction of the published benchmark results.

## Checks

Use the repository's existing Slick environment (no new dependency):

```bash
rtk proxy optimizer/.venv/bin/python -m unittest tests.test_protegi
rtk proxy ../slick/.venv/bin/ruff check protegi tests/test_protegi.py
```

The deterministic tests cover expansion and selection, budgets, weighted rewards,
exploration, finite-score handling, ties, small datasets, final validation
separation, generated-output rejection, and template loading from another working
directory. No paid model calls or benchmark performance claims.
