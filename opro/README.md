# OPRO: Large Language Models as Optimizers

Problem-agnostic Slick implementation of [Yang et al.](https://arxiv.org/abs/2309.03409).
Provide a task description, an optimizer provider, an async scalar evaluator, and
initial solutions. Solutions are strings: instructions, JSON parameters, routes,
or any other representation your evaluator understands.

## Use

Install Slick in your environment (this workspace uses the adjacent `../slick`
checkout). Configure the template root once at application startup, using the
module path so it works independently of the launch directory:

```python
import json
from pathlib import Path

import opro
from slick import prompts
from slick.providers import Provider
from opro import OPRO

prompts.TEMPLATE_ROOT = Path(opro.__file__).resolve().parent / "prompts"

async def optimize(provider: Provider):
    async def evaluate(candidate: str) -> float:
        x, y = json.loads(candidate)
        return (x - 3) ** 2 + (y + 2) ** 2

    optimizer = OPRO(
        task="Find a pair of real numbers minimizing the measured loss. "
             "Encode each solution as a JSON array [x, y].",
        provider=provider,
        evaluate=evaluate,
        maximize=False,
    )
    return await optimizer.run(
        ["[0, 0]", "[10, 10]", "[-5, -5]"],
        iterations=200,
        proposals_per_step=8,
        history_size=20,
        patience=None,
    )

# result = await optimize(your_provider)
# result["best"]["prompt"] is the winning solution text; ["score"] is its measured loss.
```

For prompt optimization, use `maximize=True` (the default), seed with `[""]` or
your existing instructions, and evaluate accuracy over a fixed training subset.
Pass formatted training examples through `exemplars`; show where the candidate
instruction will be inserted. By default, three examples are sampled without
replacement each step (or all if fewer are available). `exemplars_per_step=0`
disables them. `seed` controls example sampling, not model randomness. Keep
held-out data outside the optimizer and evaluate it only after selecting a result.

## Algorithm and contract

1. Evaluate each distinct seed, recording iteration `-1`.
2. Select the best `history_size` measurements and order them worst to best:
   ascending scores for maximization, descending for minimization.
3. Sample task examples once and generate `proposals_per_step` independent
   proposals from that same history and example snapshot.
4. Evaluate each new distinct solution and append its actual score to the archive.
5. Repeat until `iterations` steps finish, or optional `patience` consecutive
   steps fail to improve the best score. Ties and duplicate-only batches count
   as no improvement; any strict improvement resets patience.

`run(initial_prompts, ...)` retains its original argument and result field names:
`initial_prompts` and `prompt` mean arbitrary solution text. Supply at least one
seed; an empty string is valid. The default is 20 iterations for compatibility
with the existing port; set 200 for the paper's prompt-search budget. History
size and proposal count default to the paper's 20 and 8. Configure optimizer
temperature on your provider (the paper uses 1.0); an LLM scorer belongs inside
your evaluator and uses temperature 0 in the paper.

Results contain `best`, `archive`, per-step best `history`, `optimizer_calls`,
`evaluations`, `duplicates`, `raw_responses`, and `stop_reason` (`iterations` or
`patience`). Archive records contain `prompt`, `score`, and `iteration`. The
best result prefers the earliest equal-score entry; history truncation keeps
later entries when equal scores straddle the cutoff, as upstream does.

Exact-text duplicates consume their generation slot but skip evaluation.
Generated text loses surrounding whitespace, while seed text is unchanged.
Scores must be finite; negative scores work in either direction. Evaluation
must be repeatable for this deduplication policy to make sense.

Malformed, blank, or multiple `<TEXT>...</TEXT>` pairs raise `ValueError`.
The raw response is retained before this check. Provider and evaluator failures
propagate without retries; counters increment before calls and failed evaluation
never enters the archive. On failure, inspect the agent's partial `archive`,
`raw_responses`, and counters. A generation failure aborts the batch before its
evaluation phase. A new `run()` resets state. The caller owns transport retries,
domain validation, execution isolation, persistence, and provider construction.
Calls use `provider=` without shared conversational sessions. Slick's template
root is process-global; do not change roots concurrently.

## Official implementation and adaptations

Reviewed and used [Google DeepMind's official implementation](https://github.com/google-deepmind/opro)
at revision `a76bdce2cbf6d4a0d1e570a6fcfe17be9c2abdd7`:

- [`opt_utils.py`](https://github.com/google-deepmind/opro/blob/a76bdce2cbf6d4a0d1e570a6fcfe17be9c2abdd7/opro/optimization/opt_utils.py):
  `gen_ins_and_score_pairs_substr`, `gen_meta_prompt`, `parse_tag_content`, and
  `run_evolution` inform history selection, tagged output, sampling, and batches.
- [`optimize_linear_regression.py`](https://github.com/google-deepmind/opro/blob/a76bdce2cbf6d4a0d1e570a6fcfe17be9c2abdd7/opro/optimization/optimize_linear_regression.py):
  minimization presents the lowest objective values in descending order.
- [`optimize_instructions.py`](https://github.com/google-deepmind/opro/blob/a76bdce2cbf6d4a0d1e570a6fcfe17be9c2abdd7/opro/optimization/optimize_instructions.py):
  experiment defaults for 8 proposals, 20 history entries, and 3 examples.

This is an algorithm adaptation, not a dependency on the upstream benchmark
runner. The local prompt is deliberately task-neutral and uses the upstream
instructions-only `<TEXT>` delimiter for every solution type. It does not copy
benchmark-specific prompts, normalize sentences, restrict digits or lengths,
filter scores by a threshold, or bucket arbitrary objective values. Python's
local RNG replaces NumPy's per-step reseeding, so sampled examples differ from
upstream. Optional patience makes the paper's convergence stop explicit; it is
disabled by default. No benchmark downloads or legacy model SDKs are needed.

## Verification

From the repository root:

```sh
rtk proxy optimizer/.venv/bin/python -m unittest tests.test_opro
```

Scripted-provider checks cover both score directions, frozen batches, bounded
history, duplicate accounting, example sampling, stopping, empty seeds, tagged
output, raw failure records, and evaluator/transport failures. These verify the
algorithm and Slick integration; they do not reproduce the paper's accuracies.
