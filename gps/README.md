# GPS: Genetic Prompt Search

A problem-agnostic Slick implementation of [Xu et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.559/),
adapted from the [official implementation](https://github.com/hwxu20/GPS).
Supply seed prompts, a generation provider, and an async evaluator. GPS searches
discrete prompt text without training model parameters.

## Use on your task

Configure the local template root once at application startup, before any calls:

```python
from pathlib import Path

from slick import prompts
import gps

prompts.TEMPLATE_ROOT = Path(gps.__file__).resolve().parent / "prompts"

async def optimize(task, provider, evaluate, initial_prompts):
    agent = gps.GPS(task, provider, evaluate)
    return await agent.run(
        initial_prompts,
        generations=7,              # G0 plus six reproduction steps
        top_k=None,                  # number of distinct initial prompts
        pool_size=30,                # attempted children per SC/cloze generation
        strategy="sentence_continuation",
        rescore_final=False,         # official code reuses measured scores
    )

# result["best"]["prompt"] is the optimized prompt text.
```

`evaluate(prompt: str) -> float` is async and returns a finite score to maximize.
Close over your fixed development set and target model in that callback. For
classification, return development accuracy; for a loss, return its negative.
Your evaluator owns prompt rendering, output interpretation, batching, and any
required execution isolation. The generator sees task context and parent prompts,
never development examples or labels. Keep held-out test data outside search.

The paper uses 32 development examples. No dataset, class labels, target model,
credentials, or benchmark infrastructure are built into this package. Configure
the generation provider's sampling separately (`top_p=0.9` in the paper); the
Slick call boundary here does not expose per-call sampling settings.

Settings are caller-owned: supply nonempty seeds, positive budgets, a nonempty
language list, and `mask_fraction` in `(0, 1]`. Use one active run per instance.
Imports never modify Slick configuration; its template root is process-global,
so concurrent applications needing different roots require separate processes.

## Search behavior

1. Deduplicate and score the seeds; select the top K.
2. Generate children from selected parents. Reject empty text, previously
   accepted/initial text, and violations of `accept(parent, child)`.
3. Score valid children and replace the population with them. If every attempt
   is rejected, retain selected parents with their existing scores.
4. Archive each evaluated generation's top K and repeat.
5. Return the top K unique archived prompts. Stable sorting favors earlier
   candidates on ties. `rescore_final=True` evaluates every unique archived
   prompt again before final ranking, as described in the paper's prose.

The archive preserves earlier winners even when offspring perform worse.
Only selected parents reproduce; GPS has no crossover. Invalid outputs consume
their attempt without replacement, so the resulting pool can be smaller than
the attempt budget. `generations=1` scores and selects seeds only.

The default `accept` preserves the exact multiset of literal `{{...}}` slots
(including multiline slots), plus the order of `{%...%}` and `{#...#}` tags.
This is a lexical check, not semantic equivalence or a full Jinja parser.
Candidate text is never rendered or executed by GPS. Supply a custom callback
for another placeholder syntax, allowed labels, or stricter task constraints:

```python
from gps.agent import preserves_placeholders

def accept(parent, child):
    return preserves_placeholders(parent, child) and "Return JSON" in child

agent = gps.GPS(task, provider, evaluate, accept=accept)
```

## Generation strategies

| Strategy | Operation | Default budget |
| --- | --- | --- |
| `sentence_continuation` | Complete the paper's equivalent-sentence instruction | 30 attempts distributed evenly across selected parents |
| `back_translation` | Translate to an intermediate language, then back to English | One round trip per language per parent |
| `cloze` | Mask words outside template tags, generate sentinel fills, reconstruct unmasked text | 30 attempts distributed evenly across selected parents |

`offspring_per_parent=n` overrides either default with `n` attempts per parent.
For SC/cloze, budget remainders go to the highest-ranked parents first.
BT cycles through `languages` in order when an explicit per-parent budget is
supplied. Its default list is the paper's Chinese, Japanese, Korean, French,
Spanish, Italian, Russian, German, Arabic, Greek, and Cantonese. BT uses two
provider calls per completed attempt. Blank translations or changed literal
tags stop an attempt before the return trip. Final back translations also pass
the caller's `accept` callback.

```python
bt = await agent.run(initial_prompts, strategy="back_translation")
cloze = await agent.run(initial_prompts, strategy="cloze", mask_fraction=0.15, seed=0)
```

Cloze masks `ceil(mask_fraction * eligible_words)` words, at least one. `seed`
controls the local masking RNG, not provider sampling. Outputs follow
`<extra_id_0> fill <extra_id_1> ... <extra_id_n>`, with a terminal sentinel and
no surrounding prose. Missing, reordered, or empty fills are rejected. Unmasked
bytes are reconstructed in Python. A prompt with no editable words consumes an
attempt but no model call. For the paper's cloze scoring, your evaluator should
return average correct-target logits over the development set; GPS cannot
recover logits from a text provider.

## Results and failures

- `best`, `finalists`: `{ "prompt": str, "score": float }` records.
- `population`: the last population, which can differ from the final winners.
- `archive`: unique per-generation winners with their original search scores.
- `generations`: ordered snapshots containing `population` and `selected`.
- `history`: one record per attempt with parent, strategy, generation, raw
  response, and rejection reason. BT also records its intermediate translation;
  cloze records the masked text. `raw=None` means no final response was obtained.
- `attempts`, `optimizer_calls`, `evaluations`: separate counters. Evaluations
  include optional final rescoring; calls count Slick invocations, not internal
  transport retries or target-model calls inside your evaluator.

Provider/evaluator exceptions and nonfinite measurements propagate immediately.
The agent retains attempt history and counters after failure. An interrupted
generation record remains `rejection="pending"`; retries belong to the caller.
Raw cloze responses are recorded before parsing, including malformed responses.
All operations return text, so no JSON schema or mutable Session is needed.

## Official source use and adaptations

Inspected and adapted revision
[`f5b24843efa8e209cee5fc1257ecc55a39134d11`](https://github.com/hwxu20/GPS/tree/f5b24843efa8e209cee5fc1257ecc55a39134d11):

- [`ga_processer_t0.py`](https://github.com/hwxu20/GPS/blob/f5b24843efa8e209cee5fc1257ecc55a39134d11/ga_processer_t0.py):
  score/select/reproduce order, empty-generation fallback, and unique final
  top-K selection from per-generation winners using stored scores.
- [`dino/use_dino_to_generate_template_t5.py`](https://github.com/hwxu20/GPS/blob/f5b24843efa8e209cee5fc1257ecc55a39134d11/dino/use_dino_to_generate_template_t5.py):
  paraphrase generation, input-interface filtering, and deduplication.
- [`dino/task_specs/generate_task_description_en_v2.json`](https://github.com/hwxu20/GPS/blob/f5b24843efa8e209cee5fc1257ecc55a39134d11/dino/task_specs/generate_task_description_en_v2.json):
  released question-continuation prompt. This port uses the paper's sentence
  version with added task context and literal-tag preservation instructions.

The official algorithm is ported into Python and Slick; its GPU scripts and
dataset-specific template reconstruction are replaced with injected dependencies.
The original MIT notice is included in [LICENSE](LICENSE).

The released entrypoint uses nine evaluated generations, task-specific K, and
DINO batches of 30 candidates per input. This port defaults to six reproduction
steps, K equal to the seed count, and a total SC/cloze attempt pool of 30, following
the paper's experimental settings. Global deduplication follows the paper's
existing-prompt filter more strictly than the release's per-augmentation sets.
Final rescoring is optional because the release reuses scores despite the
paper's prose describing another evaluation.

BT and cloze implement Section 3.2; the released driver only wires SC. Cloze uses
whitespace words instead of T5 tokenization, and portable Slick completions
instead of T5 beam search. These are deliberate model-interface adaptations,
not a reproduction of the paper's benchmark scores.

## Checks

From the repository root, using its existing Slick environment:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_gps
../slick/.venv/bin/ruff check gps tests/test_gps.py
../slick/.venv/bin/ruff format --check gps tests/test_gps.py
```

Checks use the shared `tests/providers.py` scripted provider to cover selection,
replacement, budgets, all generation operations, filtering, failures, final
rescoring, and local template rendering. No paid calls or benchmark experiments.
