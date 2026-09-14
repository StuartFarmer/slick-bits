# APEX with Slick

An implementation of **Automatic Engineering of Long Prompts** (Hsieh, Si,
Yu, Dhillon): sentence-level beam search, history-guided mutation, and LinUCB
sentence selection. Slick's `@prompt` handles both mutation and target-model
prediction; ordinary Python owns search state. The local Slick library is used
without modification.

## Run offline

```bash
cd ~/Developer/AI/slick-bits/apex
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python test_apex.py
.venv/bin/python run.py --iterations 5 --output runs/demo
```

Python 3.10+ is required. The default provider is a **canned demo** with synthetic
arithmetic examples, synthetic embeddings, and a predetermined useful mutation.
It exercises Slick and the search loop; its accuracy is not research evidence.
During implementation, checks used the existing `../optimizer/.venv/bin/python`,
which already has NumPy and the local editable Slick installed.

Model instructions live in this implementation's own `prompts/` directory. The CLI
sets Slick's template root at startup, so it can launch from any working directory.
Before calling `optimize`, `predict`, or their prompt renderers programmatically,
set `slick.prompts.TEMPLATE_ROOT = Path(apex.__file__).resolve().parent / "prompts"`
once (with `import apex`, `import slick.prompts`, and `from pathlib import Path`).
This root is process-global; do not switch it during concurrent calls.

## Optimize a real prompt

Install the local T5 encoder and Slick's optional LiteLLM provider:

```bash
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/python -m pip install -e '../../slick[litellm]'
# Set credentials for your selected provider in the environment.
.venv/bin/python run.py --provider litellm \
  --model YOUR_LITELLM_MODEL --mutator-model YOUR_LITELLM_MUTATOR_MODEL \
  --prompt prompt.txt --data examples.json --answer-mode choice \
  --iterations 50 --seed 0 --output runs/task-seed-0
```

`--model` uses LiteLLM's provider/model syntax. Evaluation requests temperature 0;
mutation requests 0.5. Use models that support these parameters: unsupported
parameters raise errors rather than being silently discarded. Target and mutator
may use different models; the mutator defaults to the target model.
`--max-tokens` defaults to 2048 per response; `--timeout` to 120 seconds per call.

`--provider codex` uses Slick's Codex CLI provider and existing CLI authentication.
It does not expose the paper's temperature settings, so it is a convenience
integration, not an exact experimental configuration.

The default encoder is
[`sentence-transformers/sentence-t5-base`](https://huggingface.co/sentence-transformers/sentence-t5-base),
loaded locally through `SentenceTransformer.encode`. Embeddings are cached and
normalized to unit length. First use downloads model weights. `--encoder` accepts
another checkpoint or a local model directory. The paper does not specify its
exact T5 checkpoint, so this is an explicit implementation choice.

## Inputs and scoring

Use a JSON list, BBH's `{"examples": [...]}` object, or JSONL rows:

```json
[
  {"input": "Which option is correct? ...", "target": "(A)"},
  {"input": "Another question ...", "target": "(B)"}
]
```

GSM8K's `question`/`answer` keys are also accepted. Use `--data` for a seeded
50/50 split, or `--train train.jsonl --test test.jsonl` for explicit splits.
For GSM8K, supply your chosen 1,000 training examples and official test examples
as separate files. Both splits must be nonempty and cannot share question text
(case and whitespace normalized). Training examples score candidates; held-out
examples are evaluated only after the final prompt is selected. Keep evaluation
examples out of the prompt's demonstrations when preparing your data.

Each target call receives `prompt + separator + input + suffix`. Defaults append
`\n\nQ: ` before the question and `\nA: Let's think step by step.` after it.
Override `--separator` and `--suffix` to fit another task; these strings are fixed
throughout search. The original prompt's demonstrations and instructions are
eligible for mutation.

| `--answer-mode` | Scoring rule |
| --- | --- |
| `text` (default) | Case-insensitive stripped text after the final “answer is/answer:” marker; ignore trailing periods |
| `choice` | Last parenthesized letter, or a single letter, after a final-answer marker if present |
| `number` | Last signed decimal, ignoring commas; supports GSM8K's `####` gold delimiter |
| `dyck` | Bracket-only answer after a final-answer marker, ignoring whitespace |

Unparseable predictions count as incorrect. Unparseable targets are rejected
before model calls. Number scoring is intended for GSM8K decimal answers, not
fractions or symbolic math. Custom tasks can pass their own asynchronous evaluator
to `optimize`; its finite scalar score is maximized.

## Sentence boundaries and fixed formatting

The automatic splitter retains all original bytes, keeps line-leading `Q:`,
`A:`, `Question.`, `Answer.`, option labels, and enumerated prefixes fixed, and
splits prose at common punctuation boundaries. Replacing one fragment leaves all
other fragments unchanged. This heuristic is not a linguistic sentence tokenizer;
abbreviations, equations, and complex markup may need explicit boundaries.

For precise control, use a `.json` prompt with explicit mutable fragments:

```json
[
  {"text": "Solve the problem carefully.", "mutable": true},
  {"text": "\nQ: What is 1 + 1?\nA: ", "mutable": false},
  {"text": "Add the two numbers.", "mutable": true},
  {"text": " The answer is 2.\n", "mutable": false}
]
```

The mutator is instructed to preserve meaning, facts, names, numbers, and logic.
Empty/multiline replies and obvious formatting wrappers are rejected. This does
**not** establish semantic equivalence; the paper itself identifies incorrect
rephrasing as a limitation. Inspect the saved prompts and mutation history.

## Search behavior and ablations

| Parameter | Default |
| --- | --- |
| `--iterations` | 50 mutation attempts, plus initial evaluation |
| `--beam-size` | 4 |
| `--alpha` | 0.05 |
| `--regularization` | 1 (not specified in the paper) |
| `--random-probability` | 0.5 |
| `--history-limit` | 4 |
| `--history-distance` | Strict normalized Euclidean distance < 0.5 |

Each attempt selects a random beam member, then a random or LinUCB-selected
sentence. Rewards are candidate score minus **that parent's** score, including
edits rejected by the beam. Ridge features use the sentence before mutation.
The implementation solves the mathematically equivalent dual ridge system to
avoid a large embedding-dimension matrix inversion at each step.

History retrieval compares against the **before** sentence even for negative
edits; negative edits become after→before examples. Zero-reward edits do not
enter the mutation examples, but valid zero-reward observations train LinUCB.
The top-k pool updates immediately after each candidate. Equal scores retain
older candidates; ties between UCB values are broken using the seeded RNG.

Duplicate prompts reuse cached training scores. Invalid mutations consume an
attempt but do not call the evaluator or update the bandit. Thus 50 attempts
use **at most 51 full training-set evaluations**, not exactly 50. For a cap of
50 including the baseline, pass `--iterations 49`. At most `iterations` mutator
calls, `(iterations + 1) * n_train` target calls for search, and `2 * n_test`
target calls for final testing are made. Calls run sequentially. Provider
errors stop the run and retain already-written records; there are no hidden
retries or resumable workflow claims.

Useful ablations:

```bash
# Add these flags to the same real-data command, with distinct output directories.
--no-guided-mutation                 # Figure 2-style vanilla mutation
--random-probability 1               # no LinUCB sentence selection
--beam-size 1 --random-probability 1 --no-guided-mutation  # vanilla greedy
```

## Outputs and validation

Each run requires a new output directory and writes:

- `metadata.json`, `initial.txt`, `fragments.json`: settings, data split, local
  Slick path, initial prompt hash, and the exact mutable boundaries.
- `history.jsonl`: flushed baseline and mutation records, selected sentence,
  parent/candidate prompts, reward, retrieved examples, and best training score.
- `evaluations.jsonl`: flushed training predictions and correctness per evaluated prompt.
- `best.txt`, `search.json`: selected prompt and complete search result, saved
  before any test calls.
- `result.json`: training results, original/best held-out predictions and scores,
  plus completed-run model-call counts.

The offline check compares LinUCB against the primal ridge equations and checks
beam retention, reward orientation, history reversal/retrieval, embedding cache
normalization, invalid outputs, duplicate evaluation budgets, exact formatting,
answer extraction, and training/test isolation through real Slick rendering.
The canned demo also verifies output artifacts. Real model calls and T5 weight
loading were not run during implementation. The paper's BBH/GSM8K accuracy gains,
three-run averages, GA/APO/PromptBreeder comparisons, and figures are not reproduced.
