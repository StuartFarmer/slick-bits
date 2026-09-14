# EvoPROMPT with Slick

An implementation of the GA and DE prompt optimizers from **EvoPROMPT:
Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers**
(Guo et al.). The supplied paper's Sections 3.1–3.3 define the algorithms.

[Slick](../../slick) renders the evolutionary instructions and performs model
calls through `@prompt`. Python handles fitness, randomness, selection, and
population updates. The optimizer needs only a scalar development score; it
never needs gradients, token probabilities, or access to model parameters.

## Run

Requires Python 3.10+ and the local `../../slick` checkout.

```bash
cd ~/Developer/AI/slick-bits/evoprompt
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python test_evoprompt.py

# Canned providers, no credentials or network calls.
.venv/bin/python run.py --algorithm ga --population-size 4 --iterations 2 --output runs/demo-ga
.venv/bin/python run.py --algorithm de --population-size 4 --iterations 2 --output runs/demo-de
```

The demo uses canned offspring and a lexical sentiment scorer. Its scores are
only a plumbing check, not evidence of optimization or reproduced paper results.
Demo mode accepts only the bundled `demo.json` and cannot be mixed with a real
provider. Tests separately exercise selection using candidates with known,
unequal fitness values.

For real model calls, the runner supports Slick's text-only OpenAI and OpenRouter
API providers. The optimizer and target LLM can differ. Install the optional SDK
and supply explicit model IDs:

```bash
.venv/bin/python -m pip install 'openai>=2,<3'
# OPENAI_API_KEY must be set through your normal environment.
.venv/bin/python run.py --data task.json --provider openai --model OPTIMIZER_MODEL \
  --target-provider openai --target-model TARGET_MODEL --output runs/task-api
```

Use `--provider openrouter` and/or `--target-provider openrouter` for models served
through OpenRouter, with `OPENROUTER_API_KEY` in the environment. Both providers
use the optional OpenAI SDK. The runner supplies no tools or workspace access to
these model calls, keeping dataset answers outside their inference context.

`--target-provider` defaults to `--provider`. If the providers match,
`--target-model` defaults to `--model`; otherwise specify the target model
separately. `--timeout` sets seconds per model call. Calls are sequential.
API providers use Slick's sampling defaults; the historical GPT-3.5 and
Alpaca sampling configurations in the paper are not reproduced here.

## Dataset and scoring

Provide explicit, disjoint splits. The runner does not download or split datasets.

```json
{
  "prompts": [
    "Classify the sentiment as positive or negative. Return only the label.",
    "Read the review and output its sentiment: positive or negative."
  ],
  "demonstrations": [
    {"input": "I enjoyed this.", "target": "positive"},
    {"input": "I hated this.", "target": "negative"}
  ],
  "dev": [
    {"input": "A wonderful performance.", "target": "positive"},
    {"input": "A tedious film.", "target": "negative"}
  ],
  "test": [
    {"input": "A great soundtrack.", "target": ["positive", "favorable"]}
  ]
}
```

`prompts` and nonempty `dev` are required. `demonstrations` and `test` are optional.
Targets can be strings or nonempty lists of acceptable references; demonstration
targets must be strings. Additional row fields, such as dataset IDs, are retained
for custom metrics. Inputs that overlap across splits after case and whitespace
normalization are rejected. This detects duplicate text, not semantic leakage.

The task prompt contains the evolving instruction, fixed demonstrations, and
the current input. Development and test answers are passed only to the metric.
The winner is selected by development score; only that winner is evaluated on
the optional test split, once, after optimization finishes.

The default metric is accuracy in `[0, 1]`: exact match against any target after
case and whitespace normalization. It does not extract labels from explanations
or answers from chain-of-thought output. Put output constraints in your initial
prompts or supply a task-specific metric.

For generation tasks, `--metric my_metrics:score` loads a trusted, importable
Python function with this contract:

```python
def score(examples: list[dict], predictions: list[str]) -> float:
    # Compute your task's corpus metric using input texts, targets, and predictions.
    # Return one finite, nonnegative number; higher is better.
    ...
```

This hook supports ROUGE, SARI, BBH answer extraction, or another task's metric
without imposing their dependencies or silently substituting approximate scores.
Implement the selected metric in `my_metrics.py` beside `run.py`, or install its
module. ROUGE/SARI implementations, benchmark downloads, and the 31-dataset
experiment harness are not bundled.

## Use the optimizer directly

From this directory, any Slick provider and any async development evaluator work:

```python
import asyncio
from pathlib import Path
import evoprompt
from slick import prompts
from slick.providers import OpenAIAPI
from evoprompt import optimize
from run import evaluate_dataset, exact_match

prompts.TEMPLATE_ROOT = Path(evoprompt.__file__).resolve().parent / "prompts"
target = OpenAIAPI(model="TARGET_MODEL")
dev = [{"input": "A great film.", "target": "positive"}]

async def fitness(instruction):
    result = await evaluate_dataset(instruction, dev, [], target, exact_match)
    return result["score"]

result = asyncio.run(optimize(
    ["Return the review sentiment as positive or negative."],
    fitness, OpenAIAPI(model="OPTIMIZER_MODEL"), algorithm="de", population_size=4,
    iterations=2, seed=7,
))
print(result["best"])
```

Model instructions live in this implementation's own `prompts/` directory. The CLI
sets the absolute template root at startup, so it can launch from any working
directory. Programmatic callers configure it once as above before calling or
rendering prompts. Slick's root is process-global; do not switch it during concurrent calls.

For BBH-style answer-prefix prompting or a local Alpaca server, replace the
evaluator with your own Slick task prompt and provider. The optimizer remains
unchanged. `on_event=callback` receives JSON-serializable audit records.
Keep custom providers text-only as well: an agent with filesystem tools could
read dataset answers even if those answers are omitted from the rendered prompt.

## Algorithm choices

| Stage | GA | DE |
| --- | --- | --- |
| Parents | Two roulette draws, probability proportional to fitness | Two distinct random donor indices excluding the target |
| LLM operation | Crossover, then mutation | Identify donor differences, mutate only those differences, combine with the best prompt, then cross over with the target |
| Survivors | Highest-scoring N from parents plus N offspring | Trial replaces its own target only if its score is higher |

Both algorithms use a fixed population snapshot for each generation and produce
exactly N offspring. DE's best prompt is fixed within that generation and may
also be a donor or target. GA draws parents with replacement. Stable score ties
keep incumbents ahead of offspring; DE ties keep the target. These conventions
make details left unspecified in the supplied text explicit.

Initial manual prompts are retained and evaluated. If fewer than N are supplied,
Slick generates variations of randomly selected manual prompts until N is
reached. Supplying more than N is an error. The caller chooses any manual prompt
ranking or subset before the run. Duplicate offspring remain valid population
members; no unbounded uniqueness retries change the generation budget.

GA requires N ≥ 2; DE requires N ≥ 3. Both accept zero iterations to evaluate
initialization alone. The CLI defaults to N=10, T=10 as useful starting values,
not an asserted reproduction of every experimental setting in the paper.
All scores must be finite and nonnegative. All-zero GA fitness falls back to
uniform selection. Roulette weights are scaled to avoid overflow.

Fitness is cached by exact prompt string within one optimization run, assuming
repeatable evaluation. `--no-cache` (or `cache=False`) reevaluates duplicates.
`--seed` controls Python selection and initialization choices, not provider-side
sampling. With M supplied prompts, N population members, T iterations, and D dev
examples, a run makes `(N-M) + N*T` optimizer calls and at most `(N + N*T)*D`
target calls, plus the test set size. Cached duplicates reduce target calls.

## Artifacts and verification

Every run requires a new output directory and saves:

- `metadata.json` and `data.json`: settings, versions, source hashes, and input data.
- `events.jsonl`: raw evolutionary responses and parents, evaluated prompts,
  development predictions, cache hits, and full population snapshots.
- `best_prompt.txt`: the selected instruction.
- `result.json`: winner, initial best, final population, convergence history,
  evaluation counts, and optional held-out predictions and score.

Events flush after each completed operation. Malformed offspring, invalid scores,
and provider failures stop the run; there are no hidden retries. Errors are logged,
and a selected prompt survives a later test-evaluation failure. Logs are an audit
trail, not resumable checkpoints. Multiple runs never overwrite the same directory.

The offline checks exercise GA elitism and roulette selection, DE pairwise
replacement and generation snapshots, fixed population sizes, zero-score
selection, seeded repeatability, initialization, cache behavior, malformed output,
custom metrics, held-out isolation, and both CLI paths through real Slick rendering.
No live provider call or published benchmark superiority is claimed.
