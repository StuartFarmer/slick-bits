# Evolving code with Slick

Implementation of **Evolving code with a large language model**, Erik Hemberg,
Stephen Moskal and Una-May O'Reilly (2024), using the local
[Slick](../../slick) checkout. Implements the symbolic regression demonstration
in Section 5 and the LLM operators from Appendix 3.

```bash
cd ~/Developer/AI/slick-bits/llm_gp
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd ..
llm_gp/.venv/bin/python -m llm_gp.test_llm_gp
llm_gp/.venv/bin/python -m llm_gp.run --variant all --provider demo
```

Python 3.10+; only Slick and its dependencies are required. Commands below run
from `slick-bits`. The local `llm_gp/.venv` is installed in this workspace.
The existing `optimizer/.venv/bin/python` also works here.
The **demo returns canned expressions, including the known solution**. It tests
Slick rendering, parsing and the evolution loop; it is not an LLM experiment.

## Live runs

```bash
# Uses your existing Codex CLI authentication, with an explicit model choice.
llm_gp/.venv/bin/python -m llm_gp.run --provider codex --model YOUR_MODEL_ID \
  --population-size 3 --generations 2 --max-calls 20

# API route: install the SDK and configure OPENAI_API_KEY in your environment.
llm_gp/.venv/bin/python -m pip install 'openai>=2,<3'
llm_gp/.venv/bin/python -m llm_gp.run --provider openai --model YOUR_MODEL_ID \
  --variant llm-gp --population-size 3 --generations 2 --max-calls 30

# Paper-sized comparison, 30 seeds × 5 variants, 300 FEs per run.
llm_gp/.venv/bin/python -m llm_gp.run --provider openai --model YOUR_MODEL_ID \
  --variant all --runs 30 --output llm_gp/runs/comparison

# SS24 scale: 20 × 60 = 1,200 FEs per run.
llm_gp/.venv/bin/python -m llm_gp.run --variant all --scale --runs 30
```

The final command is offline unless a live provider is supplied. API calls use
Slick's current OpenAI Responses interface, not the historical GPT-3.5 Chat
Completions endpoint. Temperature defaults to 0.8; use `--omit-temperature` for
models that do not support it. Codex uses its CLI sampling settings; the API
temperature flag does not configure it. CLI invocations use Slick's read-only
execution mode and receive no application tools.

## Algorithms

| CLI variant | Initialization | Selection | Variation | Replacement |
| --- | --- | --- | --- | --- |
| `llm-gp-mu-xo` | LLM | Tournament, size 2 | LLM crossover/mutation | Generational, elite 1 |
| `llm-gp` | LLM | LLM | LLM crossover/mutation | LLM selection from parents + offspring |
| `gp` | Ramped half-and-half | Tournament, size 2 | Subtree crossover/mutation | Generational, elite 1 |
| `random` | Ramped half-and-half | None | Fresh independent population | Fresh samples |
| `llm-random` | One LLM call per candidate | None | Fresh independent population | Fresh samples |

Defaults: population 10, 30 evaluated generations, crossover 0.8, mutation 0.2,
two few-shot examples, GP maximum depth 5 (edges from root). All candidates use
`+`, `-`, `*`, `x0`, `x1`, `0`, `1`, and parentheses. LLM variation manipulates
expression **text**, while the traditional GP baseline exchanges tree substructures.
Full LLM_GP also asks the model to designate its final solution. Results retain
both that designation and the independently measured best expression seen.

`@prompt` supplies Jinja rendering and Pydantic JSON parsing. Every operator
formulates, calls, and checks. Malformed initialization falls back to `0`;
malformed variation retains its parents; malformed selection samples randomly
with replacement; malformed replacement/ranking uses numeric ordering. Selection
returns existing integer IDs so a model cannot invent individuals or alter their
measured scores. Crossover checks syntax and primitives, but does not enforce
conservation of parent terms, as in the paper.

Evaluation uses a small AST interpreter, **never `eval` or `exec`**. Calls,
attributes, indexing, powers, arbitrary constants and other Python syntax are
rejected. Bounds of 4,096 characters and 256 AST nodes limit evaluation work;
nonfinite arithmetic receives worst fitness (`null` in saved JSON). LLM expression
length is an explicit additional constraint. Traditional GP rejects over-depth
crossover children and bounds mutation depth.

## Data, counting and fidelity

The target is `x0*x0 + x1*x1`. Since the supplied paper does not specify the
coordinate range or split seeds, this implementation reconstructs a 121-point
integer grid on `[-5,5]²`. A seeded shuffle creates 25 holdout, 29 test and 67
training examples (20% holdout, then 70/30 of the remainder, with rounded counts).
The actual rows are saved. The same split is shared by every algorithm at a seed.
LLM variants use 10 sampled training examples; conventional variants use all 67.
Pass `--train-size 10` (or 67) for an equal-data comparison across every variant.

Fitness is locally measured **mean squared error, minimized**. Test and holdout
data never reach evolutionary operators; they score the returned solutions after
evolution ends. Prompts do not reveal the target formula or data: initialization
uses primitives; variation uses population expressions; selection uses measured
training fitness. The problem is well known, so training-data contamination cannot
be excluded for a real model.

Generations include the evaluated initial population. Every evaluation request
counts, including an elite or a cached expression: `population × generations`
is exactly 300 or 1,200. There is no final unevaluated generation and no extra child
when the population size is odd. The paper's Algorithm 1 loop bounds/indentation
are inconsistent, so the implementation follows these explicit invariants.
Full-variant replacement uses a parent-plus-offspring candidate pool; fallback
ordering is deterministic, and no elitism is imposed on a valid LLM choice.
The measured best-so-far archive does not influence that choice. These are stated
implementation choices, not a claim of byte-for-byte reproduction of the tutorial.

This implements the **demonstrated variants**, not hypothetical LLM execution or
LLM fitness measurement from general Algorithm 1. There is no arbitrary Python
program synthesis or model fine-tuning. Published costs, model-version effects,
statistical comparisons and plots have not been reproduced.

## Artifacts and budgets

New output directories contain `metadata.json`, source snapshots, saved data
splits, per-run `config.json`, `calls.jsonl`, `generations.jsonl`, `result.json`,
and an aggregate `summary.json`. Existing directories are rejected. Call and
generation logs flush incrementally, so completed records survive interruption;
the CLI does not resume interrupted runs. Each call includes its prompt, raw
response, operation, attempt, elapsed time, error/fallback and available usage.
Generation records include populations, fitness, sizes and evaluation counts.
Metadata includes Slick's version, path and source hashes, Python version,
requested model and sampling configuration; API calls record the resolved model.

`--max-calls`, `--seconds`, and `--timeout` bound LLM calls per run. Retries (two
by default) count toward the call budget and back off for 1, then 2 seconds.
Malformed structured responses fall back immediately. Budget termination retains
evaluated candidates and marks an incomplete generation. Cancellation propagates.
No early stop occurs on a perfect solution, preserving the paper's FE budget.

API token counts are observed from the SDK response via a small Slick provider
subclass. Codex and demo token counts are **unknown**, not estimated from text.
Cost is `null` unless no paid calls occur or both `--input-rate` and
`--output-rate` are supplied (USD/million tokens). This simple estimate charges
all input/output tokens at those rates; cache discounts and other billing items
are not modeled. It is accounting, not a dollar spending cap. Use the call cap
and token limit to bound an experiment. Failed calls with unavailable usage make
the total cost unknown. Model/provider failures and fallbacks are visible in the
terminal summary and logs; inspect them before interpreting a run scientifically.

## Attribution

Method and adapted prompt wording: Hemberg, E., Moskal, S. and O'Reilly, U.-M.,
*Evolving code with a large language model*, published September 12, 2024.
The user-supplied article is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
This is a new Python/Slick implementation with the changes described above.
The authors' [Tutorial_GP-LLM repository](https://github.com/ALFA-group/Tutorial_GP-LLM)
provides their original tutorial implementation. Live models were not called
during implementation verification.

## Prompt templates

Model instructions live in this implementation's own `prompts/` directory.
The CLI sets Slick's absolute template root once at startup. Programmatic callers
configure it before rendering or running the algorithm:

```python
from pathlib import Path

from slick import prompts
import llm_gp.operators as operations

prompts.TEMPLATE_ROOT = Path(operations.__file__).resolve().parent / "prompts"
```

The root is process-global; run implementations with different roots in separate
processes. Prompt wording, output parsing, and caller-owned validation are unchanged.
