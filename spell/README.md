# SPELL

Problem-agnostic implementation of **SPELL: Semantic Prompt Evolution based on a
LLM** (Li and Wu, 2023). Supply the task, a Slick provider, initial prompts, and
an async fitness evaluator. No task model, dataset, classifier, or execution
environment is built into the optimizer.

```python
from pathlib import Path

import spell
from slick import prompts
from spell import SPELL

# Configure once at application startup, independently of the launch directory.
prompts.TEMPLATE_ROOT = Path(spell.__file__).resolve().parent / "prompts"

async def optimize(task, initial_prompt, provider, evaluate):
    # evaluate(prompt: str) -> float, asynchronously; higher is better.
    agent = SPELL(task, provider, evaluate)
    return await agent.run([initial_prompt] * 20, iterations=500, seed=42)
```

For a shorter experiment, set `iterations=10`. For a diverse starting population,
pass your own sequence of prompts. Population size is the sequence length and
stays constant. Every initial entry is evaluated, including duplicates.

## Algorithm

1. Evaluate the initial population using caller-measured fitness.
2. For each offspring, select parents by exponential-fitness roulette:
   `P(i) = exp(score_i) / sum(exp(score_j))`. Subtracting the largest score before
   exponentiation avoids overflow without changing the probabilities.
3. Give the selected complete prompts and scores to one LLM reproduction call.
   Extract one prompt from the brace-delimited response, then evaluate it.
4. Select the next population from the union of parents and offspring: reserve
   one slot for the best individual and fill the remaining slots by roulette.
5. Repeat for 500 rounds by default.

The default `parent_counts=(1, 1, 1, 1, 1, 2, 2, 2, 2, 2)` generates five
single-parent offspring followed by five two-parent offspring each round.
All offspring in a round use the same parent population; earlier children do
not become parents until survivor selection. There is one reproduction operation,
combining semantic variation and inheritance, rather than separate crossover and
mutation prompts. Initial prompts are supplied directly, without an LLM call.

Roulette samples **with replacement** for parents and survivors. The paper does
not specify replacement semantics or exactly how the elite occupies its slot;
these are explicit implementation choices. The elite stays eligible for roulette
and can appear multiple times. There is no deduplication or fitness cache.

Score scale matters: accuracy as a fraction and accuracy as a percentage produce
different selection pressure under `exp(score)`. Supply finite scores on a
consistent scale; negate costs if minimizing. The optimizer does not normalize,
round, or rescale fitness. Only the supplied evaluator affects selection. Keep
held-out evaluation separate from search and use the same evaluation conditions
across candidate prompts.

## Generation and failure contract

The local `prompts/reproduce.j2` follows the paper's task/definition/reproduction,
scored-parent examples, final request, and additional extraction-instruction order.
The task is caller-supplied; the example prompt says “solve the task” instead of
“classify.” It retains the paper's brief reasoning request and textual curly-brace
format. No JSON schema is imposed on a naturally textual result.

`reproduce()` returns the stripped content of exactly one balanced outer brace
group. Nested braces are preserved so prompts can contain placeholders such as
`{input}` and `{{style}}`. Empty, unmatched, or multiple outer groups raise
`ValueError` before evaluation. Braces are literal delimiters, including inside
quoted text; escaping is not a separate grammar. This stricter extraction contract
is an adaptation of the paper's unspecified `EXT` function.

Malformed output, nonfinite fitness, provider failures, and evaluator exceptions
are recorded and **propagate**, with no retry or fallback. A failed round leaves
`agent.population` at the last completed generation. Successfully scored children
from an incomplete round remain in the audit records but are not selected.
The caller owns transport retries, model configuration, measurement, and execution
isolation. Candidate prompts are never executed by this package.

`run()` returns:

- `best`: immutable `Individual(prompt, score)`.
- `population`: the final population, including any repeated individuals.
- `history`: maximum score at initialization and after each completed round.
- `evaluations`: evaluator call attempts, including calls that failed.
- `optimizer_calls`: reproduction call attempts.
- `attempts`: each offspring's generation, parents, raw response, extracted prompt,
  score, status, and error where available.

Raw output is saved before postprocessing. After an exception, inspect
`agent.attempts`, `agent.evaluations`, and `agent.optimizer_calls`; transport failures
have no raw response. A successful default run with 20 seeds makes 5,000 generation
calls and 5,020 evaluator calls. These counts exclude any provider-internal retries.
Each run resets state and its seeded Python RNG. Generation has no shared Session
or implicit conversation history. Use one active run per instance, and configure
Slick's process-global template root before running; different concurrent template
roots require separate processes.

## Sources and verification

The algorithm and reproduction instructions come from the supplied paper,
[arXiv:2310.01260v1, sections 2–3](https://arxiv.org/html/2310.01260v1).
No official SPELL implementation was identified in the supplied paper, the
[arXiv record](https://arxiv.org/abs/2310.01260), the
[author's publication page](https://liyujian.cn/), or a title/GitHub search checked
on 2026-09-14. This is a paper-based implementation, not a translation of verified
author code. The paper's Hugging Face and Qianfan links identify experimental
infrastructure, not official SPELL algorithm repositories.

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_spell
../slick/.venv/bin/ruff check spell tests/test_spell.py
../slick/.venv/bin/ruff format --check spell tests/test_spell.py
```

Offline tests use the shared scripted provider to check roulette odds, elite
preservation, population snapshots, budgets, nested extraction, failure accounting,
and template rendering from another working directory. No reported benchmark
accuracy or real-model improvement has been reproduced.
