# LLM-GA

Slick implementation of **Llm-ga: A gradient-based multi-label adversarial attack
by large language models** (Liu et al., 2025), generalized to caller-defined tasks.
It evolves an idea and its implementation through E1 (different ideas), E2
(shared ideas), M1 (structural refinement), and M2 (parameter tuning). Each
operator receives its own blacklist of rejected candidates, with preparation
instructions corresponding to the paper's Problem–Thinking–Solution prompts.

## Use

Use the existing Slick installation. From the repository root:

```python
from pathlib import Path

import llm_ga
from slick import prompts

# Configure once at application startup, before any model calls.
prompts.TEMPLATE_ROOT = Path(llm_ga.__file__).resolve().parent / "prompts"

async def search(provider, evaluate):
    agent = llm_ga.LLMGA(
        task="Design a heuristic that minimizes weighted job completion time.",
        provider=provider,
        evaluate=evaluate,
        template="def priority(duration, weight):\n    # Implement the priority score.\n",
        information="Positive numeric inputs. Return a numeric priority; larger runs first.",
        requirements="Return only the function. Use no imports or external state.",
    )
    population = await agent.run(population_size=8, generations=20, parents=2, seed=2024)
    return population[0], agent
```

Supply a Slick provider and `async evaluate(content: str) -> float`. **Lower
fitness wins**; negate a reward if your problem is naturally a maximization.
Change `task`, `template`, `information`, `requirements`, and the evaluator for
another problem. Content can also be prose or another textual artifact. Empty
optional context fields are supported. No image, model, dataset, tensor library,
or attack signature is built into the optimizer.

The evaluator owns syntax/interface checks, execution isolation, deadlines, and
fitness measurement. It should raise `CandidateRejected("reason")` for an invalid
candidate and `TimeoutError` for an evaluation deadline. Convert exceptions from
generated code at that boundary; unrelated evaluator errors propagate. The agent
never executes generated content. For code tasks, enforce the required isolation
in your evaluator; a prompt instruction does not provide it.

`Proposal` contains `description` and `content`; returned `Individual` objects
also have `id` and finite `fitness`. Descriptions are trimmed, but candidate
content preserves its whitespace. Structured JSON and nonblank fields are
checked by Slick/Pydantic; those checks do not prove candidate correctness.

## Search and failure policy

- Defaults: N=8, 20 evolution generations, two exploration parents. Initialization
  fills N evaluated slots, with at most `3*N` attempts unless `init_attempts` is
  supplied. Failure to fill the population raises with partial records retained.
- Each generation performs N attempts for each of E1/E2/M1/M2, in that order.
  Exploration samples the requested number of parents; modification samples one.
  Sampling uses `random.Random(seed).choices`, with replacement and rank weights
  `1/(one_based_rank + N)`, following the official implementation.
- Each operator batch sees a fixed population and worst-fitness threshold.
  Candidates must score **strictly below** that threshold to enter the survivor
  pool. The best N incumbents and eligible offspring survive after each batch.
  Stable ties favor incumbents. Equal scores do not collapse the population.
- Invalid generated JSON, blank content, evaluator rejection, evaluation timeout,
  and nonfinite fitness consume their attempt. Evolution failures and candidates
  at or above the worst fitness enter only their operator's blacklist. Subsequent
  calls see that feedback, including later calls in the same sequential batch.
  An eligible candidate later displaced by selection is not blacklisted.
- Blacklists are prompt feedback, not a hard duplicate prohibition. All blacklist
  records are included; long searches need enough model context. There is no
  silent truncation, score caching, duplicate retry, or M3 simplification operator.
- Provider errors, provider timeouts, and unexpected evaluator errors abort after
  recording the error. Provider setup, transport retries, and raw transport
  logging belong to the caller. Every proposal attempt invokes the supplied
  provider once; a provider's internal retries are outside this count.

`agent.attempts` records IDs, parent IDs, operation, generation, raw model output
before parsing, parsed proposals, measured finite fitness, status, and failures.
`accepted` means eligible for survival, not necessarily retained in the final
population. `agent.evaluations` counts evaluator invocations, including failures.
`agent.blacklists` contains rejected evolution records by operator, and
`agent.history` contains immutable population snapshots after initialization and
each generation. A successful default run takes 648 calls with no initial
rejections, at most 664 with the default initialization budget.

Each run resets state and RNG. Calls are sequential and independent, with exactly
one `provider=` argument per decorated operation and no shared Session history.
Use one active run per instance. Slick's template root is process-global; configure
it once and use separate processes for concurrent applications with different roots.

## Official implementation and deliberate differences

Reference: [liuyujiang123/LLM-GA](https://github.com/liuyujiang123/LLM-GA), inspected
at commit `1f59d10bdcc722e11712ca2af78c093f1a78c9e6`. This implementation uses its
operator scheduling and parent-selection algorithm as source references:

- [eoh.py](https://github.com/liuyujiang123/LLM-GA/blob/1f59d10bdcc722e11712ca2af78c093f1a78c9e6/eoh/methods/eoh/eoh.py):
  N offspring per operator and survivor replacement after each operator batch.
- [prob_rank.py](https://github.com/liuyujiang123/LLM-GA/blob/1f59d10bdcc722e11712ca2af78c093f1a78c9e6/eoh/methods/selection/prob_rank.py):
  rank-weighted parent sampling with replacement.
- [eoh_interface_EC.py](https://github.com/liuyujiang123/LLM-GA/blob/1f59d10bdcc722e11712ca2af78c093f1a78c9e6/eoh/methods/eoh/eoh_interface_EC.py):
  operator-local bad-idea lists and evaluation dispatch. The active import selects
  `eoh_evolution_origin`, and `_get_alg` does not pass the recorded bad-idea lists
  into its prompts. We implement the feedback path described in the paper.
- [eoh_evolution.py](https://github.com/liuyujiang123/LLM-GA/blob/1f59d10bdcc722e11712ca2af78c093f1a78c9e6/eoh/methods/eoh/eoh_evolution.py):
  task/interface prompt components and bad-idea prompt text. Its evolution wrappers
  omit the required `bad_ideas` argument when calling their prompt builders.
- [pop_greedy.py](https://github.com/liuyujiang123/LLM-GA/blob/1f59d10bdcc722e11712ca2af78c093f1a78c9e6/eoh/methods/management/pop_greedy.py):
  lower-fitness elitism. We preserve N candidates even with tied scores, instead
  of deduplicating scores and potentially shrinking the population.

The paper describes random parents and generation-level selection less precisely
than the code; we use the code's rank sampling and per-operator replacement.
For acceptance, we use the paper's explicit strict `< worst` rule instead of the
code's `> worst` blacklist test. Fitness is not rounded to five decimals. Calls
are sequential, so blacklist updates are visible immediately; there are no
joblib workers, global counters, or hidden duplicate/parse retries.

This is an intentional problem-agnostic adaptation, not a verbatim prompt port.
The paper's image-specific instructions become caller-supplied context, and the
braced idea plus Python response becomes typed JSON. Thinking instructions remain
in the operation templates; only a concise idea and implementation are returned.
The paper's attack fitness `a*(labels-positive_labels) - b*success_rate + c*L2`
(a=1, b=10, c=0.5), attack helpers, and dataset splits belong in a task evaluator.
No official source files are vendored or required at runtime.

## Verification

From the repository root, using a Python environment containing Slick:

```sh
python -m unittest tests.test_llm_ga
```

Tests use the shared `tests/providers.py` scripted provider. They verify scheduling,
sampling, batch snapshots, blacklist routing, strict acceptance, budgets, failure
accounting, raw rejected output, repeated runs, and all external templates.
These are offline algorithm/interface checks, not a reproduction of the paper's
attack-success rates or evidence of improved optimization with a live model.
