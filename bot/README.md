# Boosting of Thoughts

Task-agnostic implementation of Chen, Li and Niu's **Boosting of Thoughts:
Trial-and-Error Problem Solving with Large Language Models**, using the paper
supplied with this implementation. The algorithm builds weighted binary trees,
aggregates their best paths, analyzes the resulting chain, and appends the chain
and feedback to the next iteration's prompt. All model instructions live in this
folder's `prompts/` directory. No examples or mathematical task rules are built in.

```python
from pathlib import Path

import bot
from slick import prompts

async def solve(task: str, provider):
    # Configure once at application startup, before concurrent calls.
    prompts.TEMPLATE_ROOT = Path(bot.__file__).resolve().parent / "prompts"
    agent = bot.BoostingOfThoughts(task, provider)
    result = await agent.run(iterations=3, trees=4, max_depth=4)
    return result.answer
```

Supply the question, context, constraints, and desired artifact format in `task`.
The provider is any Slick-compatible object implementing async `acall`. Provider
construction, model choice, transport retries, persistence and any execution of
generated artifacts belong to the application. No generated code is executed here.

`Result` contains the final answer, last aggregated chain, all experiences, and
the number of model calls. `agent.forests` retains each tree's leaf paths (including
their ancestors), growth strategy, sampling settings, and expansion count.
`agent.calls` retains operation names, rendered prompts, raw responses and errors;
external scoring attempts are in `agent.assessments`. Records reset on each run.

## Algorithm choices

- Defaults follow the paper: 10 iterations, 15 trees, maximum depth 5, greedy
  aggregation, growth scores in inclusive `[0.3, 0.8]`, and similarity strictly
  greater than `0.7`. The empty root is not a generated thought. Each expansion
  samples two children independently, then scores each node and incoming edge.
- Trees run concurrently, alternating level-wise and best-leaf-first expansion.
  Best-leaf-first expands the eligible leaf with the highest local node-plus-edge
  score. Either score outside the growth range stops that branch, including high
  scores; it does not establish that the problem is solved.
- Each tree contributes the root-to-leaf chain maximizing the **sum of node and
  edge scores**, as in §3.2. `aggregation="best_first"` selects the highest-scoring
  complete path. Greedy aggregation starts with the strongest first thought,
  then selects the strongest successor whose original parent matches the last
  selected thought. Matching uses an LLM similarity score, with exact text matches
  scoring 1 without a call. Exact repeats are excluded and depth bounds joining.
- Every round appends its chain and the model's analysis, error reports, advice,
  recommendations and confidence. Experiences are accumulated without truncation.
  All rounds run even when the model claims success. A final generation uses the
  last chain and all experience, following §3.2; the last chain is also returned
  separately, as in Algorithm 1. `iterations=0` makes just the final generation.

The paper's appendices sometimes use edge-only scoring, unlike §3.2; this version
consistently uses node plus edge. Greedy similarity can join steps from different
contexts, so the resulting chain still requires analysis; greedy search does not
guarantee a global optimum. Prompts deliberately generalize the paper's math
roles, use JSON for scores, and emphasize checking task state during similarity.
These are documented implementation choices, not an exact prompt reproduction.

## Sampling and external evaluation

Slick's checked provider interface has no portable per-call sampling parameters.
For the paper's temperature/top-p heterogeneity, supply a provider factory:

```python
agent = bot.BoostingOfThoughts(
    task,
    provider,
    tree_provider=lambda temperature, top_p: make_provider(
        temperature=temperature, top_p=top_p
    ),
)
result = await agent.run(seed=42)
```

`make_provider` is application code constructing a provider that actually supports
those settings. The factory receives a temperature sampled from
`[0.2, 0.4, 0.6, 0.7, 0.9, 1.1, 1.5]` and top-p from `[0.1, 0.3, 0.5, 0.7, 0.9]`,
once per tree per iteration. Settings are sampled before launching trees, so the
seed controls their selection independently of request timing. Without the factory,
trees reuse the supplied provider's decoding settings and vary growth order only;
their recorded temperature/top-p are `None`. Without a binding expansion limit,
growth order alone need not change the resulting tree's shape.

Generation uses each tree's provider; scoring, similarity, feedback and final
generation use the main provider. Providers and any external evaluator must support
concurrent calls. BoT uses no mutable conversational Sessions.

To replace both model scoring calls with task-specific evaluation, inject an async
callback returning normalized scores. It receives the entire proposed path:

```python
async def evaluate(steps: tuple[str, ...]) -> bot.Scores:
    node, edge = await assess_path(steps)  # Your evaluation, with any needed isolation.
    return bot.Scores(node=node, edge=edge)

agent = bot.BoostingOfThoughts(task, provider, evaluate=evaluate)
```

Both scores must be finite and in `[0, 1]`, with higher better. This callback affects
tree construction and selection; the model still supplies chain feedback. It does
not verify the final generated artifact automatically.

## Budgets, failures and checks

`max_expansions` optionally caps expansions **per tree**. By default the depth bound
allows at most `2**max_depth - 1` expansions: two generations and four scoring calls
per expansion, plus similarity, one analysis per round, and one final answer.
An external evaluator replaces the four scoring calls with two evaluations.
At a budget cutoff, unexpanded frontier nodes remain eligible leaf paths.
`growth_range` and `similarity_threshold` are configurable algorithm thresholds.

Malformed JSON, blank text, invalid scores, tool requests and provider/evaluator
errors abort without retry. Already captured raw responses and partial records
remain on the instance. A tree failure cancels and awaits the remaining trees before
returning; provider-side requests already sent may still incur costs. There is no
partial-forest fallback or hidden repair budget. Use one run at a time per instance.

Slick's template root is process-global: configure it before use, without import
side effects. Concurrent implementations needing different roots require separate
processes. The absolute-path example works regardless of the launch directory
when the repository is on Python's import path.

From the repository root, using the existing Slick environment:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_bot
../slick/.venv/bin/ruff check bot tests/test_bot.py
```

The scripted-provider checks exercise orchestration, scoring, aggregation, feedback,
sampling configuration, cancellation and template contracts without paid calls.
They do not reproduce the paper's solve rates or establish answer correctness.
