# GreenTEA

Problem-agnostic implementation of [GreenTEA: Gradient Descent with Topic-modeling
and Evolutionary Auto-prompting](https://arxiv.org/abs/2508.16603).
Supply seed instructions, a task description, Slick providers, an async evaluator,
and a text encoder. The optimizer returns the best instruction and final population.

```python
from pathlib import Path

from slick import prompts

import greentea
from greentea import GreenTEA, KMeansTopics


async def optimize(task, provider, evaluate, encode, initial_prompts):
    # Configure once at application startup, before rendering or generation.
    prompts.TEMPLATE_ROOT = Path(greentea.__file__).resolve().parent / "prompts"
    agent = GreenTEA(task, provider, evaluate, KMeansTopics(encode))
    result = await agent.run(initial_prompts, iterations=20, seed=0)
    return result["best"].prompt
```

Install from this directory with `pip install -r requirements.txt` (the requirements
refer to the adjacent Slick checkout). No model weights or datasets are downloaded
by this package. `encode` can be an existing sentence-transformer's `.encode`, a
local encoder, or a synchronous embedding API adapter.

## Caller contracts

- `task` describes the desired behavior and output constraints. Supply a nonempty
  sequence of distinct initial instructions; its length is the population size K.
  The paper uses K=4 and 20 generations. Initial prompts are supplied directly.
- `evaluate(instruction)` returns `Evaluation(score, errors)` asynchronously.
  Fitness must be finite and nonnegative; higher is better. Convert costs or losses
  to a suitable nonnegative fitness in your evaluator.
- Each `ErrorCase(input, expected, predicted, justification="")` describes one
  failed training example. Include the reference answer **and its explanation**
  in `expected`; this text is embedded for topic modeling. For label-only tasks,
  provide meaningful reference explanations when available. `justification` is
  the predictor's explanation. The caller decides correctness, so the algorithm
  works with classification, free text, code, or other outputs represented as text.
- The evaluator owns the predictor, dataset, parsing, scoring, and any execution
  isolation. Keep held-out evaluation outside optimization. The paper's predictor
  XML wrapper is intentionally left to the evaluator, allowing arbitrary models
  and output formats. Configure predictor temperature 0 and optimizer temperature
  1 in the caller to follow the paper's experimental settings.
- `encode(texts)` returns one numerical embedding per text. `KMeansTopics` returns
  selected error indices. To use a different topic model, supply a callable
  `topics(reference_texts, rng) -> indices` instead of `KMeansTopics(encode)`.
- `provider` generates crossover and mutation responses. Optional
  `analyzer_provider=` uses a separate model for feedback. All calls are independent
  and sequential, with explicit `provider=`; the optimizer creates no sessions.

For example, an evaluator can return
`Evaluation(0.75, (ErrorCase("input", "reference explanation", "wrong output"),))`.
Return **all** failures, not just preselected examples; the optimizer chooses the
major topic. Its empty default `errors=()` means no failures were reported.

## Algorithm and results

1. Evaluate each seed and collect feedback on its largest error topic.
2. Sample two parents with replacement using fitness-proportional roulette.
3. Cross over the parents, then mutate using both parents' examples and feedback.
4. Evaluate K children against the same training objective.
5. Merge parents and children, deduplicate by exact instruction, and keep the best
   K. Repeat for exactly `iterations` generations, evaluating the final children.

K-means follows the author's implementation: sample at most 100 failures, choose
the cluster count from 5–20 (limited by available distinct vectors) by minimizing
`inertia / total_inertia + 0.02 * k`, then refit and take at most 10 examples from
the largest cluster. These topic settings are constructor arguments. Model fits
use fixed seeds 0 and 10; the run's local RNG controls sampling and roulette.

Evaluation and feedback are cached by exact instruction per run. Use a fixed
training set and repeatable evaluation; cached results are not refreshed for noisy
objectives. No failures means no embedding or analyzer call. All-zero fitness uses
uniform selection; ties retain incumbents in stable order. Identical embeddings
form one topic without division by zero. Duplicate children consume generation
calls but reuse their cached evaluation and cannot displace distinct incumbents.

Results include `best` and `population` (`Candidate` objects with prompt, score,
selected examples, and feedback), initial and per-generation score `history`,
`evaluations`, `optimizer_calls`, `cache_hits`, and raw generation `responses`.
Counters count attempted calls; they remain on the agent after failure. With K
distinct seeds and T generations, evaluation calls are at most `K * (T + 1)`,
crossover/mutation calls are exactly `2 * K * T` on success, and analyzer calls
are at most one per uncached evaluation with selected errors.

Malformed, blank, or repeated required output tags raise `ValueError` before child
evaluation. Every returned raw response is retained before tag validation.
Provider, encoder, and evaluator exceptions propagate; there are no automatic
retries or fallback prompts. Transport retries and logging responses that never
reach Slick's decorated body belong to the provider. Each run resets agent state.

Slick's template root is process-global. Absolute startup configuration works
from the repository root or another working directory when this package is on
`PYTHONPATH`; concurrent applications with different roots need separate processes.

## Upstream provenance and deliberate differences

Adapted from the [author's repository](https://github.com/McDaniel7/GreenTEA) at
commit [`b1a76514b392a1f6789c0d96ac5c3965751a949a`](https://github.com/McDaniel7/GreenTEA/tree/b1a76514b392a1f6789c0d96ac5c3965751a949a).
The upstream MIT notice is preserved in [LICENSE.upstream](LICENSE.upstream).

- [`utils/text_utils.py`](https://github.com/McDaniel7/GreenTEA/blob/b1a76514b392a1f6789c0d96ac5c3965751a949a/utils/text_utils.py):
  adapts `KmeansTopicCluster` and its penalized-inertia criterion. Although the
  paper mentions nearest neighbors, released code uses K-means. Embedding batches
  replace its per-example calls; no sentence-transformer model is hardcoded.
- [`utils/ga_utils.py`](https://github.com/McDaniel7/GreenTEA/blob/b1a76514b392a1f6789c0d96ac5c3965751a949a/utils/ga_utils.py):
  follows `ParentPromptSelectorWheel` and `Evolutor.evolute` for selection,
  parent/child merging, and unique top-K retention. We cache repeated children
  per Algorithm 1; upstream reevaluates them. Stable ties replace unordered sets,
  and a local Python RNG replaces global NumPy randomness.
- [`models/prompt_generator.py`](https://github.com/McDaniel7/GreenTEA/blob/b1a76514b392a1f6789c0d96ac5c3965751a949a/models/prompt_generator.py):
  retains the guided generator's parent/example/feedback inputs and
  `<OptimizedPrompt>` contract. Slick's required separate operations split the
  paper's combined generation into two calls with `<ChildPrompt>` as the
  intermediate result. Feedback is applied in mutation. This changes the model
  context and doubles generation calls per child; it is not prompt-equivalent.
- Upstream references external prompt files it does not ship. Templates here adapt
  Appendix B, adding caller task context and splitting generation. Analyzer output
  retains the paper's `<erroranalysis>` and `<suggestion>` sections; upstream's
  analyzer instead extracts `<List>`. These are checked tagged text, not JSON.

Dataset loading, Bedrock clients, checkpoint files, random development batches,
benchmark-specific labels, and held-out scoring stay outside this implementation.
This implements the algorithm; it does not reproduce the reported benchmark gains.

## Checks

With dependencies installed, run from the repository root:

```sh
python -B -m unittest tests.test_greentea
```

The checks use the shared scripted provider and real K-means with deterministic
embeddings. They exercise major-topic selection, feedback routing, roulette,
elitism, caching, tagged-output rejection, score failures, and local templates.
They do not call a paid model or measure real optimization quality.
