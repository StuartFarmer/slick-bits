# AutoHint

This is a **paper-based reconstruction**. No official code link was found in the
paper or survey bibliography. The discovered `prapti19/prompt_autohint` repository
is not identified as an author release and is not used as an official source.

`AutoHint(task, provider, evaluate, correct=..., cluster=..., actor_provider=...)`
first predicts every training example and keeps only incorrect residuals. It
then generates a hint for **every residual before sampling**, summarizes sampled
hints, and appends the resulting hint to the current instruction. Iterations
repeat on the newly enriched prompt; separate validation scores choose the
best-ever prompt.

```python
from autohint import AutoHint, Example
result = await AutoHint(task, provider, evaluate).run(
    initial_prompt, [Example(input_text, target_text)],
    iterations=1, sampling="random", sample_size=5,
)
```

`evaluate(prompt)` asynchronously returns a finite validation score, higher is
better. `correct(prediction, target)` defaults to exact text equality; provide
task-specific answer parsing here. Optional `actor_provider` separates task
inference from hint generation. No residual errors ends the search without a
hint-generation call. Results distinguish scalar evaluations, inference calls,
optimizer calls, all residual hints, sampled hints, current prompt, and best-ever
prompt. No hint-generation data is automatically drawn from validation/test data.

Sampling modes preserve the paper's distinction: `random` draws up to
`sample_size` total hints, `balanced` draws that many per target class, and
`cluster` draws that many per caller-supplied cluster. For clustering, supply
`cluster(hints) -> group_ids`; the paper combines encoded input, label, and hint
vectors with weights then applies K-means. This callback owns that encoder,
weights, and clustering implementation; the agent still performs per-group random
selection. The callback is not called in random or balanced modes.

Source inspected: [paper](https://arxiv.org/abs/2307.07415), Algorithm 1,
Sections 3.2-3.5, Eq. 3, and the multi-iteration discussion in Section 4.3.

Adaptations: local prompts restate the hint-generation and summarization tasks;
model construction, answer parsing, and optional BERT/K-means are caller-owned.
The summary is appended as `Hint: ...`, preserving earlier instructions and hints.
The paper's separate hyperparameter search over sampling strategies is not run
implicitly; callers can compare configurations using validation scores. All
residual hints are generated as written in Algorithm 1, so a large residual set
can still be expensive despite a small summary sample.

Configure Slick's process-global template root before use:

```python
from pathlib import Path
from slick import prompts
import autohint
prompts.TEMPLATE_ROOT = Path(autohint.__file__).parent / "prompts"
```

Requires the repository's Slick installation. Provider/model construction,
benchmarks, credentials, persistence, and execution isolation belong to callers.
All model calls are sequential, and provider/evaluator failures propagate without
retries. Blank optimizer prose raises before scoring. Use one active run per
agent instance. Deterministic tests check algorithm mechanics, not benchmark
performance or reproduction of the published numerical results.
