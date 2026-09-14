# PromptBreeder

Paper-based port of [Fernando et al., section 3](https://arxiv.org/html/2309.16797v1)
([ICML 2024 publication](https://proceedings.mlr.press/v235/fernando24a.html)).
No official repository was located; community implementations were not used.

Preserved mechanisms: paired task/mutation genomes, thinking-style initialization,
binary tournament loser replacement, direct zero/first-order mutation,
diversity-filtered EDA and ranked EDA, elite lineage, zero/first-order
hypermutation followed by task mutation, successful-working induction, context
shuffling, and additional 10% fitness-proportional task-prompt crossover.

Adaptations: JSON responses and rewritten templates; one random prompt slot
changes per event; lineage records the first prompt; each tournament rescores
competitors independently; context maintenance uses bounded replacement and
shuffling, omitting the paper's separate probabilistic whole-context resampling.
This is a mechanism port with explicit differences, not the full original experiment.

```python
from pathlib import Path
from slick import prompts
from promptbreeder import PromptBreeder, Evaluation

prompts.TEMPLATE_ROOT = Path("promptbreeder/prompts").resolve()
# evaluate(unit) -> Evaluation(score, correct_workings)
# similarity(text_a, text_b) -> embedding cosine similarity
result = await PromptBreeder(task, provider, evaluate, similarity).run(mutations, styles)
```

Both callbacks are async. Evaluator executes the prompt sequence/context and owns
training-batch coordination. Scores must be finite/nonnegative; similarity finite.
Lamarckian selection without correct workings raises explicitly. Errors propagate
without retries; callers can choose an operator subset for ablations. Configure
the process-global root once. Tests establish mechanics, not reported performance.
