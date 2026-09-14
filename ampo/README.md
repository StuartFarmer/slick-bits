# AMPO

Task-agnostic paper reconstruction of **Automatic Multi-Branched Prompt
Optimization** (Sheng Yang et al., 2024), [paper](https://aclanthology.org/2024.emnlp-main.1130/),
especially Algorithm 1 and §§4.1–4.4. No author-maintained implementation was located.

AMPO samples failures, analyzes each separately, summarizes reasons into ranked
error patterns, and selects the most important patterns. Each selected pattern
independently revises the same parent instruction: extend an applicable existing
branch or add a conditional branch. A separate pruning operation removes redundant
and case-specific branches before validation. The highest scoring child becomes
the next iterate; optional early stopping implements pre-pruning when validation
gain is not significant. The best observed instruction is retained separately.

```python
from pathlib import Path
from slick import prompts
from ampo import AMPO

prompts.TEMPLATE_ROOT = Path("/absolute/path/to/ampo/prompts")
result = await AMPO(task, provider, evaluate, failures).run(
    initial_prompt, branches=3, pre_prune=True, min_gain=0.0,
)
instruction = result["best"].prompt
```

`evaluate(text)` asynchronously supplies a finite higher-is-better validation
score. `failures(text)` asynchronously supplies incorrect training observations as
dictionaries containing enough input, expected answer, and model-response context
for analysis. Generation has separate analyzer, pattern summarizer, branch revisor,
and branch-pruning templates. Summarized patterns use structured JSON with a prose
description and finite importance score. Branch instructions remain ordinary text.

The templates deliberately remove benchmark-specific language. Importance ranking
is performed in Python; the model supplies the scores. The paper's significance
condition is exposed as an absolute `min_gain` threshold on the supplied validation
metric. Post-pruning is always used, while `pre_prune=False` disables early stopping.
Python sampling is seeded; model sampling is caller-owned. All pruned proposals
are measured, including duplicates, so the implementation makes no deterministic
score-cache assumption. Blank text, invalid patterns, and nonfinite scores raise;
provider and evaluator failures propagate without hidden retries.

The result exposes best/current candidates, parent/pattern/expanded/pruned history,
and generation/evaluation counts. Dependencies: Slick, Pydantic, and the standard
library. The shared-provider tests verify independent siblings, importance order,
post-pruning before scoring, early stopping, best retention, and generated pattern
rejection; they are algorithm tests, not reproduced benchmark results.
