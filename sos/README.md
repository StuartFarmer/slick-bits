# Survival of the Safest (SoS)

Implements [Sinha et al., 2024, Algorithm 1](https://arxiv.org/html/2410.09652v1).
No official code repository was located in the paper or source search.

Preserved mechanics: semantic seed expansion; objective-specific feedback and
improvement; exhaustive optimization of each objective before switching; final
crossover; and local-optimum selection using proximity in the other objectives.
The retained pool is the union of local optima across objectives. Weighted scores
control convergence and final ranking; neighborhood selection keeps full vectors.

Adaptations: generic objective descriptions and error evidence, separate JSON
calls, configurable hard iteration cap and crossover count, exact-text
deduplication. Final crossover samples unordered pairs. These details replace
the paper's benchmark and safety-model infrastructure; selecting a prompt does
not itself establish safety.

```python
from pathlib import Path
from slick import prompts
from sos import SoS, Evaluation

prompts.TEMPLATE_ROOT = Path("sos/prompts").resolve()
# async evaluate(prompt) -> Evaluation(scores=(...), errors=(...))
result = await SoS(task, provider, evaluate, objectives).run(seed, weights=(0.5, 0.5))
```

Configure the process-global root once. Scores and observed-error descriptions
follow objective order; higher finite scores are better. Caller owns comparable
metric scales, evaluation and data splits. Errors propagate without retries.
Results include the ranked pool, objective-round history and evaluation count.
Tests establish neighborhood and transition behavior, not benchmark performance.
