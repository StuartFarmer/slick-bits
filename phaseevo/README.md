# PhaseEvo

Task-agnostic implementation of [Cui et al., 2024, section 3](https://arxiv.org/html/2402.11347v1).
No official code repository was located in the paper or source search.

Preserved mechanics: joint instruction/example candidates; Lamarckian
initialization; examiner→improver feedback; global crossover/distribution
mutation; final semantic mutation; complementary performance vectors using
Hamming distance; elitist population updates and adaptive phase stopping.

Adaptations: meta-prompts are rewritten with JSON contracts. Global operators
are chosen equiprobably. EDA uses an anchor and its two most distant peers;
crossover uses its most distant peer, following section 3.3's complementary-error
description rather than section 3.1's conflicting argmin notation. Phase progress
uses population mean gain with configurable stalled-round tolerance and a hard
cap. Caller-supplied seeds bypass initialization. Examples are text, not a fixed
dataset schema. These decisions are exposed adaptations, not a benchmark reproduction.

```python
from pathlib import Path
from slick import prompts
from phaseevo import PhaseEvo, Evaluation

prompts.TEMPLATE_ROOT = Path("phaseevo/prompts").resolve()
# async evaluate(candidate) -> Evaluation(score, aligned_boolean_outcomes, errors)
result = await PhaseEvo(task, provider, evaluate).run(training_examples)
```

Configure the process-global root once. Outcomes must share development-set order;
errors describe observed failures. Higher finite scores win. Errors propagate
without retries. The evaluator owns model execution and data splits. Tests verify
phase decisions and template contracts, not semantic preservation or accuracy.
