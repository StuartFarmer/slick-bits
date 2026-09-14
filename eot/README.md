# Zero-shot EoT

Instance-level implementation of [Jin et al., 2024, section 3](https://arxiv.org/html/2402.05376v1).
The paper links [official code](https://github.com/stan-anony/Zero-shot-EoT-Prompting).
Repository and raw README fetches failed during this implementation, so author
code could not be inspected. This is a paper-based port with that provenance gap.

Preserved mechanics: two seed reasoning instructions, one crossover followed by
mutation, model selection from both seeds and both offspring for the current
problem, instruction-guided problem rewriting, solution generation, and separate
answer extraction. This method selects per instance; it does not evolve prompts
using measured training fitness.

Adaptations: the stages use separate Slick calls and checked JSON contracts;
wording and the second default seed are paraphrased. The evaluator measures only
the final answer. It closes over any ground truth and receives `answer: str`.
No answer score influences selection. Provider and evaluator errors propagate,
with no retries. Nonfinite scores and out-of-range generated indices fail.

```python
from pathlib import Path
from slick import prompts
from eot import EoT

prompts.TEMPLATE_ROOT = Path("eot/prompts").resolve()
result = await EoT(task, provider, evaluate).run(problem, answer_format="A single integer")
```

Configure the process-global template root once before execution. Tests use a
scripted provider and establish stage routing, not reasoning accuracy.
