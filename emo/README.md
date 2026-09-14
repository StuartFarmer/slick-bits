# EMO-Prompts

Implements [Baumann and Kramer, 2024, section 3](https://arxiv.org/html/2401.09862v1).
No official implementation repository was located. A catalogue's code link points
to Ollama, the paper's model runtime, rather than EMO-Prompts algorithm code.

Preserved mechanics: random two-parent crossover, random choice among three
mutation instructions, prompt→generated-text phenotype, vector-valued evaluation,
and parent-plus-offspring survival. Both NSGA-II and SMS-style selection are
available. The default offspring count is 20, as in the paper's (10+20) setup.

Adaptations: task-agnostic rewritten JSON prompts; injected text evaluator;
normalized crowding; batch S-metric pruning using exact two-dimensional
hypervolume. SMS removes maximally dominated points from the worst front before
using hypervolume contribution on the non-dominated front. No model hosting or
sentiment classifiers are bundled. This is a paper-based mechanism port.

```python
from pathlib import Path
from slick import prompts
from emo import EMO

prompts.TEMPLATE_ROOT = Path("emo/prompts").resolve()
# async evaluate(generated_text) -> objective vector, all higher-is-better
result = await EMO(task, provider, evaluate, objectives).run(seeds, selection="nsga2")
```

Configure the global root once. `sms` supports two objectives and requires a lower
reference point; NSGA-II supports arbitrary vector length. Generated text is
nonblank and scores finite. Errors propagate without retries. Tests verify
selection and routing, not reported sentiment performance.
