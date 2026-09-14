# SPELL

Task-agnostic implementation of [Li and Wu, 2023, sections 2–3](https://arxiv.org/html/2310.01260v1).
No official code repository was located in the paper or source search; this is
a paper-based port, not a verified translation of author code.

Preserved mechanics: scored whole-prompt reproduction, five single-parent and
five two-parent offspring by default, exponential-fitness roulette for both
parents and survivors, and one retained elite. Each generation reads a frozen
population. Brace-delimited output is checked before evaluation.

Adaptations: task descriptions and meta-prompts are rewritten for general tasks;
caller supplies the initial population and async fitness. Roulette samples with
replacement (the paper leaves this detail unspecified). Higher finite scores win;
use the paper's accuracy scale if comparing its sampling behavior. No caching,
automatic retries, or benchmark/model infrastructure. Errors propagate.

Configure Slick's process-global template root once before calling:

```python
from pathlib import Path
from slick import prompts
from spell import SPELL

prompts.TEMPLATE_ROOT = Path("spell/prompts").resolve()
result = await SPELL(task, provider, evaluate).run(initial_prompts)
```

`evaluate(prompt)` returns a float. Results expose the best individual, population,
best-score history, and evaluation count. Deterministic tests verify mechanics;
they do not reproduce reported performance.
