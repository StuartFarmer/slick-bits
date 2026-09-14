# MOPO

Implements the three-layer method in [Menchaca Resendiz and Klinger](https://arxiv.org/html/2412.12948v1):
task prompts, evolving combination/paraphrase operators, and fixed prompts that
revise those operators. Uses NSGA-II survival, additional objective specialists,
operator attribution, and word addition/deletion/replacement.

Inspected official [main](https://github.com/YarikMR/MOPO/blob/main/MOPO.py),
[selection](https://github.com/YarikMR/MOPO/blob/main/MOPO/NSGAII.py), and
[utilities](https://github.com/YarikMR/MOPO/blob/main/MOPO/utils.py).
Both variation branches read the same population, as in that main script.

Adaptations: rewritten JSON prompts; whitespace tokenization with injected
mask-token prediction; normalized crowding actually affects truncation; operator
credit uses surviving descendants and random fill to a fixed count. Initial seeds
are evaluated before champion pairing. Duplicate text retains its first measurement.
This is not benchmark reproduction or byte-equivalent upstream behavior.

```python
from pathlib import Path
from slick import prompts
from mopo import MOPO

prompts.TEMPLATE_ROOT = Path("mopo/prompts").resolve()
result = await MOPO(task, provider, evaluate, objectives, fill_mask,
                    required_tokens=["<class>"]).run(seeds, combine_operators, paraphrase_operators)
```

Both callbacks are async. `evaluate(prompt)` owns conditional generation, text
filtering and aggregation, returning finite maximizing objective scores.
`fill_mask(prefix, suffix)` returns one predicted whitespace token. Missing required
tokens or generation/evaluation failures raise; no retries. Configure the global
root once. Results retain vectors and a Pareto subset; tests establish mechanics only.
