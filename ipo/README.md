# IPO — interpretable vision-language prompts

This is Yingjun Du et al.'s vision-language IPO. `IPO` searches readable templates
containing a class placeholder. It conditions proposals on training-image
descriptions and retrieved prompt/accuracy/loss triples. Retrieval keeps the best
accuracy records and always includes the baseline; final selection also maximizes
accuracy. Loss breaks accuracy ties.

Sources: [NeurIPS paper §4](https://arxiv.org/html/2410.15397v1),
[official optimizer inspected](https://github.com/lmsdss/IPO/blob/main/optimization/opt_utils.py),
[official CLIP trainer inspected](https://github.com/lmsdss/IPO/blob/main/trainers/coop.py).
The code sorts memory by accuracy then inverse loss and substitutes class names
before CLIP encoding. Its image-description generation is a separate script.

Adaptations: task context and image descriptions are injected, replacing dataset
specific wording. JSON lists replace tagged generations. One call proposes the
round's candidates; duplicate strings are scored once per run. Top `memory_size`
records plus the baseline can occupy `memory_size + 1` entries. No convergence
heuristic is added; rounds bound the run. This ports the optimization mechanics,
not the original prompts or benchmark results.

```python
from pathlib import Path
from slick import prompts
from ipo import IPO, Evaluation

prompts.TEMPLATE_ROOT = Path("ipo/prompts").resolve()
agent = IPO(task, provider, evaluate, descriptions=training_image_descriptions)
result = await agent.run("a photo of <CLASS>")
template = result["best"].prompt
```

`evaluate(template)` asynchronously returns `Evaluation(accuracy, loss)` on the
fixed training split. The caller owns the frozen vision-language model, image/text
encoders, class substitution, cross-entropy, and image descriptions produced by a
multimodal model. Empty descriptions support the paper's no-description setting.
The cache assumes deterministic measurements on the same data. Held-out/new-class
evaluation stays outside search. Alternative `class_token` values require a matching
caller baseline. Generated templates must retain the token; measurements must be
finite. Generation, provider, and evaluation failures propagate without retries or
synthetic scores. Returned archive contains all unique measured templates.
Caller configuration is trusted. Set the process-global template root once before use.

Validation: `python -m unittest tests.test_ipo` checks baseline retention, visual
context, loss tie-breaking, caching, class tokens, and failure propagation.
