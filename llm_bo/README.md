# LLM-BO / LLAMBO

The bibliography's **Large Language Models to Enhance Bayesian Optimization**
([ICLR 2024 paper](https://openreview.net/forum?id=OOxotBmGol),
[arXiv 2402.03921](https://arxiv.org/abs/2402.03921)) is **LLAMBO**. This folder
implements its discriminative-surrogate pipeline from official
[tennisonliu/LLAMBO, master](https://github.com/tennisonliu/LLAMBO):
`llambo/llambo.py`, `acquisition_function.py`, `discriminative_sm.py`, and the
warm-start stage. It does not substitute a Gaussian-process optimizer.

LLAMBO induces initial configurations, samples candidates conditioned on a desired
objective, predicts each candidate's outcome repeatedly under shuffled observation
contexts, fits its predictive mean/population standard deviation, and maximizes
Gaussian expected improvement. Only the selected candidate is truly evaluated.
The target is `best ± alpha * observed_range`; `alpha=-0.2` matches the release's
default and targets a value slightly worse than the current best. Optional target
jitter samples between target and best. Predicted standard deviation has the
source's `1e-5` floor.

```python
from pathlib import Path
from slick import prompts
from llm_bo import LLMBO

prompts.TEMPLATE_ROOT = Path("/absolute/path/to/llm_bo/prompts")
agent = LLMBO(task, provider, evaluate, bounds={"temperature": (0.0, 1.0)})
result = await agent.run(initial_configurations, candidates=5, predictions=10)
best = result["best"]
```

`evaluate(configuration)` is async and lower-is-better by default; set
`lower_is_better=False` for maximization. `task` carries domain/feature semantics.
Bounds define a continuous numerical configuration domain. Generated missing,
extra, out-of-range, nonfinite and duplicate configurations are rejected with
records. Attempts are bounded; no novel candidate ends the search. Supplied initial
configurations are caller-owned inputs and are measured as given. An entirely
rejected warm start returns `best=None` without inventing observations.

Explicit adaptations: structured candidate JSON replaces source ad-hoc parsing;
prediction retains `## NUMBER ##`. Context shuffling replaces the release's
multi-template batching. Objective-specific [0,1] clipping, integer/log feature
warping, recalibration, and the separate generative 0/1-logprob surrogate are not
included. Domain transformations can be expressed in the callback/task bounds.
Invalid prediction/provider errors propagate rather than using source NaN
imputation; candidate sampling has a caller-controlled finite attempt bound.

The result contains actual observations, surrogate means/std/EI, targets, selected
points, rejections and generation/evaluation counts. Dependencies: Slick, Pydantic,
NumPy and SciPy. Tests verify uncertainty-sensitive EI, conditioning target,
out-of-domain rejection, duplicate-attempt exhaustion and actual query counts.
These are algorithm checks, not benchmark reproduction.
