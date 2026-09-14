# BBTv2

Implements divide-and-conquer deep prompt search from [BBTv2](https://aclanthology.org/2022.emnlp-main.259/).
Inspected the official [deepbbt.py](https://github.com/txsun1997/Black-Box-Tuning/blob/main/deepbbt.py),
particularly its final layer loop: independent persistent CMA instances optimize
one layer at a time while other layers remain fixed, and install each layer's
best-ever latent after its generation. Historical layer losses can therefore have
different other-layer contexts; this source behavior is retained explicitly.

`BBTv2(task, evaluate, initial_prompts, projections=...).run(...)` accepts a
layer-first soft-prompt array and async lower-is-better evaluator. Supply one
projection per layer, or pass `layer_stds` to generate matrices using the source
Gaussian scale `alpha * layer_std / (sqrt(d) * sigma)`. The caller computes the
frozen model's input/hidden-state statistics (the release clips hidden-state
outliers before estimating those statistics). Inserting prompts into transformer
layers is part of the forward evaluator, not simulated here.

Only complete layer sweeps fit within `budget`; one additional final evaluation
measures the assembled prompt, since per-layer historical losses do not determine
its current score. This extra call is included in `evaluations`. The returned
`prompts`, `loss`, projections and history describe that final composite, not the
release's separate development-checkpoint selection. This port reuses the
positive-weight standard CMA core in `bbt/cma.py`; active-CMA negative weights,
bound transforms and library termination heuristics are omitted. Dependencies:
NumPy and the sibling `bbt` package. Tests check changing layer context, persistent
states, sweep order, and exact final measurement count. No language generation
occurs, hence no Jinja templates or provider dependency.
