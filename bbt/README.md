# BBT

Implements the fixed random subspace search from [Black-Box Tuning for
Language-Model-as-a-Service](https://arxiv.org/abs/2201.03514). Inspected official
[bbt.py](https://github.com/txsun1997/Black-Box-Tuning/blob/main/bbt.py): frozen random
projection, Gaussian initialization scaled by embedding standard deviation, and
CMA ask/evaluate/tell. `BBT(task, evaluate, initial_prompt, projection=...).run(...)`
accepts an async **lower-is-better** loss evaluator over soft prompt arrays. No
LLM generation is involved, so no provider or template is required.

The local CMA core implements positive-weight rank-mu/full-covariance adaptation,
both cumulative evolution paths, and cumulative step-size adaptation. This is an
explicit standard-CMA variant: pycma's active negative covariance weights, bounded
coordinate transforms, restarts and automatic convergence stopping are omitted.
The shared implementation is `bbt/cma.py`; BBTv2 and FedBPT reuse only that math.
Projection entries have standard deviation `alpha * embedding_std / (sqrt(d) *
sigma)`, matching the current release. Supply a projection to reproduce particular
weights. The task callback owns model forwarding and the desired CE/hinge/task
loss; no optimizer step is delegated.

`budget` bounds training evaluations to complete populations; an insufficient
budget makes zero calls and returns the initial prompt with `loss=None`. The
reported best is the lowest training loss, not the release's periodically selected
development-accuracy checkpoint. Returned arrays include prompt, latent and fixed
projection, plus generation history and actual evaluations. Nonfinite measured
losses raise. Dependencies: NumPy and Python. Tests verify the search subspace,
budget, rank-weighted mean, learned covariance, and convergence on a rotated
quadratic; no model weights, benchmark claims, or paid calls are included.
