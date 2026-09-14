# InstructZero

Slick port of instruction-coupled Bayesian optimization over soft-prompt latent
vectors. The inducing model generates hard instructions, and an injected task
evaluator measures their performance.

Official sources inspected:

- [Repository and paper](https://github.com/lichang-chen/InstructZero)
- [Search, projection, caching and performance vectors](https://github.com/lichang-chen/InstructZero/blob/main/InstructZero/experiments/run_instructzero.py)
- [CombinedStringKernel and EI optimization](https://github.com/lichang-chen/InstructZero/blob/main/InstructZero/experiments/instruction_coupled_kernel.py)

Required dependencies are NumPy, SciPy, Slick and caller-supplied model resources.
No dependencies or model weights are installed by this module.

`InstructZero(task, provider, evaluate, condition)` takes two model boundaries:

- `condition(provider, projected_prefix) -> Provider`: return an adapter that
  prepends the numeric soft-token embeddings during generation. The adapter owns
  reshaping the flattened vector into tokens and model hidden dimensions. **A
  normal text-only API or embedding vector printed in the template cannot supply
  this capability.** Projection is performed locally as `projection @ latent`.
- `await evaluate(instruction) -> Evaluation(score, behavior)`: return a finite
  higher-is-better objective and a finite NumPy vector of per-example performance
  on the same ordered validation examples. `behavior` is a performance vector,
  not a sentence embedding. The caller owns execution and dataset separation.

`run(demonstrations, projection, ...)` evaluates scrambled Sobol initial latents
in `[0,1]^d`, matching the source. At least two initial samples are needed for
sample standardization. Each iteration locally fits a Gaussian process with
the source coupled covariance

```text
K(x,z) = scale * k_latent(x,X) A^-1 K_behavior A^-1 k_latent(X,z)
A = k_latent(X,X) + 0.0001 I
```

Both component kernels are Matérn 5/2. A zero-mean GP is fit to standardized
scores; marginal likelihood optimizes latent length, behavior length, scale and
noise. Analytic expected improvement is maximized from the best observed
starting points. The resulting batch is ordered by EI and evaluated against the
same frozen surrogate before refitting. Exact repeated instruction text reuses
its score and behavior vector. Duplicate latent observations remain in the GP.

```python
from pathlib import Path
from slick import prompts
from instructzero import Evaluation, InstructZero

prompts.TEMPLATE_ROOT = Path("instructzero/prompts").resolve()
agent = InstructZero(task, provider, evaluate, condition)
result = await agent.run(demonstrations, projection)
```

Intentional numerical adaptations: installed SciPy's bounded multistart L-BFGS-B
replaces the source CMA-ES EI optimizer, so local acquisition optima can differ.
GP fitting uses isotropic component lengthscales, bounded log-parameters and
maximum likelihood rather than GPyTorch's ARD scales, learned constant mean and
Gamma priors. The instruction-coupled covariance, standardization, EI and model
feedback remain explicit. A finite GP fit is retained even when the numerical
optimizer reaches its convergence limit; `fits` records that status. These are
algorithm-preserving structural adaptations, not numerical reproduction claims.
The caller supplies the fixed random projection instead of the module reading
an inducing model's embedding statistics to initialize one.

Generated blank instructions, nonfinite measurements/model optimization output
and dependency failures propagate. Attempts are counted before calls. Returned
records distinguish model generations, evaluator calls and cache hits. No task
benchmark runner, paid model call, or claimed published score. Configure Slick's
process-global template root once before execution.

Check: `rtk proxy optimizer/.venv/bin/python -m unittest tests.test_instructzero`.
