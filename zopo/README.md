# ZOPO

Slick port of zeroth-order prompt optimization using a local Gaussian process
with an empirical neural tangent kernel, continuous ascent and discrete projection.

Official sources inspected:

- [Repository and paper](https://github.com/allen4747/ZOPO)
- [ZOPO fitting, ascent, restarts and exploration](https://github.com/allen4747/ZOPO/blob/main/InstructOpt/experiments/optimization.py)
- [GP_NTK and empirical neural tangent features](https://github.com/allen4747/ZOPO/blob/main/InstructOpt/experiments/learner_diag.py)
- [Pool initialization and evaluation loop](https://github.com/allen4747/ZOPO/blob/main/InstructOpt/experiments/opt_instruct.py)

`ZOPO(task, provider, evaluate, embed, features)` requires an async finite,
higher-is-better evaluator and an async batch instruction embedder. `features(x)`
provides the fixed neural model's parameter Jacobians `(n,p)` and their input
derivatives `(n,p,d)`. The latter are the derivatives of those parameter features
with respect to input embeddings; ordinary sentence embeddings alone do not
provide them. The source uses a fixed randomly initialized ReLU network. The
callback computes model primitives only, never a GP posterior or search step.

`run(initial_prompts, ...)` optionally induces additional pool instructions,
deduplicates and embeds the pool, and evaluates random seeds. It fits a local
GP using the nearest observed embeddings, coordinate normalization, the kernel
`J(x) @ J(z).T`, and observation-noise variance. The agent analytically calculates
posterior mean, variance and input gradient, takes Adam ascent steps, and projects
to the nearest unqueried instruction embedding. It retains Adam state across
queries, restarting at the best observation after stagnation.

High posterior uncertainty accumulated across rounds, or a zero gradient,
triggers local neighbor evaluations and refitting. A high-uncertainty first step
can still make one projection, matching the source rule. The caller's budget
includes seed, ascent and exploratory evaluations. Results expose each measured
instruction with its phase and every attempted continuous update.

```python
from pathlib import Path
from slick import prompts
from zopo import ZOPO

prompts.TEMPLATE_ROOT = Path("zopo/prompts").resolve()
agent = ZOPO(task, provider, evaluate, embed, features)
result = await agent.run(candidate_instructions)
```

Intentional adaptations: NumPy computes GP solves and Adam; neural derivatives
and sentence embeddings are injected. A simple optional induction stage can
build the fixed pool in-process; the source loads a previously generated pool.
Stable deduplication is by text, so equal embeddings can represent different
instructions. Initial sampling uses a local NumPy RNG. Exact derivative-vector
zero replaces the source gradient-sum test, avoiding cancellation of nonzero
components. The finite pool and remaining evaluation budget bound exploration;
the released zero-gradient loop can otherwise revisit an exhausted neighborhood
without progress. Nearby **unqueried** points are sampled directly. These fixes
make exhaustion explicit; they do not claim numerical reproduction.

Nonfinite measurements, embeddings or model derivatives fail explicitly.
Linear-algebra and dependency errors propagate, replacing source broad exception
fallbacks and interactive breakpoints. No benchmark execution, model downloads,
paid calls or paper-performance claim. Configure Slick's process-global template
root once before execution.

Check: `rtk proxy optimizer/.venv/bin/python -m unittest tests.test_zopo`.
