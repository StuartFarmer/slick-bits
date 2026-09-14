# INSTINCT

Slick port of instruction optimization using neural contextual bandits. It
constructs a soft-prefix domain, represents each prefix with the inducing model's
hidden states, and searches using diagonal neural UCB.

Official sources inspected:

- [Repository and paper](https://github.com/xqlin98/INSTINCT)
- [Induction loop and hidden-state domain](https://github.com/xqlin98/INSTINCT/blob/main/Induction/experiments/run_neural_bandits.py)
- [NeuralTSDiag.select/train](https://github.com/xqlin98/INSTINCT/blob/main/Induction/experiments/LlamaForMLPRegression.py)

`INSTINCT(task, provider, evaluate, condition, hidden_states, surrogate)` requires:

- An async scalar evaluator with finite higher-is-better scores.
- `condition(provider, prefix) -> Provider`, performing **actual soft-token
  embedding conditioning**. The adapter reshapes the projected flat vector for
  its model. Ordinary text APIs cannot provide this by printing numeric vectors.
- `await hidden_states(prefixes) -> ndarray`, returning `(domain_size, features)`
  inducing-model hidden states for the batch of projected prefixes. Extraction
  must use the same task demonstrations and prefix placement as generation.
- `surrogate(theta, contexts) -> (means, jacobians)`, a pure numerical model
  evaluation returning `(n,)` predictions and `(n, parameter_count)` parameter
  derivatives. It must not fit, acquire, or choose arms. The official model is
  a scalar ReLU MLP with 100 hidden units; callers can supply its forward and
  derivative implementation without coupling this port to GPU frameworks.

`run(demonstrations, projection, initial_theta, ...)` uses a scrambled Sobol
domain, computes the fixed random projection locally, extracts hidden-state
contexts once, and measures random initial arms. As in the released loop, the
first UCB acquisition uses the untrained initial surrogate. Initial seeds enter
training when the first acquired reward is added, and do not update UCB precision.

At each subsequent decision, local code computes
`mu + sqrt(sum(lambda * nu * gradient**2 / U))`, updates `U` with the selected
parameter-gradient squares, decodes and measures that arm, then resets model
parameters to their original values and fits all observed rewards. Training
standardizes contexts and uses full-batch squared error, Adam and weight decay
`lambda / observations`, matching the selected source implementation. U persists
across neural refits. Optional random domain subsets limit acquisition work.
Repeated arms and repeated instruction text are allowed; text scores are cached.

```python
from pathlib import Path
from slick import prompts
from instinct import INSTINCT

prompts.TEMPLATE_ROOT = Path("instinct/prompts").resolve()
agent = INSTINCT(task, provider, evaluate, condition, hidden_states, surrogate)
result = await agent.run(demonstrations, projection, initial_theta)
```

Intentional adaptations: NumPy implements the local optimizer and statistics;
SciPy supplies Sobol points. The caller supplies model parameters, derivatives,
projection initialization and hidden-state pooling instead of this package
constructing Vicuna/PyTorch/BackPACK. The generic induction template is not a
byte-for-byte task benchmark prompt. `iterations` means acquisitions after the
initial samples; the source CLI's `total_iter` includes initialization. Seeded
NumPy sampling differs from source global RNG ordering. The neural-UCB and
reset-and-refit mechanics are preserved; a linear surrogate used in tests is
only a deterministic derivative fixture, not a substitute claimed to reproduce
the paper's neural model.

Malformed generation, nonfinite model outputs, measurement and transport errors
propagate. Counters cover generation attempts, evaluations, cached scores and
refits. No model downloads, paid calls, benchmark execution or reproduction
claims. Configure the process-global template root once before running.

Check: `rtk proxy optimizer/.venv/bin/python -m unittest tests.test_instinct`.
