# MIPRO

Task-agnostic Slick port of joint instruction and demonstration optimization for
multi-stage language-model programs, grounded in the maintained official MIPROv2
implementation. This is explicitly the v2 source variant; the older MIPRO file
is no longer present on the current official branch.

Official sources inspected:

- [Paper](https://arxiv.org/abs/2406.11695)
- [Grounded proposals and joint optimization](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/mipro_optimizer_v2.py)
- [Candidate demonstration-set construction](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/utils.py)

`MIPRO(task, provider, evaluate, execute)` owns teacher acceptance, demonstration
sampling, instruction proposals, categorical acquisition and full-score selection.
Programs are tuples of `ModulePrompt(name, instruction, demonstrations)` in
execution order. The asynchronous `execute(program, training_example)` callback
returns `Trace(score, demonstrations)`, containing a teacher's end-to-end score
and one actual executed demonstration per module. The agent accepts traces above
`bootstrap_threshold` and constructs candidate demonstration sets itself.
`evaluate(program, examples)` executes a complete program on the supplied batch
and returns a finite higher-is-better score. Both callbacks own task execution and
isolation, and must not perform search.

`run(modules, training, validation, ...)` summarizes training data and generates
module-specific instructions grounded in the program and accepted traces. Every
trial jointly selects one instruction and one demonstration set **for each
module**. The local categorical tree-structured Parzen estimator partitions
observed configurations by score, models good and bad densities as mixtures over
complete configurations, and maximizes their density ratio. Complete-configuration
mixtures retain dependencies between stages and between instruction/demo choices.
Startup trials sample uniformly.

The baseline is fully evaluated. Minibatch trials drive search, with periodic
full evaluations of the strongest average configuration not yet fully measured.
Full measurements also enter the surrogate history. The returned `best` is always
chosen from full evaluations; noisy minibatch scores cannot directly replace it.
`max_bootstrapped_demos=0` supports zero-shot choices. Existing module demos remain
the baseline option, so pass empty baseline tuples for a fully zero-shot search.

```python
from pathlib import Path
from slick import prompts
from mipro import MIPRO, ModulePrompt, Trace

prompts.TEMPLATE_ROOT = Path("mipro/prompts").resolve()
agent = MIPRO(task, provider, evaluate, execute)
result = await agent.run(modules, training_examples, validation_examples)
```

Intentional adaptations: the local categorical Parzen implementation uses fixed
smoothing, a uniform prior and a configurable good quantile instead of Optuna's
adaptive multivariate TPE implementation. Optuna is not installed or required.
Accepted teacher traces are pooled once and resampled rather than rerunning the
teacher separately for every demonstration set. There is no automatic labeled-
demo derivation, DSPy program introspection, auto-budget presets or randomized
proposal tips. Program/module descriptions are explicit caller data. The generic
templates preserve grounded proposal stages but do not reproduce DSPy wording.
Full-evaluation intervals count actual search iterations rather than DSPy's
adjusted display-trial numbering. These differences are observable and are not
presented as exact numerical reproduction.

Teacher, evaluator, blank generation and nonfinite-score errors propagate.
Counters distinguish teacher runs, optimizer calls, program evaluations and
evaluation-example counts. Configure the process-global template root once.
No paid calls, model construction, benchmark execution or published-score claim.

Check: `rtk proxy optimizer/.venv/bin/python -m unittest tests.test_mipro`.
