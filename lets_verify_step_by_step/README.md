# Let's Verify Step by Step

Problem-agnostic Slick implementation of the algorithm in
[Lightman et al. (2023)](https://arxiv.org/abs/2305.20050): fixed-generator
sampling, process reward scoring, best-of-N search, first-error supervision,
and active selection for reward-model training.

The supplied [APET link](https://github.com/daankepel/APET) belongs to a different
paper, already implemented in `../apet`. This implementation uses the paper's
actual [official PRM800K release](https://github.com/openai/prm800k).

## Use with any task

Use the existing `optimizer/.venv` environment or install the adjacent Slick
checkout. No additional dependencies are needed. Configure templates once at
application startup, before generation:

```python
from pathlib import Path
from slick import prompts
import lets_verify_step_by_step as verify

prompts.TEMPLATE_ROOT = Path(verify.__file__).resolve().parent / "prompts"


async def solve(task, problem, generator, score_steps, evaluate=None):
    agent = verify.VerifyStepByStep(task, generator, score_steps, evaluate)
    result = await agent.run(problem, n=32)
    return result.best.solution.text
```

- `task`: reusable instructions, rubric, and output requirements.
- `problem`: arbitrary source material or a concrete task instance.
- `generator`: a stateless Slick provider, configured by the caller. Use
  stochastic sampling (the paper uses temperature 1) for independent candidates.
- `score_steps(problem, steps)`: async reward-model callback returning one
  `StepProbabilities(positive, neutral, negative)` for every step, in order.
- `evaluate(problem, solution)`: optional async final-answer correctness check.
  It receives a `Solution` containing the original `text` and parsed `steps`.
  At inference it runs only on the selected solution, after selection.

The scorer must use **only the problem and the prefix ending at each step**.
A causal trained PRM can score all boundaries in one forward pass. Alternatively,
a backend can score individual prefixes. Do not use later steps, reference
answers, or outcome grades as inputs to earlier predictions. Return the model's
label probabilities; an LLM's self-reported numerical confidence is not the
paper's trained PRM. There is deliberately no prompted-judge fallback.

The prompt requests newline-delimited steps, including the final output. Blank
lines do not create steps; original output text is preserved. A task may require
multiline code or other artifacts: each nonblank line becomes a scored step.
The caller owns artifact extraction and any execution isolation. Generation
instructions do not establish a sandbox or prove a solution is complete.

## Scoring and supervision

By default a step's correctness probability is `positive + neutral`. A solution
score is the product over **all** steps, including its final output. Ranking uses
the sum of log probabilities to avoid underflow. `score` exposes the product
and can underflow to zero; `log_score` preserves the ranking. Ties retain the
first sample, and exact zero probabilities produce negative infinity.

`neutral_is_correct=False` and `reduction="minimum"` expose Appendix F.2's
alternative scoring rules. Product scoring retains the paper's length bias.
`Result` contains `best`, `samples` in generation order, `calls`, and optional
`correct`; there is no generator revision, RL, or majority-vote substitution.

`process_examples(problem, steps, labels)` produces `TrainingExample` records,
each containing the prefix through a labeled step and its target in `{-1, 0, 1}`.
It includes the first negative and discards everything after it. Short labeling
sequences must end in a negative; otherwise every step needs a label.

`synthetic_labels(probabilities)` implements Appendix H: a step is negative
only when its negative probability is **strictly greater than 0.2**, and labeling
stops there. `synthetic_outcome(probabilities)` uses that same rule to label the
whole solution. Both accept a `threshold=` override.

## Active learning and fitting

```python
async def train(task, generator, seeded_selector, evaluate, label_steps, fit_reward, problems):
    agent = verify.VerifyStepByStep(task, generator, seeded_selector, evaluate)
    training = await agent.learn(
        problems, label_steps, fit_reward,
        pool_size=1000, k=10, rounds=1,
    )
    return agent.reward, training
```

`learn` requires an evaluator and a seeded selector. For each problem it scores
the pool and grades final answers, then selects `floor(0.8 * k)` highest-scoring
wrong-answer samples and fills the remainder with the best remaining samples
of either outcome. Insufficient wrong answers also draw from the remainder.
Selection never repeats a sample position; independently generated duplicate
texts remain distinct samples. The pool must contain at least `k` samples to
collect exactly `k`. `wrong_fraction=1` gives per-problem wrong-first selection.

`label_steps(problem, steps)` is async and returns positive/neutral/negative
labels through the first error or the complete solution. It can use humans or
a separate teacher. A synthetic teacher adapter can return
`synthetic_labels(await teacher(problem, steps))`. Keep that teacher separate
from the evolving selector.

`fit_reward(examples, epochs)` is async and returns a trained `score_steps`
callback. It receives accumulated training prefixes and defaults to **2 epochs**.
The backend must minimize negative log likelihood of the target label token
after the last step in each prefix, masking all problem/solution tokens from
the loss. Prefixes must not contain future steps or reference answers. Use
three label tokens/classes for human labels and retain neutrals as a distinct
training target. The backend owns tokenization, label-token mapping, checkpoint
loading, optimizer, learning rate, batching, and persistence; it must leave the
generator fixed. Task instructions can be captured in the backend's closure.

The default single round follows Section 4.2's fixed-selector collection.
Additional rounds explicitly opt into the iterative collection strategy of
Section 2.4: fit after all problems in a round, replace only the reward callback,
then collect another pool with it. Each `learn` call starts a new training set.
This does not claim iterative retraining improves performance; the small-scale
paper experiments found it unstable. Global-across-problems selection and
ORM token-level training are not implemented.

## Using the official release

Sources checked on 2026-09-14:

- [Official data schema](https://github.com/openai/prm800k/blob/main/README.md):
  `data.phase2_examples` consumes the original completion at each phase-2 step,
  stops at its first error, ignores proposed repairs, and excludes QC, screening,
  incomplete work and flagged/null ratings. Phase 1 is explicitly rejected
  because its branching/human continuations need a different adapter.
- [Official evaluation code](https://github.com/openai/prm800k/blob/main/prm800k/eval/eval.py):
  `data.scored_sample_trial` adapts its sample-padding, subsampling, answer
  filtering, and highest-score selection, using the standard library and an
  injected RNG. It accepts the released `prm_score` or `orm_score` fields.
  OpenAI's MIT notice is retained in `UPSTREAM_LICENSE`.

```python
from lets_verify_step_by_step.data import phase2_examples, read_jsonl

training = tuple(
    example
    for row in read_jsonl("phase2_train.jsonl")
    for example in phase2_examples(row)
)
# trained_reward = await fit_reward(training, 2)
```

Files must be downloaded separately, including Git LFS objects when cloning.
`read_jsonl` also supports `.gz`. Use only training files when fitting; keep the
official 500-problem evaluation split held out. The original MATH train/test
split differs from the paper's split. The official math grader can be wrapped
in `evaluate` for a math application; it is not imported into this generic agent.

The official repository releases data, grading, and evaluation code, not the
trained GPT-4 PRM checkpoint or its training implementation. This package
implements orchestration and training-data preparation; **actual gradient
training requires the caller's `fit_reward` backend**. It does not reproduce
MathMix pretraining, model-scale experiments, or the paper's reported accuracy.
The generic generator prompt is an adaptation, not the paper's fine-tuned
generator. Offline checks do not establish reward-model calibration or quality.

## Failures and checks

Generation and reward failures stop immediately, without repair or retries.
`agent.calls` retains prompts, raw responses received before parsing, and their
errors; completed scored samples remain in `agent.samples`. Labeler, trainer,
and evaluator errors propagate, retaining prior state. A failed fit leaves the
previous scorer intact. Training/model calls inside callbacks are not counted
as generator calls. Caller settings are trusted; measured probabilities are
checked for count, range, finiteness, and normalization.

Use one operation at a time per instance. Calls are sequential and do not share
a Session. Configure Slick's process-global template root at startup; concurrent
algorithms with different roots require separate processes.

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_lets_verify_step_by_step
../slick/.venv/bin/ruff check lets_verify_step_by_step tests/test_lets_verify_step_by_step.py
../slick/.venv/bin/ruff format --check lets_verify_step_by_step tests/test_lets_verify_step_by_step.py
```

Tests use the shared scripted provider through real Slick prompts, and cover
selection, underflow, neutral/minimum alternatives, first-error labels, active
quotas, retraining, official adapters, failures, and both launch directories.
