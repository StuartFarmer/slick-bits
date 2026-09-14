# PROMST

Task-agnostic implementation of **Prompt Optimization in Multi-Step Tasks**.
Supply an initial instruction, a Slick provider, and an async evaluator that returns
an average score and categorized feedback from your human-designed rules.
Task execution, tools, datasets, and any required execution isolation stay in your evaluator.

```python
from pathlib import Path
from statistics import mean

import promst
from slick import prompts
from promst import PROMST, Evaluation, LongformerFit

# Set once at application startup; Slick uses a process-global template root.
prompts.TEMPLATE_ROOT = Path(promst.__file__).resolve().parent / "prompts"

async def optimize(task, initial_prompt, provider, cases, run_trial):
    # run_trial(instruction, case) -> Evaluation for one complete task trial.
    # It owns TaskLLM calls, environment interactions, and human feedback rules.
    async def evaluate(instruction):
        trials = [await run_trial(instruction, case) for case in cases]
        return Evaluation(
            mean(trial.score for trial in trials),
            [feedback for trial in trials for feedback in trial.feedback],
        )

    agent = PROMST(task, provider, evaluate, fit=LongformerFit())
    result = await agent.run(initial_prompt)
    return result["best"].prompt, result
```

The evaluator's feedback items are `Feedback(category, text)`. Choose categories
appropriate to your task, for example `syntax`, `invalid_action`, or `loop`.
Include the offending response, relevant execution context, and the violated rule
in `text`. End a trial on a detected error, completion, or its step limit. Use a
sliding history window in your task runner when necessary. No task-specific error
classifier or environment is hard-coded here. The optimizer's provider handles
SumLLM and GenLLM; your task runner may use a different TaskLLM provider.

Scores must be finite, with higher values better. Evaluate every candidate on the
same nonempty training trials. Keep a final test set outside optimization. Empty
feedback stops expansion of that parent, following the official code. If a task
is incomplete but produced no other error, supply feedback for its unmet objective.

## Search

1. Evaluate and archive the initial prompt (generation zero).
2. For each child, independently sample up to ten feedback instances, group by
   category, summarize each group, and revise using the summaries and ancestry.
3. Starting at generation four, fit the supplied heuristic on measured
   prompt-score pairs. Accept a proposal when
   `mean(predictions) + population_variance(predictions) + mean(errors)` is at
   least `0.8 * best_observed_score` (paper Equation 3).
4. Evaluate accepted unique prompts and retain the global top five from the
   entire archive. Predicted scores never become measured scores.
5. Stop after three consecutive generations without a strict improvement, no
   expandable parents with feedback, or the depth limit.

Defaults: `first_children=20`, `children=8` per parent thereafter, `beam_size=5`,
`feedback_count=10`, `score_start=4`, `threshold_factor=0.8`, `patience=3`, `seed=0`.
`depth=20` is an explicit local cap including generation zero; set `patience=None`
to use only exhaustion/depth stopping. A screened parent gets at most `3*n`
proposal attempts for `n` children; an unscreened parent gets `n`. Duplicates
consume attempts but skip task evaluation. Tied scores retain older candidates.
For a small run, set **both** `first_children` and `children`.

Passing `fit=None` runs the no-score-model ablation with no torch dependency.
An injected async `fit(history)` can instead return `Heuristic(predict, errors)`
or `None` when there is insufficient training data. `predict(prompt)` returns an
ensemble of scores; `errors` contains held-out absolute errors in the same score
units. The archive provided to fitting contains only actual evaluations.

## Longformer ensemble

Install the optional dependencies into your chosen environment:

```sh
python -m pip install -r promst/requirements-score-model.txt
```

`LongformerFit()` fine-tunes five `allenai/longformer-base-4096` regression models,
using independent seeded 4:1 train/validation splits and held-out mean absolute
errors. Each generation starts from the pretrained checkpoint and uses the
accumulated archive; this keeps the new validation split free of prior training
label leakage. Fewer than five measured prompts disables screening for that fit.

The default local training choices are three epochs, batch size two, AdamW with
learning rate `2e-5`, MSE loss, and at most 4096 input tokens. The paper and released
code do not specify these training hyperparameters. They are configurable via
`LongformerFit(checkpoint=..., epochs=..., batch_size=..., learning_rate=...,
max_length=..., device="cpu", seed=...)`. The checkpoint can be a local directory.
Long prompts are truncated by the tokenizer. All five models stay on the selected
device; use appropriate memory and compute for the base model. The first actual
fit downloads pretrained weights unless they are already cached or local.

Training and prediction run in a worker thread. Use one active run per optimizer
and one active fit per trainer; do not run concurrent training that shares global
PyTorch RNG state. CPU execution is tested; accelerator execution is not validated.
`trainer.splits` records train/validation indices into its latest input archive.

## Results and failures

The result dictionary includes `best`, `beam`, `archive`, `generation_best`, and
`stop_reason` (`depth`, `stagnation`, or `exhausted`). Candidates retain prompt,
measured score, feedback, and chronological ancestors. `history` records proposal
attempts, duplicate/heuristic rejections, bounds, and thresholds. `responses`
retains raw optimizer prose before nonblank checks.

Counters distinguish proposal `attempts`, `optimizer_calls`, candidate
`evaluations`, `prediction_calls`, and `fit_calls`. Evaluator calls can contain
many TaskLLM calls; the optimizer does not count those internal calls. Generation,
training, and evaluation errors propagate without retries. On failure, the agent
retains counters and raw responses; an unfinished generation attempt remains
marked `generation_pending`. Provider transport retries belong to the provider.

## Sources and deliberate differences

Based on [the paper](https://aclanthology.org/2024.emnlp-main.226/) Algorithms 1–3,
Equation 3, Appendix C, and Section 4.3, and the
[official implementation](https://github.com/yongchao98/PROMST) at commit
`64a0785e7520cb775980596d315394c05cd92d9c`:

- [BoxLift/prompt_env3.py](https://github.com/yongchao98/PROMST/blob/64a0785e7520cb775980596d315394c05cd92d9c/BoxLift/prompt_env3.py):
  `prompt_to_error_summarizer_PROMST` and
  `prompt_to_promptLLM_func_total_with_env_act_feedback_PROMST_APO` supply the
  wording for the local summary/revision templates. This port adds task context,
  category labels, readable spacing, and JSON ancestry serialization. Outputs
  remain prose. The upstream MIT license is retained in `LICENSE.upstream`.
- [BoxLift/env3_func.py](https://github.com/yongchao98/PROMST/blob/64a0785e7520cb775980596d315394c05cd92d9c/BoxLift/env3_func.py):
  `new_prompt_construct_func_PROMST` supplies categorized summarization and
  no-feedback termination. Its score-model branch is a placeholder. Its tree
  search uses recursive child selection and declining-score stopping. This port
  follows the paper's global beam, ten-instance sampling, and stagnation stopping.

Fitting happens once per generation on its starting archive, following the
paper's prose, rather than repeatedly per parent as Algorithm 2 is written.
The threshold is also fixed at generation start. Finite budgets, duplicate
suppression, and raw-response records make execution inspectable. This is an
algorithm implementation, not a reproduction of the published benchmark scores.
The score model uses Hugging Face's
[Longformer regression implementation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/longformer).

## Checks

From the repository root with Slick installed:

```sh
python -B -m unittest tests.test_promst
```

The tests cover feedback sampling/grouping, ancestry, global selection, generation
schedules, stagnation (including all-duplicate generations), heuristic admission and budgets, duplicate handling, error
propagation, and template rendering from another working directory. With the
optional dependencies installed, the same command trains five tiny local
Longformer models and independently checks held-out errors; it needs no network
or API credentials. Without them, only that training test is skipped.
