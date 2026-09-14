# APE

Task-agnostic Slick port of Automatic Prompt Engineer's instruction induction,
deduplication and UCB evaluation allocation. Higher finite scores are better.

Official sources inspected:

- [Paper](https://arxiv.org/abs/2211.01910)
- [find_prompts](https://github.com/keirp/automatic_prompt_engineer/blob/main/automatic_prompt_engineer/ape.py)
- [UCB evaluator](https://github.com/keirp/automatic_prompt_engineer/blob/main/automatic_prompt_engineer/evaluation/bandits.py)

`APE(task, provider, evaluate).run(demonstrations, ...)` samples demonstration
subsets, generates `prompts_per_sample` instructions per subset, deduplicates,
then allocates evaluator calls over `rounds`. The first UCB batch is random;
subsequent batches use mean reward plus `c * sqrt(log(sum(counts + .001)) /
(count + .001))`, following the official implementation. `evaluate(prompt)`
must return a mean over a fresh batch of `samples_per_eval` equally weighted
examples; the caller owns execution and the evaluation dataset. Results include
observed means, sample counts, selected batches and call counts.

Configure Slick's process-global template root once before running:

```python
from pathlib import Path
from slick import prompts
from ape import APE

prompts.TEMPLATE_ROOT = Path("ape/prompts").resolve()
agent = APE(task, provider, evaluate)
result = await agent.run(demonstrations, demos_per_sample=3)
```

Intentional adaptations: generic demonstration strings and a task description
replace dataset-specific templates; sequential calls replace server-side sample
batches. Deduplication and ties are stable instead of unordered set/NumPy ties.
Unobserved candidates report `score=None` and cannot win; a zero-round run has
`best=None`. The official `find_prompts` implementation does not perform iterative
paraphrase refinement; this port follows that released search, not an invented
refinement loop. No insert-mode decoding or likelihood scorer is bundled.

Generation, blank-output, nonfinite-score and evaluator failures propagate without
retry. Counters increment before calls and remain inspectable on an interrupted
agent. No model is constructed or paid evaluation run. Use a fresh agent per
concurrent search; do not change the global template root during a run.

Check: `rtk proxy optimizer/.venv/bin/python -m unittest tests.test_ape`.
Deterministic checks exercise mechanics, not paper performance reproduction.
