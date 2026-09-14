# CAPO

Implements **CAPO: Cost-Aware Prompt Optimization**, [paper](https://arxiv.org/abs/2504.16005),
grounded in official [finitearth/capo](https://github.com/finitearth/capo), particularly
`capo/capo.py`, `capo/statistical_tests.py`, and instruction/example handling.

CAPO evolves both instructions and attached demonstrations. It crosses instruction
parents and samples half their combined demonstrations, mutates the instruction,
then adds/removes/shuffles examples. Demonstration reasoning is included only when
the task-model prediction matches the known answer. Selection is a statistical
race: evaluate survivors on another shared block, subtract relative prompt-length
cost, perform all paired one-sided t comparisons, and drop a candidate only when
at least the desired survivor count significantly dominates it. Eliminated
candidates receive no later block calls.

```python
from pathlib import Path
from slick import prompts
from capo import CAPO

prompts.TEMPLATE_ROOT = Path("/absolute/path/to/capo/prompts")
result = await CAPO(task, provider, evaluate, demonstrate, token_count).run(
    initial_prompts, block_ids, demonstrations, upper_shots=3,
)
```

`evaluate(full_prompt, block_id)` asynchronously returns a vector of per-example
higher-is-better scores. Shared block IDs must refer to identical ordered examples
for every candidate. Scores are cached by full prompt/block across races.
`demonstrate(instruction, example)` asynchronously returns `(predicted_answer,
response_with_reasoning)`. Demonstration dictionaries contain `input` and `target`.
The tokenizer callback counts tokens in the rendered instruction plus examples;
the default whitespace count is the source fallback. Cost normalization uses the
longest initial rendered prompt, held fixed during search.

Two explicit source fixes: the release's independent `if` statements accidentally
overwrite its add-demonstration branch; this port implements the intended
mutually-exclusive add/remove/shuffle choices. `max_blocks` is an actual count,
not the release's zero-indexed comparison that can evaluate one extra block.
One source behavior is deliberately retained: the final survivor ordering uses
cached **raw mean reward**, whereas statistical elimination uses length-adjusted
reward. Thus the returned ordering is not claimed to maximize the penalized score.
Zero iterations returns unevaluated initial candidates with `score=None`.

The t statistic and exact zero-variance limits are computed locally with SciPy;
one observation cannot establish significance. Templates are task-neutral rewrites
with the source's `<prompt>...</prompt>` contract. Results expose race comparisons,
survivors, cached scores, block/example/model-generation counts and demonstration
calls. Errors propagate without retries. Dependencies: Slick, NumPy, SciPy.
Tests check cost-adjusted early elimination, avoided later blocks, and the
correct-answer condition on reasoning demonstrations.
