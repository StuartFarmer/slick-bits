# EASE

Slick port of the official **joint instruction and ordered exemplar** variant of
Efficient Ordering-aware Automated Selection of Exemplars.

Official sources inspected:

- [Paper](https://arxiv.org/abs/2405.16122)
- [Joint instruction/exemplar search](https://github.com/ZhaoxuanWu/EASE-Prompt-Optimization/blob/main/experiments/run_choose_instruction_ease.py)
- [NeuralTSDiag acquisition and training](https://github.com/ZhaoxuanWu/EASE-Prompt-Optimization/blob/main/experiments/LlamaForMLPRegression.py)

`EASE(task, provider, evaluate, hidden_states, surrogate)` requires an async
evaluator of `(instruction, ordered_example_strings)`, returning a finite
higher-is-better task score. `hidden_states(rendered_strings)` returns model
representations sensitive to the **complete ordered prompt**; independent example
embeddings averaged without order cannot provide EASE's intended capability.
`surrogate(theta, contexts)` returns reward predictions and parameter Jacobians
only. The source uses a 100-hidden-unit ReLU network. Acquisition, confidence
updates, model reset and Adam fitting all remain in this class.

`run(instructions, examples, validation_examples, initial_theta, ...)` samples
initial ordered exemplar tuples paired with instructions and fits a neural
surrogate. Each later round oversamples ordered tuples, computes exact uniform
optimal-transport distance to representative validation examples, and retains
low-cost tuples. It samples instruction choices for each retained tuple,
embeds each complete prompt, and acquires the highest neural-UCB candidate.
Optimal transport is solved locally using SciPy's linear programming solver.
The released script calls this a DPP in comments, but actually uses transport
cost; the port follows the executable source.

One source quirk is deliberately preserved: **each domain batch's local UCB
winner updates diagonal confidence U, although only the best winner across all
batches is evaluated**. Set `domain_batches=1` for one update per acquisition.
It is not claimed that multiple batches implement confidence updates based only
on observed rewards. Training resets the surrogate to its initial parameters,
standardizes observed contexts and fits accumulated rewards with squared error,
Adam and `lambda / observations` weight decay.

```python
from pathlib import Path
from slick import prompts
from ease import EASE

prompts.TEMPLATE_ROOT = Path("ease/prompts").resolve()
agent = EASE(task, provider, evaluate, hidden_states, surrogate)
result = await agent.run(instructions, examples, validation_examples, initial_theta)
```

`Configuration.indices` is an ordered tuple; permutations remain distinct cache
keys and representations. Validation examples are used for optimization-time
transport filtering, not held-out testing. They may include labels if the caller
chooses the source input-label representation, and must be separate from any
untouched test set. The caller owns task execution and dataset separation.

Intentional adaptations: NumPy/SciPy replace GPU neural wrappers and POT; the
hidden-state and derivative model boundaries are injected. Batch sizes and
instruction sampling are configurable instead of source constants. Single-
observation/constant-feature normalization uses unit scale, avoiding source NaNs
or amplification by `1e-30`. The loop does not assume accuracy is capped at one,
stop at a score of one, or freeze training after a fixed iteration. The example-
only, retrieval, evolutionary and subset baseline variants are not implemented.
Generated templates are generic, not byte-for-byte source prompts.

Blank generation, nonfinite model/evaluation output and transport solver failure
propagate. Counts distinguish generations, real evaluations, cached rewards and
refits. Configure the global template root once. No paid calls or benchmark claim.

Check: `rtk proxy optimizer/.venv/bin/python -m unittest tests.test_ease`.
