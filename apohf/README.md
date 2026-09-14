# APOHF

Slick port of automatic prompt optimization from human feedback using the
official neural dueling-bandit acquisition and preference-model training rules.

Official sources inspected:

- [Repository and paper](https://github.com/xqlin98/APOHF)
- [Candidate generation and comparison loop](https://github.com/xqlin98/APOHF/blob/main/Induction/experiments/run_dbandits_po.py)
- [NeuralDBDiag.select/train/find_best](https://github.com/xqlin98/APOHF/blob/main/Induction/experiments/LlamaForMLPRegression.py)

`APOHF(task, provider, evaluate, embed, surrogate)` accepts an async pairwise
evaluator: `evaluate(left_instruction, right_instruction)` returns preference
for the left candidate (`1` left, `0` right, `0.5` tie). Values must be finite in
`[0,1]`; fractional preference labels are a documented extension. This can be
actual human feedback or caller-owned task evidence. Unlike the benchmark script,
the port does not inspect test-set scores to synthesize human feedback.

`await embed(instructions)` supplies fixed instruction vectors.
`surrogate(theta, vectors)` supplies `(n,)` utility predictions and `(n,p)`
parameter Jacobians for a reward model. The official default is a ReLU neural
model with width 32 and two hidden layers. This callback computes forward values
and derivatives only; all fitting and acquisition logic stays inside the agent.

`run(initial_prompts, initial_theta, ...)` optionally generates additional
instructions by induction or paraphrase, then deduplicates the fixed pool.
Random initial distinct pairs train the reward model. Later acquisitions select
the highest predicted utility as the first arm and the strongest utility-plus-
uncertainty challenger as the second. Uncertainty uses gradient differences
relative to the first arm and a diagonal precision rebuilt from all previous
pair differences under the **current** model. This follows the official neural
implementation, which uses a diagonal calculation even with its diagonalization
argument disabled.

The agent resets parameters and fits all accumulated preferences using
Bradley-Terry binary cross-entropy, local Adam updates, and source weight decay
`lambda / (pairs + 50)`. The returned best instruction has greatest learned
utility among queried arms. Utility is a surrogate value, not measured accuracy.

```python
from pathlib import Path
from slick import prompts
from apohf import APOHF

prompts.TEMPLATE_ROOT = Path("apohf/prompts").resolve()
agent = APOHF(task, provider, evaluate, embed, surrogate)
result = await agent.run(initial_instructions, initial_theta)
```

Intentional adaptations: pair feedback replaces the source's calibrated-score
Bernoulli simulation; embedding and reward-model execution are injected; NumPy
owns training. Fixed generation attempts replace an unbounded retry-until-unique
loop, so duplicates can reduce pool size. Ranking occurs after the final fit;
the source reports its last recommendation before adding that round's label.
The unused evolutionary-pool helper, alternative DoubleTS baseline, benchmark
normalization queries and held-out evaluation are omitted. This is the neural
dueling-UCB method, not a generic optimizer alias.

Blank generation, invalid measured preferences and nonfinite model outputs
raise errors. Provider/evaluator failures propagate without retries. Counters
record comparison attempts, generation attempts and refits. Callers own execution
and feedback collection; the agent never sends messages to obtain feedback.
Configure the global template root once. No paid calls or reproduced-score claims.

Check: `rtk proxy optimizer/.venv/bin/python -m unittest tests.test_apohf`.
