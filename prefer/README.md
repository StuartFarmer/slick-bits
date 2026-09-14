# PREFER

`PREFER` learns a weighted prompt ensemble. For each prompt and example it subtracts
backward elimination confidence from forward support confidence, then chooses the
highest-scoring label. These bilateral predictions determine weighted training error.
SAMME assigns prompt weights and increases the weights of mistakes. Reflection on
those mistakes guides the next instruction. `predict` uses the same bilateral
predictions inside the learned weighted vote.

Sources: [AAAI paper, equations 3–6](https://arxiv.org/html/2308.12033v1),
[official repository](https://github.com/zcrwind/PREFER),
[boosting source inspected](https://github.com/zcrwind/PREFER/blob/main/prefer.py),
[paired-input source inspected](https://github.com/zcrwind/PREFER/blob/main/prefer_pair.py).
The inspected release scripts use direct parsed label predictions in their solver
paths; this port implements the paper's explicit bilateral score subtraction.

Adaptations: bilateral scoring is always performed rather than confidence-gated.
One reflected revision is generated per round rather than sampling multiple
instructions and choosing the first. Chance-level or worse learners are skipped
and refined, using the SAMME threshold `1 - 1/K`. A perfect learner terminates and
becomes the sole weight-1 learner, representing the limiting dominating vote without
infinite weights. Prompts use new task-generic JSON contracts. These differ from
release defaults and are not benchmark reproduction claims.

```python
from pathlib import Path
from slick import prompts
from prefer import PREFER, Example

prompts.TEMPLATE_ROOT = Path("prefer/prompts").resolve()
agent = PREFER(task, provider, confidence, labels=["yes", "no"])
result = await agent.run(initial_instruction, training_examples)
label = await agent.predict(result["ensemble"], new_input)
```

The async `confidence(instruction, input, direction)` callback returns one finite
score per label. `direction="forward"` measures support for each answer;
`"backward"` measures confidence in excluding it. Scores must be comparable across
directions. The callback owns model inference, fixed demonstrations/output formatting,
and confidence elicitation. `Example(input, label)` is training data. Search edits
instructions only. Returned ensemble, reflection trace, errors, and instance weights
remain available even when no informative learner is found; `predict` then raises.
Ties use caller label order. Malformed generations and all model/measurement failures
propagate without retries or fallback labels. Caller configuration is trusted.
Set Slick's process-global template root before use.

Validation: `python -m unittest tests.test_prefer` checks bilateral reversal, SAMME
weights, rejected learners, perfect learners, weighted inference, and failures.
