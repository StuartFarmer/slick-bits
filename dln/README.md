# DLN

Slick port of DLN-2: a two-layer stochastic language network whose learnable
parameters are its two text instructions. Training uses sampled latent text,
posterior reweighting and separate variational updates for both prompts.

Official sources inspected:

- [Repository and paper](https://github.com/microsoft/deep-language-networks)
- [VILModel.sample_hidden_states/inference_vi](https://github.com/microsoft/deep-language-networks/blob/main/dln/vi/model.py)
- [Prompt and posterior samplers](https://github.com/microsoft/deep-language-networks/blob/main/dln/vi/sampler.py)
- [Forward and residual language layers](https://github.com/microsoft/deep-language-networks/blob/main/dln/vi/layers.py)

`DLN(task, provider, evaluate, log_probability)` requires an async task-loss
evaluator `(prediction, target) -> finite nonnegative loss` and an async model
likelihood primitive `(rendered_context, target) -> finite log likelihood`.
Likelihood must come from the same language model and forward formatting used
by the layers; self-reported confidence is not a replacement for log probability.
The agent renders the exact local forward template before calling the likelihood
primitive, so scoring and generation contexts agree.

`run(hidden_instruction, output_instruction, examples, ...)` executes the hidden
layer, then feeds both its intermediate text and the original input into the
output layer. If every task loss is zero, it stops without prompt optimization.
Otherwise it samples revised intermediate texts conditioned on input and target.
Posterior weights use stable softmax of
`(log p(target | input, latent, output_prompt) + log p(latent | input, hidden_prompt))
/ temperature`; prior inclusion is configurable. This is the released default
posterior-sharpening variant, without proposal-density importance correction.

Output-prompt candidates include the incumbent and generated revisions. They are
ranked by posterior-weighted expected target log likelihood. Hidden-prompt
candidates then use the highest-posterior latents as proposal targets, while
ranking marginalizes over all posterior samples. A configurable penalty subtracts
the log likelihood of original incorrect latent texts, matching the source's
wrong-thought penalty. The output update precedes the hidden update; the sampled
posterior stays fixed throughout both updates. Ties retain incumbents.

```python
from pathlib import Path
from slick import prompts
from dln import DLN, Example

prompts.TEMPLATE_ROOT = Path("dln/prompts").resolve()
agent = DLN(task, provider, task_loss, log_probability)
result = await agent.run(first_prompt, second_prompt, training_examples)
```

The implementation supports error-only latent rewriting and posterior argmax
instead of marginalization. `prompt_samples` includes the incumbent, matching
the source sampler. Counts distinguish forward layer calls, task-loss evaluations,
posterior/prompt generations and likelihood queries. The final prompts have
variational training objectives, not an automatically measured held-out score.

Intentional adaptations: generic local templates replace dataset-specific
classification templates, while retaining an explicit residual input connection.
Every iteration uses the caller's supplied example batch. There is one coordinate
update per layer per iteration. No DLN-1 mode, prompt memory, KL trust penalty,
NCE objective, mutual-information sharpening, proposal-density correction,
class-constrained decoding or accuracy-based likelihood replacement is included.
Likelihood is scored even for a single posterior sample; its normalized weight
is still one. There are no silent context-shrinking retries or hidden fallback
behavior. Blank posterior/proposal text, invalid losses, nonfinite likelihoods
and provider errors propagate.

Configure Slick's process-global template root once. The injected provider and
evaluator own model execution; no model construction, paid calls or reproduced
benchmark scores are claimed.

Check: `rtk proxy optimizer/.venv/bin/python -m unittest tests.test_dln`.
