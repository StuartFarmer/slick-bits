# Adversarial in-context learning

`AdvICL` alternates two prompt players on a sampled training batch. It first maximizes
the discriminator's GAN objective, then minimizes it for the generator. For each
player it searches the instruction and every demonstration in sequence. Each
coordinate compares all proposed replacements against the current prompt and
accepts only a strict improvement. Results retain both prompts and a coordinate trace.

Sources: [ACL paper, algorithm 1](https://arxiv.org/html/2312.02614v3),
[official repository](https://github.com/zhaoyiran924/Adv-In-Context-Learning),
[MMLU implementation inspected](https://github.com/zhaoyiran924/Adv-In-Context-Learning/blob/main/eval/adversarial/gan_chat_mmlu.py).
The paper optimizes both instructions and demonstrations through opposite directions
of the same discriminator loss. The inspected code instead stops at the first
improving proposal and also updates discriminator wording during generator edits.

Adaptations: this port follows the paper's best-of-r independent coordinates,
with rewritten JSON prompts. Input/output pairs are generic text; task-specific
multiple-choice formatting belongs to the caller. The same sampled data supplies
both real and generated loss terms. Generator outputs are recomputed for each loss
evaluation, as in the source. There are no 600-attempt retries, guessed token
likelihoods, or hidden dataset paths. This is a mechanics port, not a reproduction.

```python
from pathlib import Path
from slick import prompts
from adv_icl import AdvICL, GeneratorPrompt, DiscriminatorPrompt

prompts.TEMPLATE_ROOT = Path("adv_icl/prompts").resolve()
agent = AdvICL(task, modifier_provider, generate, discriminate)
result = await agent.run(generator_prompt, discriminator_prompt, training_pairs)
optimized_generator = result["generator"]
```

`generate(GeneratorPrompt, input)` asynchronously returns generated text.
`discriminate(DiscriminatorPrompt, input, output)` asynchronously returns **log
P(real)** from a model's actual label likelihood, not a numeric self-rating.
The loss is mean `log P(real | real pair) + log(1 - P(real | generated pair))`.
The caller owns these models, token-probability extraction, task formatting, and
held-out testing; the modifier alone uses Slick. `Example` carries input/output;
`DiscriminatorExample` additionally carries `real`/`generated`. Generated examples
are structurally checked; semantic correctness is a model dependency, not guaranteed.

Empty generated text, malformed proposals, invalid likelihoods, nonfinite loss,
and provider/evaluator failures propagate immediately. Exact probabilities that
would yield infinite loss are rejected rather than clipped. Configuration is trusted.
`evaluations` counts complete GAN loss assessments, not individual model calls.
For controlled comparisons use deterministic inference. Configure the process-global
Slick template root before running; do not switch it concurrently.

Validation: `python -m unittest tests.test_adv_icl` checks player direction, every
coordinate, frozen best-of-r proposals, strict acceptance, and failure boundaries.
