# TUNA

`TUNA` implements the two-stage response-model tuning procedure in [Li et al., EMNLP 2023](https://arxiv.org/abs/2310.13385): rank teacher samples by length-penalized log probability, train the student, then sample the updated student and train again on contextual rankings. Sources inspected include the official [overview/data contracts](https://github.com/microsoft/LMOps/tree/main/tuna), [data preparation](https://github.com/microsoft/LMOps/blob/main/tuna/src/train_tuna.py) and [custom loss](https://github.com/microsoft/LMOps/blob/main/tuna/src/custom.py).

```python
agent = TUNA(task, ranking_provider, teacher, generate, forward)
result = await agent.run(parameters, [Example(instruction, original_response)])
```

Async `teacher(instruction, seed)` returns `TeacherResponse(text, token_logprobs)`. Async `generate(parameters, instruction, temperature, seed)` samples the student. Async `forward(parameters, instruction, response)` returns response-only token log probabilities `[tokens]` and their Jacobian `[tokens, parameters]`. Model architecture, tokenizer and model execution are adapter responsibilities. The owner computes teacher length normalization, rank-gap hinge losses, negative mean reference log likelihood, chain-rule gradients and Adam updates. Parameters are copied across callbacks and returned as a learned model artifact; this method does not produce one globally optimized prompt.

This is the **paper-objective variant**: sum every rank-gap hinge and regularize against the original reference. The released `custom.py` instead averages hinge terms within each rank-distance group and applies its MLE loss over supplied candidate tokens. Also, the sign on the paper's Eq. 6 MLE term is missing a minus; the negative-log-likelihood convention in Eqs. 1–2 is used here. These distinctions are explicit rather than presented as byte-for-byte trainer equivalence.

Contextual sampling enforces pairwise ROUGE-L F1 diversity with bounded retries, increasing temperature by 0.1 and retaining the least similar attempt on exhaustion. Whitespace tokenization replaces the source scoring package. The local `rank.j2` follows the paper's open/closed question assessment, reference answer and three-criterion comparison, adapting its textual ranking into checked JSON. Configure `slick.prompts.TEMPLATE_ROOT` to this folder's `prompts/` once at application startup. Exactly one contextual ranking call is made per example. Sample calls are bounded by `examples * candidates * (1 + resample_attempts)`; no proxy ranking model or initial supervised instruction-tuning stage is supplied.

Malformed/duplicate/incomplete ranking IDs, blank generations and nonfinite likelihoods/derivatives raise; there are no automatic repairs or retries beyond the specified diversity loop. Tests cover both stage order and numerical derivatives, not benchmark quality.
