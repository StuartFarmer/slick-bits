# PROPANE

Implements the inverse-prompt maximum-likelihood/GCG method from **PROPANE: Prompt
design as an inverse problem**, [original v1](https://arxiv.org/abs/2311.07064v1),
§§3–5 and Appendix Algorithm 2. The linked official repository
[`rimon15/propane`](https://github.com/rimon15/propane) now redirects to
[`rimon15/evil_twins`](https://github.com/rimon15/evil_twins), and the paper's later
versions are titled *Prompts have evil twins*. Inspected current
`evil_twins/prompt_optim.py`: `compute_grads`, `replace_tok`, `optim_gcg` and the
document-likelihood objective, alongside the original paper to confirm identity.

Given desired output documents, the port minimizes average document token negative
log likelihood under a fixed model. It differentiates that objective with respect
to one-hot prompt tokens, ranks substitutions per position, samples one top-k
substitution for each position, and evaluates these complete candidate prompts.
The lowest-loss proposal becomes the next iterate even if it deteriorates; the
best observed prompt is retained. Optional LLM induction provides the paper's
warm start; supplying initial tokens selects its arbitrary/cold-start path.

`PROPANE(task, provider, forward, backward, encode, decode, vocabulary_size)` has
two model-primitive callbacks, both async:

- `forward(one_hot_prompt, documents)` returns document-predicting logits shaped
  `[documents, document_tokens, vocabulary]` with the task/model wrapper fixed.
- `backward(one_hot_prompt, documents, logit_cotangent)` returns the model VJP
  with respect to the prompt, shaped `[prompt_tokens, vocabulary]`.

Cross entropy, its logit derivative, top-k ranking, vocabulary restriction,
sampling, full proposal evaluation and search selection are implemented locally.
Call `run(documents, initial_tokens, top_k=..., allowed_tokens=...)`. `documents`
is a rectangular integer token array; padding/masking and variable-length batching
must be normalized by the caller. Optional `teacher_log_probs` contains the
source prompt's log probability of each observed document token and enables the
empirical forward-KL report. It does not affect optimization. KL is an empirical
estimate over supplied documents, not a full distribution comparison.

This port covers the paper's hard-prompt objective with `gamma=0`; the optional
fluency-regularized and soft-prompt experiments are not included. It scores whole
document batches consistently rather than reproducing source batch-layout quirks.
Best retention uses document likelihood at every step, equivalent to ranking
empirical KL against a fixed teacher on those same documents; separate held-out
KL checkpoint selection remains caller-owned. Gradient row normalization is
omitted because it does not change per-row substitution ranks.

Warm starts use a task-neutral local `prompts/warm_start.j2`; configure
`slick.prompts.TEMPLATE_ROOT` to this directory once before use. Other operations
are numerical and need no artificial LLM calls. With `T` iterations and `n` prompt
tokens the forward count is `1 + T * (1 + n)` and the backward count is `T`.
Errors and nonfinite model outputs propagate without retries. Dependencies:
NumPy, SciPy, Slick; caller-owned model/autodiff and tokenizer. Tests verify the
local loss derivative against finite differences, one-coordinate proposals,
warm-start generation and true likelihood improvement on an analytic model.
