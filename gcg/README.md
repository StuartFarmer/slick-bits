# GCG

Greedy Coordinate Gradient token search with **lower loss preferred**.
`GCG(gradient, evaluate, accept=...).run(token_ids)` requires async one-hot token
gradients and token-sequence loss from your own model, plus an optional synchronous
candidate acceptance predicate (for example tokenizer round-trip preservation).
Token IDs are not text and this algorithm needs model internals, so there is no
Slick text-generation boundary. NumPy is already used in slick-bits.

Sources: [paper](https://arxiv.org/abs/2307.15043), official
[minimal GCG utilities](https://github.com/llm-attacks/llm-attacks/blob/098262edf85f807224e70ecd87b9d83716bf6b73/llm_attacks/minimal_gcg/opt_utils.py).
Ported mechanics: lowest-gradient top-k vocabulary choices, forbidden token mask,
evenly distributed single-coordinate sampling, and minimum-loss batch selection.
The survey's prose about selecting the *highest* gradient values is misleading:
the authors use `(-grad).topk`, which is preserved here.

Adaptations: model/gradient calculation and tokenizer constraints are injected;
NumPy replaces Torch for sampling; rejected candidates are skipped rather than
padding a batch with its last valid entry. An empty accepted batch retains the
current candidate. The trajectory can worsen while `best` tracks the lowest loss.
All failures propagate, no retries; counters include attempted evaluations and
gradient calls. This is single-objective GCG, not upstream's multi-model curriculum.
Deterministic numerical tests check coordinate decisions, not real model efficacy.
