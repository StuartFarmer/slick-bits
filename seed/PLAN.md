# SEED implementation plan

Implement the supplied paper as a task-agnostic Slick algorithm. Use serialized
text inputs and JSON outputs; inject evaluation, embeddings, isolated code
execution, and model fitting. Keep all prompts local and operation-specific.

- [x] Implement cached validation, generic skyline search, specialized priority
  ordering, effectiveness-gap selection, and an exhaustive reference mode.
- [x] Implement code advice/generation/repair, coverage filtering, and ensembles.
- [x] Connect batched/tool-assisted LLM fallback, semantic cache reuse, model
  confidence gating, and periodic retraining/reoptimization without label leakage.
- [x] Verify numerical decisions, actual Slick parsing, failure accounting,
  held-out boundaries, and dynamic scheduling with the shared scripted provider.
- [x] Document the public API, source provenance, adaptations, and measured limits.

The supplied paper is the specification. Its anonymous file endpoint identifies
https://github.com/Magolor/SEED as upstream. Compare against commit
3686e9913ef0719ec82156b6bdaf34f28b2506b8 and document deliberate adaptations.
Do not claim benchmark reproduction.
