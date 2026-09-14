# Strategy finding implementation

Implement the supplied paper's three stages as a task-neutral Slick algorithm.
Use the official release at commit ebcb5ed44a71664316c99b021026358a44aef38d
as a source reference; document where its code differs from the paper.

- [x] Add scripted-provider checks for generation, incremental factory updates,
  confidence/risk selection, rejection accounting, and held-out data separation.
- [x] Implement one owning agent with local filter, categorize, generate,
  confidence and risk templates. Inject evaluation and numerical data preparation.
- [x] Implement the input → 10 ReLU units → scalar regression model in NumPy,
  training-only scaling, validation checkpoint selection, and exact local weights.
- [x] Document the public contract, source mapping, runnable usage and adaptations.
- [x] Run behavioral checks, template rendering from another working directory,
  formatting/lint, and an independent code review.

Generated candidate content remains opaque. Evaluation infrastructure, modality
extraction, domain operators, temporal alignment, and deployment belong to callers.
Selection follows Appendix A.6: strict threshold, one winner per category, weights
0.6 confidence / 0.4 risk. Empty selection returns no model. Provider, parsing and
unexpected evaluator errors propagate; explicitly rejected candidates are recorded.
No retries or shared conversational sessions. The factory can be supplied again
for incremental updates; every run re-evaluates it under the supplied context.

The nonlinear MLP has no single global vector of factor weights. Return predictions
and exact input-dependent affine coefficients, including intercept, instead of
silently interpreting hidden weights as factor weights. Paper backtests are outside
this implementation's validation claim.
