# Reliable gradient-free and likelihood-free prompt tuning

`SBIPrompt(task, evaluate, validate).run(dimension=..., training_size=...)`
implements the paper's **ABC-SMC uniform-weight path**. A Gaussian prior provides
low-dimensional soft-prompt particles; initial probes set the accuracy threshold.
Accepted particles form an ensemble. Subsequent stages resample from the previous
population, perturb with adaptive diagonal Gaussian covariance, and raise the
threshold by `1/training_size`. A separate validation callback selects the best
complete ensemble. This is a distribution over prompts, not a single instruction.

Sources: [paper](https://arxiv.org/abs/2305.00593), authors'
[ABC_SMC source](https://github.com/maohaos2/SBI_LLM/blob/2b06abc4937bd5d0f1b635cb9b586c54d6a6abb4/algorithm.py),
linked from Maohao Shen's publication page. `evaluate(theta)` supplies training
accuracy in [0,1]; `validate(particles, weights)` supplies held-out ensemble quality
(higher wins). The caller owns projection, soft-prefix model conditioning, data
and predictive aggregation. No text-generation template is involved.

Preserved: pilot maximum threshold, Gaussian prior, uniform population weights,
adaptive variance normalized to mean 10, threshold increments, and latest-tie
validation selection. Adaptations: sequential evaluation, bounded attempts even
for the first/partially filled stages, and an epsilon variance floor for collapsed
populations. A partial population never replaces a complete ensemble. Errors
propagate; no hidden retries. The paper's CMA-ELBO, neural SBI and optional
importance-weight alternatives are not included. Tests verify SMC lifecycle and
held-out selection boundaries, not posterior calibration or benchmark accuracy.
