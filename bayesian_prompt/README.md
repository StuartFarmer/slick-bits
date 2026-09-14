# Bayesian prompt optimization — Sabbatella et al.

`BayesianPrompt(task, ngrams, evaluate, validate).run(length=6)` treats a prompt as
a vector of n-gram indices. It fits a GP posterior, maximizes UCB in the continuous
index box, rounds the proposal to discrete coordinates, and evaluates the prompt.
Training scores guide search; held-out validation selects the return value.

Sources: [paper](https://arxiv.org/abs/2312.00471), author's
[repository](https://github.com/AntonioSabbatellaUni/Black-Box-Prompt-Learning/tree/f824a29b7d59e3e9d0387e894bca0ef05dbe06b2),
especially `bo_model_GPT.py`'s `BoPrompter` initialization/acquisition/train loop.
That repository is linked from the author's profile as the implementation of the
published continuation, *Prompt optimization in large language models*.

Preserved: integer n-gram search, GP posterior, UCB beta 0.4, continuous acquisition
followed by rounding, and separate train/validation measurements. Adaptations:
explicit fixed Matérn-5/2 length scale/noise, standardized scores, NumPy/SciPy
linear algebra and bounded multistart acquisition instead of BoTorch. The kernel
reuses `instructzero.agent.matern52`. No hyperparameter learning is claimed; the
inspected upstream path also constructs an MLL without fitting it. All initial
points receive real validation instead of upstream's zero-filled initial values.
The initial sampler includes the final vocabulary item. Evaluations are uncached.

The caller supplies the vocabulary, training evaluator and independent validation
evaluator. No generated instruction or Slick prompt is involved in this method.
Failures propagate without retries; score direction is higher-is-better. Tests
verify callback isolation and search domain, not benchmark accuracy.
