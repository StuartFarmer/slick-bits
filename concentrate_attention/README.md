# Concentrate Attention

`ConcentrateAttention(task, observe).run(initial, ...)` implements the soft-prompt
concentration objective and Adam updates. `observe` returns `Observation`: task
loss/gradient, per-input lookback attention over prompt tokens, its Jacobian with
respect to the optimized parameters, and caller-defined labels. Layer/head
selection and model autograd belong to this callback; the concentration loss and
its analytic derivative live here. Lower combined loss wins. The return contains
the best measured parameters, final parameters, objective history and model calls.
`global_score` also exposes hard-prompt filtering's margin/strength/KL criterion.

Based on [paper equations 8–13](https://arxiv.org/html/2406.10584v1) and inspected
[official model.py](https://github.com/czx-li/Concentrate-Attention/blob/main/model.py),
fine-tuning and reward implementations. The release hardcodes RoBERTa token IDs,
layers and heads, and uses a different contrastive denominator and normalization
in several task branches. This port follows the paper's supervised contrastive
equation, summing anchors and excluding self pairs. Singleton labels contribute
zero contrastive loss; numerical zero-norm protection uses 1e-12. Hard filtering
defaults to a negative KL weight because fluctuation is minimized (the paper
writes its coefficient as a signed parameter).

This is the soft optimization core and hard filtering objective, not the separate
MAPPO prompt-routing training pipeline. Attention/Jacobian access is required;
an ordinary text-only API cannot provide it. Dense Jacobians suit small prompts;
a model-specific vector-Jacobian product is the upgrade for large models. No
domain-generalization experiments or paper performance claims were reproduced.
