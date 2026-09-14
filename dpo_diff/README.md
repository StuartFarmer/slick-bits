# DPO-Diff

Gradient-based discrete prompt optimization over a compact per-fragment search
space. `DPODiff(task, gradient, evaluate).run(slots, embeddings)` accepts alternative
strings per slot, embedding arrays whose first axis indexes those alternatives,
and two asynchronous model callbacks. `gradient(mixed_embeddings, timestep)` must
return derivatives of the objective with respect to **each mixed embedding**;
the caller implements the diffusion shortcut gradient or an analogous model
gradient. `evaluate(tuple_of_selected_fragments)` measures discrete loss.

Sources inspected: [paper](https://arxiv.org/abs/2407.01606), official
[DPO-Diff](https://github.com/ruocwang/dpo-diffusion/tree/52a6188184a05aaa874782e938845c4292070c55)
`src/optimizers/gradient/gpo.py`, `common/prompt_optimizer.py`, `common/rmsprop.py`.
Preserved: Gumbel-softmax mixture, explicit softmax derivative, gradient clipping
to ±0.025, RMSprop alpha .99 / momentum .5 / epsilon 1e-8 / learning rate .1,
logit clipping to [0, 3], and sampling unique discrete candidates after training.

The compact vocabulary, tokenizer alignment, positive/negative prompt assembly,
image generation, and shortcut backpropagation stay with the caller. This ports
the gradient optimizer, not the alternate evolutionary optimizer or the diffusion
model itself. A fixed timestep is exposed; upstream also supplies random/stepped
schedules. NumPy replaces Torch for optimizer arithmetic. A sample-attempt budget
replaces upstream's unbounded unique-sampling loop. It may return fewer candidates
than requested. No template is needed: model differentiation supplies the update.
All measured losses/gradients must be finite; exceptions propagate without retries.
Tests validate optimization arithmetic and finite-domain termination only.
