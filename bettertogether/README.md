# BetterTogether

`BetterTogether(task, prompt_optimize, weight_optimize, evaluate).run(program,
training, strategy=("p", "w", "p"))` alternates prompt and weight optimization.
The next phase uses the previous phase's program, even if its validation score
decreased. It evaluates the baseline and every phase and returns the highest
held-out score, preserving independent program snapshots and earlier ties.

This is a meta-optimizer: supply an existing Slick prompt optimizer and a real
fine-tuner as async `(program, training_examples) -> program` functions. The
evaluator closes over a separate held-out set. Programs can be your own data
objects containing instructions, examples and versioned model/provider handles.
The default snapshot operation is `deepcopy`; pass `clone=` for noncopyable model
clients. A clone must keep old model checkpoints valid; callbacks must not mutate
the weights behind earlier checkpoints. Training examples are treated as immutable.

Based on [the paper](https://arxiv.org/abs/2407.10930) and inspected
[official DSPy BetterTogether](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/bettertogether.py).
Kept sequential composition, training-order shuffling, baseline evaluation and
checkpoint selection. This port requires held-out evaluation and propagates
errors; it does not copy DSPy's automatic splitting, broad exception recovery,
model-server lifecycle or bundled fine-tuning infrastructure. No templates are
needed: generation belongs to the composed optimizers. No training or paper
benchmark results have been reproduced.
