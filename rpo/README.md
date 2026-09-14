# RPO

Shared-token robust suffix optimization using the existing GCG implementation.
`RPO(task, contexts, gradient, evaluate, accept=...).run(token_ids)` averages loss
over training contexts and sums their row-normalized token gradients. Async model
callbacks take `(tokens, context)`. Each context may identify a model, task input,
perturbation and desired response. Define a defensive or other desired behavior
through that objective; the optimizer contains no task-specific targets.

Sources: [paper](https://arxiv.org/abs/2401.17263), official
[gradient aggregation and candidate selection](https://github.com/lapisrocks/rpo/blob/d52ed85797784cbd1877a96263543bfc750c5d2a/rpo/gcg.py).
The source sums normalized gradients across workers and evaluates aggregate target
and control loss. Those objective calculations and model execution remain with
the caller. All supplied contexts must share token IDs/vocabulary. The source's
heterogeneous-tokenizer candidate groups, outer training curriculum and benchmark
attack infrastructure are not implemented. This is the common-vocabulary search
core, not a complete reproduction of the robustness experiments.

Loss is finite and lower-is-better. Zero-gradient rows remain zero. Errors
propagate without retries. Counters distinguish model/context work from aggregate
GCG evaluations. No text generation operation means no Slick templates are needed.
