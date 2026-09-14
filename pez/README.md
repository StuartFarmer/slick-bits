# PEZ — Hard Prompts Made Easy

`PEZ(task, vocabulary_embeddings, gradient, evaluate).run(initial_embeddings)`
keeps continuous latent embeddings, projects to nearest vocabulary tokens by
cosine similarity, computes gradients **at the discrete embeddings**, and applies
those gradients to the latent embeddings using AdamW. The latent values are never
overwritten by their projection. `evaluate(tuple_of_token_ids)` returns a finite
loss; lower wins. `gradient(projected_embedding_array)` is also asynchronous.

Sources inspected: [paper](https://arxiv.org/abs/2302.03668), official
[optim_utils.py](https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/f22a1bec01991d94697304443cacbd66e0167e6b/optim_utils.py),
`nn_project` and `optimize_prompt_loop`. The CLIP model, tokenizer, target features,
fixed prompt padding, and minibatch selection belong to the caller. NumPy replaces
Torch for projection and AdamW arithmetic. This implementation supports one prompt
sequence per run and also evaluates the final update, unlike the upstream loop's
pre-update scoring. No Slick prompt calls are needed for this numerical method.

Errors propagate without retries. Best-so-far, per-step measurements and model-call
counts are returned. The test checks latent/projection separation and updates;
it does not reproduce image/text benchmark results.
