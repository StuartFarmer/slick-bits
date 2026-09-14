# ClaPS

Implements **Survival of the Most Influential Prompts: Efficient Black-Box Prompt
Search via Clustering and Pruning**, [paper](https://arxiv.org/abs/2310.12774).
Inspected official [cambridgeltl/ClaPS](https://github.com/cambridgeltl/ClaPS):
`run_prune_search.py` (`find_kl_dict`, `action_set_pruning`),
`rewards/text_classification_reward.py` (`compute_kl`) and `algs/greedy.py`.
The repository's clustering is supplied by `vocabs/get_kmeans_vocab.ipynb`;
the port follows paper §4's nearest-token-to-centroid definition.

The class clusters token embeddings with k-means++ initialization, retains the
closest token to each center, measures token influence relative to an empty
prompt, and retains only tokens strictly above the requested influence percentile.
Its final search is the authors' **ClaPS-greedy** variant: at each position,
evaluate every retained token appended to the current prefix and select the
highest reward. Genetic and particle-swarm variants are not included.

`ClaPS(task, embeddings, probabilities, evaluate).run(...)` takes a vocabulary-row
embedding array, async `probabilities(token_tuple)` returning distributions over
labels for a fixed reference batch, and async higher-is-better `evaluate(token_tuple)`.
Callbacks own token decoding, task-model inference and optional probability
calibration. The class locally computes summed `KL(prompt_probs || empty_probs)`,
matching the release's direction. `influence="reward"` uses the alternative
incremental reward score. Vocabulary IDs are the embedding-array row indices;
pre-filter special/unwanted tokens and retain the mapping in the caller.

SciPy's k-means++ initialization/Lloyd implementation replaces the release's
scikit-learn clustering; initializer sampling may differ. Request at least the
vocabulary size to skip clustering. Strict percentile pruning can retain no tokens
when all influences tie; this port returns an empty prompt with `score=None`
instead of invoking an empty argmax. Clustering, pruning and greedy search are
not delegated to callbacks. Counts separately expose distribution queries and
task-reward evaluations. Nonfinite measured influence/reward raises. Dependencies:
NumPy and SciPy. No textual model generation/provider/templates are needed.
Tests check representative reduction, influence-based pruning, restricted greedy
search and empty retained-vocabulary behavior; no benchmark results are claimed.
