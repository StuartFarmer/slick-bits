# Black-box discrete prompt learning (BDPL)

`BlackBoxPromptLearning` implements [Diao et al., TMLR 2023](https://arxiv.org/abs/2201.08531). It learns one independent categorical distribution per prompt position, samples complete discrete prompts, estimates a variance-reduced policy gradient from same-batch losses, applies AdamW and projects each row onto the simplex. This is BDPL; the bibliography's separate BBT entry optimizes continuous prompts.

The inspected [official LM training loop](https://github.com/shizhediao/Black-Box-Prompt-Learning/blob/main/run_glue_discrete_LM.py) uses a centered `+1/p` selected / `-1/p` unselected estimator divided by `samples-1`. Its unselected term cancels across centered losses, leaving twice the usual selected-coordinate score estimator. The local algebraic implementation preserves that factor of two while avoiding undefined division by zero at unused zero-probability coordinates. The paper's displayed off-coordinate denominator differs from the code; this port explicitly follows the released estimator. Exact sorting-based simplex projection replaces the release's bounded bisection, which cannot find negative thresholds for all-negative rows.

```python
agent = BlackBoxPromptLearning(task, evaluate)
result = await agent.run(vocabulary, batch_ids, prompt_length=5)
```

Async `evaluate(prompt_tuple, batch_id)` returns a finite **lower-is-better loss**. It owns the black-box model calls, labels and loss computation; only its scalar is exposed to the learner. `vocabulary` can contain tokens or domain phrases; the release's optional [PMI n-gram construction](https://github.com/shizhediao/Black-Box-Prompt-Learning/blob/main/pmi_ngram.py) is caller-side corpus preparation, not silently performed on held-out data. Local training uses exactly `epochs * len(batch_ids) * samples_per_batch` evaluations; use at least two samples for the variance-reduced estimator. The result is the coordinatewise modal prompt and learned distributions, not a validation-selected checkpoint. Validation ensembling and source benchmark-specific verbalizers are outside this variant.

No target gradients, model weights, continuous embeddings or trainable LM are needed. Generated token choices come from the local vocabulary. Nonfinite losses/updates raise; errors propagate without retries. Numeric sampling needs no prose templates; `prompts/` remains available for application presentation. Tests check estimator expectation, projection, convergence direction and zero-mass behavior without paid calls.
