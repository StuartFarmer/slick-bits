# Direct metrics optimization (DAEMON)

The bibliography's DMO entry is **DAEMON**, a decoding method, from [Ji et al., ICLR 2024](https://arxiv.org/abs/2310.01041). This implements Algorithms 1–2: fit energy coefficients using weighted importance estimates and root mean squared relative metric error, then draw fresh continuations with sampling-importance-resampling. [The paper's implementation details](https://arxiv.org/html/2310.01041v2#A10.SS2) specify Adam at 0.005 and error tolerance 0.001. No official implementation was linked in the inspected paper or [first author's publication list](https://haozheji.github.io/); this is a paper reconstruction, not an upstream code port.

```python
agent = DMO(task, sample, evaluate)
result = await agent.run(reference_prefix_continuation_pairs, new_prefixes)
```

Async `sample(prefix, temperature, seed)` generates one continuation from the fixed LM; async `evaluate(prefix, continuation)` returns a finite NumPy metric vector. The owner computes the reference mean, draws a fixed calibration sample bank at temperature 1, differentiates its weighted metric mean via the metric covariance, fits coefficients with Adam, retains the lowest observed fitting error, and performs categorical resampling with weights `softmax(-metrics @ coefficients)`. The LM itself is never trained. Coefficients and empirical convergence are returned alongside generated text and particle weights.

Metric directions need no scalarization: the target is matching each reference expectation. The paper's relative error is undefined for a measured zero target, which raises explicitly. Finite calibration samples may make the requested target infeasible; the iteration cap returns the best coefficients with `converged=False`. Zero coefficient initialization and best-observed retention are practical adaptations. Default inference temperature 1 preserves the proposal assumed by the importance weights; a different temperature follows the paper's practical Algorithm 2 heuristic without a likelihood-ratio correction and changes its limiting distribution. There is no claim that finite samples reproduce the paper's exact perplexity guarantee.

Generation budget is `len(references) * fit_samples_per_prefix + len(prefixes) * particles`; metric calls add `len(references)`. Nonfinite measurements/objectives and blank generations raise; no retries. The model primitive owns tokenizer and decoding infrastructure. No optimization prompts are generated, so `prompts/` is reserved for application presentation.
