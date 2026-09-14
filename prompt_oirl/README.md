# Prompt-OIRL

`PromptOIRL` learns an offline correctness proxy from query/prompt embeddings and selects the highest predicted reward separately for each new query. It follows [Sun et al., 2023](https://arxiv.org/abs/2309.06553) and the released Llama [data processing](https://github.com/holarissun/Prompt-OIRL/blob/main/llama_exps/llama_step3_data_processing.py), [proxy training](https://github.com/holarissun/Prompt-OIRL/blob/main/llama_exps/llama_step4_offline_evaluation.py) and [selection](https://github.com/holarissun/Prompt-OIRL/blob/main/llama_exps/llama_step5_offline_optimization.py) programs. The released proxy is XGBoost binary-logistic trees, not a neural reward model.

```python
agent = PromptOIRL(task, embed)
result = await agent.run(offline_observations, new_queries, candidate_prompts)
# Observation(query, prompt, correct), correct in [0, 1]
```

Async `embed(text)` returns a finite fixed-width NumPy vector. The owner concatenates query and prompt embeddings, fits Newton gradient-boosted trees with logistic gradient/Hessian and regularized split gain, and ranks candidates. Defaults retain the source's 2,000 rounds, depth 10 and rate 0.001. Its greedy exact NumPy splitter is an explicitly small-data adaptation of XGBoost: no histogram/approximate split search, missing-value handling, column subsampling, sparse storage or pretrained checkpoints. Regularization and minimum child Hessian are configurable. Inputs are caller-curated offline target-model measurements; embedding results are copied and cached per run.

Selection never queries target-model rewards or held-out labels. Candidate proposal and held-out benchmarking are caller-owned, as in the release's precollected prompt lists. Probability estimates are proxy predictions, not measured correctness. Errors and nonfinite measured embeddings/rewards/losses propagate without retries. This numerical method has no generated prose operation; `prompts/` is reserved for application presentation.
