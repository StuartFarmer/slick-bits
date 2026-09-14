# RLPrompt

`RLPrompt` implements fixed-length autoregressive on-policy soft Q-learning from [Deng et al., 2022](https://arxiv.org/abs/2205.12548). The inspected [SQL module](https://github.com/mingkaid/rl-prompt/blob/main/rlprompt/modules/sql_module.py), [configuration](https://github.com/mingkaid/rl-prompt/blob/main/rlprompt/modules/module_helpers.py), and [loss implementation](https://github.com/mingkaid/rl-prompt/blob/main/rlprompt/losses/sql_losses.py) define the supported variant: average temporal-advantage and reverse-cumulative-advantage squared losses, Polyak target updates, and greedy inference. Although the configuration names reversed variants, the supplied `coefficient=None` disables the reversed terms; this port follows that executed default.

```python
agent = RLPrompt(task, forward, evaluate)
result = await agent.run(parameters, queries, prompt_length=5, iterations=100)
```

`parameters` is a flat NumPy vector. Async `forward(parameters, query, prefix_ids)` returns next-token Q logits `[vocabulary]` and Jacobian `[vocabulary, parameters]`. The adapter owns the frozen compact LM, trainable MLP, tokenizer and LM head; the owner performs categorical sampling, both losses, chain rule, target updates and Adam. Async `evaluate(query, token_ids)` returns higher-is-better reward. Queries condition generation; use an empty query for a task-level policy. Reward standardization groups repeated samples by query. Optional affine reward scaling is explicit; the source classification setting maps [0,100] to [-10,10] with `normalize_rewards=False, reward_scale=0.2, reward_offset=-10`.

Exactly `iterations * len(queries) * samples_per_query` rewards are measured. Final greedy prompts are policy outputs, not a validated best-so-far archive. There is no EOS truncation, top-k/top-p filtering, task-specific bootstrap reward aggregation or automatic checkpoint selection. These are explicit adaptations around the released loss, not a claim of benchmark reproduction. Nonfinite model outputs, derivatives and rewards raise; errors propagate without retries. Numeric token generation does not need prose templates; the local `prompts/` directory is reserved for application-owned presentation.
