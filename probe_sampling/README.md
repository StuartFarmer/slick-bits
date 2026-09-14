# Probe sampling

`ProbeSampling(task, provider, draft_evaluate, target_evaluate).run(initial)`
generates a candidate batch, evaluates all candidates on the cheaper draft model,
and evaluates random probes on the target model. Spearman rank agreement selects
`max(int((1-correlation)/2 * filtered_size), 1)` draft-ranked candidates for target
evaluation. The best target result among **both probes and filtered candidates**
wins. Undefined correlation uses the full filter budget, as in upstream.
Both async evaluators return finite losses, **lower is better**.

Sources: [paper](https://arxiv.org/abs/2403.01251), authors'
[implementation](https://github.com/zhaoyiran924/Probe-Sampling/blob/main/llm_attacks/gcg/gcg_attack.py),
especially `GCGMultiPromptAttack.step`, lines 300–365 inspected 2026-09-14.

This ports the general candidate-filtering algorithm. The task-agnostic Slick
proposal operator deliberately replaces GCG's gradient-token proposal; use the
separate `gcg` implementation for coordinate search. Average ranks support ties
without SciPy. Probe measurements are reused when a filtered index overlaps;
upstream evaluates those again. Reuse assumes repeatability within a batch.
No cross-batch cache. The global best is retained separately from the trajectory.

Configure Slick's global template root once to this folder's absolute `prompts/`.
Malformed batches, provider errors, and invalid losses propagate without retries.
Tests check filtering and accounting, not the paper's speedup or task performance.
