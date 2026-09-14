# Source provenance

Paper: Zui Chen et al., *SEED: Domain-Specific Data Curation With Large Language
Models*, [arXiv:2310.00749v3](https://arxiv.org/abs/2310.00749v3).
The user supplied the complete paper, including algorithms and appendix prompts.

Official code: [Magolor/SEED](https://github.com/Magolor/SEED), inspected at commit
`3686e9913ef0719ec82156b6bdaf34f28b2506b8` on 2026-09-14.
The paper's anonymous repository has a working
[README file endpoint](https://anonymous.4open.science/api/repo/SEED/file/README.md)
that identifies this GitHub repository. Its landing/metadata endpoints did not
work during discovery. The code was cloned and read; its installers were not run.

This is a problem-agnostic reimplementation using the official algorithms as a
reference, not a vendored package or a claim of byte-for-byte behavior preservation.

| Official source at pinned commit | Use here |
| --- | --- |
| [optim_simul/simul/optim_seed.py](https://github.com/Magolor/SEED/blob/3686e9913ef0719ec82156b6bdaf34f28b2506b8/optim_simul/simul/optim_seed.py) | First-non-abstaining routing, configuration expansion, skyline and effectiveness-gap selection; replaced task fields and fixed ordering with injected predictions and paper Eq. 2. |
| [optim_simul/simul/optim_bf.py](https://github.com/Magolor/SEED/blob/3686e9913ef0719ec82156b6bdaf34f28b2506b8/optim_simul/simul/optim_bf.py) | Subplan expansion reference. Despite its filename, this source prunes dominated plans; our `exhaustive` mode actually disables pruning. |
| [SeeD/src/seed/agents/codeg.py](https://github.com/Magolor/SEED/blob/3686e9913ef0719ec82156b6bdaf34f28b2506b8/SeeD/src/seed/agents/codeg.py) | Advice/source generation, failure-based branching, precision/coverage/failure profiles and code dominance. Preserved cautious snippets when broader coverage introduces errors. |
| [legacy/SeeD/src/seed/codegen.py](https://github.com/Magolor/SEED/blob/3686e9913ef0719ec82156b6bdaf34f28b2506b8/legacy/SeeD/src/seed/codegen.py) | Operation-specific advice, code, example-generation and repair prompts, adapted to Slick JSON contracts and a generic `solve(input)` interface. |
| [legacy/SeeD/src/seed/cache.py](https://github.com/Magolor/SEED/blob/3686e9913ef0719ec82156b6bdaf34f28b2506b8/legacy/SeeD/src/seed/cache.py) | Nearest-neighbor reuse and distance gate. For unit vectors, its squared-L2 gate `distance <= 2 * threshold` equals our cosine-distance gate. |
| [SeeD/src/seed/agents/model.py](https://github.com/Magolor/SEED/blob/3686e9913ef0719ec82156b6bdaf34f28b2506b8/SeeD/src/seed/agents/model.py) | Accumulate labels, refit periodically, then confidence-gate predictions. Checkpoint/training work is injected. |
| [SeeD/src/seed/utils/lm.py](https://github.com/Magolor/SEED/blob/3686e9913ef0719ec82156b6bdaf34f28b2506b8/SeeD/src/seed/utils/lm.py) | Inverse-perplexity confidence. Its classifier uses `max(p) - 1/K`; we deliberately use the normalized formula in paper §4.2, which also agrees with the legacy binary implementation. |
| [legacy/SeeD/src/seed/utils/utils.py](https://github.com/Magolor/SEED/blob/3686e9913ef0719ec82156b6bdaf34f28b2506b8/legacy/SeeD/src/seed/utils/utils.py) | Closest/farthest grouping and balanced assignment. We handle remainders, use current-batch distances as in the paper, and implement balanced Lloyd iterations with SciPy assignment rather than adding scikit-learn. |
| [SeeD/src/seed/interface.py](https://github.com/Magolor/SEED/blob/3686e9913ef0719ec82156b6bdaf34f28b2506b8/SeeD/src/seed/interface.py) | Random, label-balanced and retrieved examples; round-robin balanced selection fills the requested count when group sizes differ. |

Additional adaptations: fixed development/validation separation, JSON output IDs
instead of `eval`-parsed response lines, explicit null versus abstention, vote ties
abstain, and a retained LLM fallback. Tool calls use native Slick sessions and a
typed final response instead of a hand-parsed `Thought/Action/SUBMIT` protocol.
The paper's optional logical-review dialogue, prompt-rephrasing generator, and
ternary search are not separate stages: measured development errors drive repairs,
diverse advice drives initialization, and finite grids drive threshold search.

## Upstream replay check

Extracted the inspected upstream `run_pipeline` function and replayed three
two-module pipelines over all **1,206** saved Abt-Buy records in
`optim_simul/data/Abt-Buy/Abt-Buy_results.jsonl`. Compared every returned prediction
and LLM-routing fraction with this implementation's cached validation. Cheap-module
costs were set to zero and LLM cost to one, matching the source's count-based cost.

| Pipeline | Matching predictions | LLM routing fraction |
| --- | ---: | ---: |
| Code ensemble → sample LLM | 1,206 / 1,206 | 0.840796 |
| Cache (`confidence >= .95`) → sample LLM | 1,206 / 1,206 | 0.000000 |
| Model (`confidence >= .8`) → sample LLM | 1,206 / 1,206 | 0.103648 |

These are checks against saved upstream outputs at selected thresholds, not newly
generated predictions, training runs, or a reproduction of the paper's reported metrics.
