# Official source and adaptations

- Paper: Zhang et al., [Darwin Gödel Machine: Open-Ended Evolution of
  Self-Improving Agents](https://arxiv.org/abs/2505.22954), 2025.
- Official implementation: [jennyzzt/dgm](https://github.com/jennyzzt/dgm), inspected
  at commit `a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2`.
- Upstream license: Apache-2.0, retained in [UPSTREAM_LICENSE](UPSTREAM_LICENSE).
  The code here is modified from/adapted to that implementation; it is not an
  unmodified upstream distribution.

| Official code | Use here |
| --- | --- |
| [`DGM_outer.py`](https://github.com/jennyzzt/dgm/blob/a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2/DGM_outer.py), `choose_selfimproves` | Reuses the `score_child_prop` sigmoid and inverse archived-child-count weighting, with sampling with replacement. |
| Same file, `update_archive`, `main` | Keep all functional offspring, retain source lineage, and select parents from an archive snapshot for each batch. |
| Same file, `no_darwin` branch | Select only the most recent functional agent in `no_open_ended`. |
| [`self_improve_step.py`](https://github.com/jennyzzt/dgm/blob/a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2/self_improve_step.py), `diagnose_problem`, `self_improve` | Fixed diagnosis provider followed by execution of the parent's current implementation. The no-self-improvement ablation uses the seed modifier. |
| [`prompts/self_improvement_prompt.py`](https://github.com/jennyzzt/dgm/blob/a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2/prompts/self_improvement_prompt.py) | A single general feature proposal grounded in code and evaluation feedback; retains `improvement_proposal`, `implementation_suggestion`, and `problem_description`. |
| [`utils/evo_utils.py`](https://github.com/jennyzzt/dgm/blob/a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2/utils/evo_utils.py), `is_compiled_self_improve` | Admission requires functional editing evidence; compilation alone does not establish it. |

## Intentional generalizations

The evolving artifact is a complete Python module with
`async agent(instruction, generate)`, rather than a Git patch against the
multi-file SWE-bench/Polyglot agent repository. The seed delegates once to a model;
its descendants can rewrite the workflow. This preserves recursive modification
while replacing benchmark-specific Bash/edit tools with a caller-owned worker
and generation RPC. Upstream repository patch application, Docker provisioning,
and benchmark dependencies are not bundled or imported.

All new model-facing instructions are local Jinja templates. Diagnosis uses
explicit Slick/Pydantic JSON parsing rather than upstream marker extraction.
Raw generation responses remain recorded even when rejected. The locally
inspected Slick source is the adjacent `slick-ai` 0.3.0 checkout.

The evaluator supplies normalized quality, functional-editing evidence, and
feedback. Upstream empty-patch/context-length/stochasticity heuristics and task
selection become general feedback-guided diagnosis. The SWE-bench/Polyglot
10/50/200-task staging and promotion thresholds belong to the evaluator, not the
generic search. Upstream's per-benchmark score aggregation is not replicated.
For arbitrary downstream domains, quality improvements need not imply improved
self-editing skill; this paper assumption must be assessed for the chosen tasks.

The default search makes 80 child attempts, with one child per batch unless
configured otherwise. Upstream CLI defaults to 80 generations and two attempted
children per generation. Workers run serially here; batch snapshots are retained
and source order is deterministic rather than completion-order dependent. The
completed children are archived immediately to preserve work if a later sibling
fails; they become selectable in the next batch. The single diagnosis attempt
and explicit model-call budget replace upstream's diagnosis retries and
long-running coding conversations. Rejecting byte-identical
source corresponds to upstream's rejection of an empty self-modification patch;
no additional semantic deduplication or improvement-only filter is imposed.

In `no_self_improve`, the fixed seed edits the selected parent's complete source.
This makes the paper's fixed-modifier ablation explicit without upstream's
benchmark-specific patch reconstruction. `no_open_ended` keeps historical data
for inspection but excludes it from selection. Like upstream, that ablation
takes one child attempt per batch.

These are algorithm/interface adaptations, not a claim of prompt equivalence,
matching random trajectories, or reproduction of reported benchmark scores.
