# Sources and implementation decisions

- Paper: [Co-Generation of Game Levels and Game-Playing Agents](https://arxiv.org/abs/2007.08497),
  supplied in full by the user, including Algorithm 1 and Table 1.
- Official repository: [aadharna/UntouchableThunder](https://github.com/aadharna/UntouchableThunder).
  Its README names the author, links this paper, and directs paper users to the
  release described as “ToG”. The actual published git tag is `1.0`.
- Inspected release: **`7abb7b746a43606d6e33ccefcca2efaa9652cc3a`**, September 2,
  2020. Sources were fetched and inspected locally; upstream code is not imported
  at runtime. This implementation independently expresses those algorithms with
  NumPy and async callbacks rather than vendoring the game infrastructure.

| Official source at pinned revision | Used here |
| --- | --- |
| [poet_distributed.py](https://github.com/aadharna/UntouchableThunder/blob/7abb7b746a43606d6e33ccefcca2efaa9652cc3a/poet_distributed.py) | Global child-attempt budget, uniform parent sampling, both solver checks, age culling, delayed transfer replacement, mutation at loop zero, transfer after each period |
| [generator/levels/base.py](https://github.com/aadharna/UntouchableThunder/blob/7abb7b746a43606d6e33ccefcca2efaa9652cc3a/generator/levels/base.py) | Mutation gate, remove/spawn/move choice, geometric continuation, inherited environment edits |
| [utils/DE.py](https://github.com/aadharna/UntouchableThunder/blob/7abb7b746a43606d6e33ccefcca2efaa9652cc3a/utils/DE.py) | DE/rand/1/bin, distinct donors excluding target, forced crossover coordinate, clipping, deferred generation selection and ties |
| [optimization/Optimizer.py](https://github.com/aadharna/UntouchableThunder/blob/7abb7b746a43606d6e33ccefcca2efaa9652cc3a/optimization/Optimizer.py) | Fresh uniform population with incumbent in slot zero |
| [utils/ADPChild.py](https://github.com/aadharna/UntouchableThunder/blob/7abb7b746a43606d6e33ccefcca2efaa9652cc3a/utils/ADPChild.py) | New optimizer per phase, `n_games // popsize` generations, F=0.6, CR=0.4, bounds [-5, 5] |
| [args.yml](https://github.com/aadharna/UntouchableThunder/blob/7abb7b746a43606d6e33ccefcca2efaa9652cc3a/args.yml) | Mutation probabilities and runtime defaults; outer iteration count uses the paper's 5000 rather than the release's 5-loop smoke setting |

Intentional adaptations:

1. Replace game-specific maps/CNN/GVGAI workers with serialized environments,
   arbitrary fixed-length numerical policies, and async evaluation/solver callbacks.
   Random and MCTS agents remain application responsibilities because their
   actions and forward models are domain-specific.
2. Offer separate Slick remove/add/move prompts as a new optional environment
   generator. These are **new prompts**, not prompts from the paper. A native
   callback can implement exact game-specific tile constraints instead.
3. Keep actual numerical DE, rather than labeling LLM revision as differential
   evolution. Scores maximize reward directly instead of negating it into loss.
4. Count initial DE evaluations and final reevaluation separately from the trial
   budget. Honor partial trial generations instead of dropping the remainder.
   Follow Algorithm 1's explicit post-optimization reevaluation; omit the
   distributed driver's redundant pre-mutation reevaluation.
5. Use an instance-local NumPy Generator. Sampling distributions and phase
   snapshots are preserved; the historical global RNG draw sequence (including
   unused random fitness draws) and exact game trajectories are not reproduced.
6. Run evaluations serially and retain in-memory records. Distributed workers,
   game assets, CNN architecture, alternative optimizers, benchmark execution,
   curriculum-extraction experiments, and disk checkpointing are outside this
   algorithm-focused implementation. No assertion of unbounded complexity or
   reproduced game performance is made.

Verification on 2026-09-14: the numerical optimizer was also compared directly
against `utils/DE.py` from the pinned release on a three-dimensional quadratic
objective, with 8 population members and 10 generations across 10 seeds. With
both implementations using matching legacy RNG streams, all final vectors
matched exactly (88 evaluations each). Only upstream's unavailable `tqdm`
progress display was substituted with an identity iterator. This comparison
establishes numerical agreement for those cases, not game-result reproduction.
