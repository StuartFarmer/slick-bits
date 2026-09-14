# DGM implementation plan

Implement the supplied paper's algorithm in this folder using the official
`jennyzzt/dgm` source at `a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2`.

- [x] Exercise recursive execution, archive stepping stones, selection weights,
  both ablations, rejection accounting, and local templates with offline tests.
- [x] Implement one owning `DGM` class, typed diagnosis, and separate generation
  and modification prompts using the installed Slick API.
- [x] Preserve sigmoid/child-count selection and functional-child admission.
  Keep benchmark staging and isolated execution in caller-supplied adapters.
- [x] Document the executable seed interface, budgets, failure policy, official
  source mapping, and intentional generalizations; retain upstream licensing.
- [x] Run focused tests, formatting/lint checks, and an independent code review.

No additional dependencies, benchmark datasets, paid runs, or host execution of
generated programs. Tests use the repository's shared scripted provider.

Validation: 9 DGM tests and 11 related tests passed; scoped Ruff lint and format
checks passed. Independent review identified partial-batch result loss and an
ablation batching mismatch; both were reproduced and fixed with regression tests.
