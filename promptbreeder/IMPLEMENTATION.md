# Promptbreeder implementation plan

Scope: complete the existing problem-agnostic optimizer from the supplied paper,
using Slick 0.3.0 and the co-author's later minimal implementation as a reference.

Existing boundary: nine JSON-generating methods, caller-owned evaluation and
similarity, no sessions or retries, exceptions propagate, three evaluations per
tournament, final-population best only. Existing tests pass (3 tests).

- [x] Add regression checks for paired generations, evaluation counts, full elite
  histories, context replacement/resampling, missing-evidence fallback, and the
  sequential inference boundary using `tests/providers.py`.
- [x] Implement those behaviors in `agent.py`; preserve text continuations and
  paper delimiters in separate local templates. Reuse the author repository's
  seed lists as defaults with its license and pinned provenance.
- [x] Document the generic evaluator contract, training/held-out separation,
  budgets, failures, and explicit deviations; run focused tests and Ruff.

Design choices: stored-fitness binary tournaments in shuffled disjoint pairs;
one offspring evaluation per event; immutable evaluated strategies and a
best-ever archive. Caller controls random training batches and scoring. A
convenience inference method executes an arbitrary prompt sequence without
exposing labels. Correct workings enter offspring contexts before evaluation.
Unavailable Lamarckian evidence falls back to direct mutation and is recorded.

Verification: 11 focused tests pass; review found no critical/important issues.
These checks exercise mechanics with scripted providers, not benchmark accuracy.
