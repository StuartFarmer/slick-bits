# CMSA with Slick

Implement the supplied paper as a standalone Python project in `optimizer/`.
Use the local Slick checkout for typed code-generation prompts and an explicit,
saved human-feedback dialogue. Use SciPy/HiGHS for the reduced MIS integer program
and NetworkX for the three graph families. Keep model output reviewable before
the user explicitly loads it as a construction function.

The authors' linked C++ archive was inspected on 2026-09-13. Follow its
minimum-degree deterministic branch, normalization over available vertices,
entropy renormalization, and removal at `age >= age_max`. Document discrepancies
with the typeset equations. PERF variants will use Python integer bitsets;
they are behavior-preserving Python alternatives, not reproductions of C++ cache
alignment or prefetch instructions. No claim of reproducing published rankings.

1. [x] Add one runnable check for known MIS optima, construction feasibility,
   age adaptation, normalized probabilities, solver timeout behavior, and
   equivalent random choices in ordinary/PERF construction.
2. [x] Implement graph validation/loading, all five construction variants, reduced
   MILP solving, bounded CMSA, and convergence records in `cmsa.py`.
3. [x] Add `experiment.py`: seeded graph generation or input files, equal budgets,
   per-instance results and mean ranks, JSON output with parameters and versions.
4. [x] Add `improve.py`: typed Slick prompt with complete CMSA source, heuristic and
   performance modes, saved previous exchanges, feedback, offline demo, and
   explicit provider/model selection. Validate syntax without executing output.
5. [x] Extend the same check to cover graph input and the real Slick rendering/parsing
   path with a scripted provider. Run it and smoke-test both CLIs. Document setup,
   commands, interpretation choices, and reproduction limits in `README.md`.
