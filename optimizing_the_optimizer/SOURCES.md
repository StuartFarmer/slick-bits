# Source and implementation notes

Paper: Camilo Chacón Sartori and Christian Blum, *Optimizing the Optimizer:
An Example Showing the Power of LLM Code Generation*. This implementation uses
the manuscript supplied with the task, especially Sections II-C and III-A–C.

Inspected 2026-09-14:

- [Official repository](https://github.com/camilochs/optimizing-the-optimizer).
  The repository contains a README and MIT license, and links externally to code.
- [Official modified implementation archive](https://imp-opt-algo-llms.surge.sh/static/cmsa-mis-llm.zip),
  linked by that README. Downloaded and inspected all four `generate_solution`
  implementations, the shared `run_cplex` age updates, and the outer `main` loop.
- [Original CMSA archive](https://www.iiia.csic.es/~christian.blum/downloads/MIS_CMSA.tgz),
  also linked by the README. Retrieval failed (404 in the web fetch, 503 on direct
  retry). The baseline constructor therefore follows the supplied paper's
  Listing 1; shared CMSA behavior comes from the downloadable modified code.
- The README's chatbot demo text has no linked implementation. Prompt wording here
  adapts the supplied manuscript's dialogue and performance prompt to arbitrary
  languages/tasks and structured Slick output; it is not an exact chatbot export.

The downloaded zip's SHA-256 is
`f18cd9863d2b1252db3c99b4155020d5b3237df6ab89d7441c84b767f3c627db`.

| Archive member | SHA-256 |
| --- | --- |
| `mis-cmsa-llmv1.cpp` | `1026cc2ae4326c0207d971b4a5eed42aca23bb228d6b9efb4a64dc045fd02958` |
| `mis-cmsa-llmv2.cpp` | `6582e9e1544db440ade99738158abe85d9397952c1dc76902eaee064d35e0458` |
| `mis-cmsa-llmv1-perf.cpp` | `1592680bfb2256ec2bdfea38f71feed089c8bb8d62553b140f455fcbcfb565bb` |
| `mis-cmsa-llmv2-perf.cpp` | `92b81bcb1466dc44e41fd8f0d64dd1e36eae94b51a37db4c42790bb279186c4e` |
| `Makefile` | `2eb90fdf878eec2eb61fb999d88d004f1e9efebb5007b0cd199a8993288c9f2d` |

## Decisions grounded in the official source

| Behavior | Source and implementation choice |
| --- | --- |
| Deterministic V1/V2 selection | All four C++ constructors choose `active_vertices[0]`, initially ordered by full graph degree. `CMSA._select` follows this, rather than the manuscript's argmin of weighted probability. |
| Weight support | Weights are recomputed over currently feasible active components, not all vertices in the graph. |
| V1 | `1/(2+age) + 1/(1+degree)`; absent age -1 is valid. `selection_probabilities` substitutes caller cost for degree. |
| V2 | Normalize weights, add entropy to each probability, then renormalize their sum. The resulting denominator is `1+n*H`, which resolves the manuscript's ambiguous denominator. |
| Merge | `generate_solution` sets age -1 to 0 as soon as a component is selected. It does not reset positive ages on reconstruction. |
| Adapt | `run_cplex` adapts only after an Optimal or Feasible result: increase active ages, reset solver-selected ages, then expire ages `>= age_limit`. |
| Incumbent | `main` considers both construction and solver scores, retaining strict improvements. |
| PERF variants | The constructors retain the same selection rules and differ in storage, compaction and prefetching. Python exposes the two distinct heuristics, without duplicate PERF algorithms or a 32,768-component bitset cap. The Slick workflow can separately request implementation-level performance revisions. |

## Deliberate generalizations

The paper describes researcher-guided dialogue, not a specified autonomous search
algorithm. `OptimizingTheOptimizer` automates a bounded version: an evaluated
baseline, heuristic proposal, feedback-driven revisions, optional performance
branches and strict best-candidate retention. Rejection accounting, structured JSON,
full replacement source, and automatic score-based acceptance are explicit local
choices. Entropy is implemented in the reusable CMSA V2 selector; the generic
revision prompt does not force that idea on unrelated algorithms.

`CMSA` replaces graph neighborhoods and CPLEX with feasibility, solving and
evaluation callbacks. Static degree becomes caller-supplied cost; components are
hashable IDs. This covers subset constructions whose terminal greedy solutions
are feasible, not every possible optimization representation. The source-improvement
workflow is independent of this restriction and can target any optimizer source.

The Python loop adds a fixed iteration cap, a seeded Python RNG, optional wall-time
budgeting, objective minimization, and finite-score/pool-membership checks. It
does not reproduce C++ RNG sequences or unspecified degree-tie ordering. It omits
the original CPU clock, 0.1-second CPLEX startup threshold, warm-start flags,
heuristic emphasis and early-abort callbacks; solver policy belongs to the
injected solver. Rescanning feasibility favors a small generic implementation
over the original MIS-specific incremental data structures.

The archive requires CPLEX headers/libraries. Full CPLEX runs, irace tuning,
GPT-4o calls, and the paper's benchmark comparison are not part of the offline
verification.

As an additional source check, the unchanged `generate_solution` functions from
the official V1 and V2 files were extracted into temporary C++17 harnesses and
compiled with Clang. On a three-component clique with ages `[-1, 2, 7]`, 30,000
constructions per variant agreed with the Python probabilities within 0.012
absolute frequency per component. Observed V1 frequencies were
`[0.56517, 0.25037, 0.18447]` versus probabilities `[0.56471, 0.24706, 0.18824]`;
V2 frequencies were `[0.39403, 0.30960, 0.29637]` versus
`[0.39195, 0.31148, 0.29658]`. Both harnesses also checked deterministic first-in-order
selection and preservation of an existing positive age. This verifies those
constructor behaviors, not solver parity or benchmark performance.

## License and API provenance

The C++ files carry **Christian Blum, copyright 2024, GPL version 2 or later**,
despite the repository's separate MIT license. The source-derived Python CMSA
adaptation in `cmsa.py` preserves that attribution and uses GPL-2.0-or-later;
[COPYING](COPYING) contains the GPL v2 text. No upstream CPLEX code or binaries
are vendored here.

Slick boundaries were checked against the adjacent checkout declaring `slick-ai`
0.3.0 (`slick/prompts.py`, `slick/providers/base.py`, `slick/session.py`), using
Python 3.14.0 and Pydantic 2.13.4. Templates explicitly include the JSON schema,
use `instance` for owner state and contain no conditional control flow. Calls use
one explicit `provider=` with a per-attempt raw-response recorder. Parsing and
blank-output failures are distinguished from evaluator rejection and operational
exceptions; retries are not delegated to Slick or a mutable Session.
