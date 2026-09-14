# Slick style and generic-agent validation

Checked 2026-09-14 against the active workspace and adjacent Slick checkout.
The eight optimizers now expose problem-agnostic classes with decorated instance
methods, local Jinja templates, and short async `run(..., *, session=None)`
methods coordinating named algorithm phases. The supplied PaperPlanner remains the design reference.

## Current redesign

| Implementation | Preserved agent decisions | Local templates |
| --- | --- | --- |
| AEL | Crossover/mutation probabilities, generation snapshots, stable elitism | `initialization.j2`, `crossover.j2`, `mutation.j2` |
| APEX | Sentence edits, immediate beam updates, history reversal, LinUCB | `mutate.j2`, `mutate_guided.j2` |
| EoH | Five operators, ranked selection without replacement, fixed attempt counts | `initialize.j2`, `explore_diverse.j2`, `explore_shared.j2`, `modify_structure.j2`, `tune_settings.j2`, `simplify.j2` |
| EvoPrompt | GA/DE sampling, distinct donors, frozen generations, strict DE acceptance | `ga_offspring.j2`, `de_offspring.j2`, `variation.j2` |
| LLMGP | Tournament/elite variant and model selection/replacement/designation variant | `initialize.j2`, `mutate.j2`, `crossover.j2`, `select_parents.j2`, `replace_population.j2`, `designate_best.j2` |
| Optimizer | Initial proposal, explicit feedback revisions, strict improvement | `propose.j2`, `revise.j2` |
| QUBE | Signature clusters, quality/uncertainty selection, island resets | `generate.j2` |
| ReEvo | Worse/better pairing, short/long reflection, elite mutation, evaluation budget | `initial.j2`, `crossover.j2`, `mutate.j2`, `reflect_pair.j2`, `reflect_long.j2` |

Each template is inside its implementation's own `prompts/` directory. All 28
prompt methods have keyword-only `generated` inputs. Structured boundaries
explicitly include JSON schemas; prose and tagged-text contracts remain textual
where those represent the algorithm's output directly.

Task descriptions and async evaluators replace built-in problems. APEX accepts a
synchronous embedding callback; QUBE evaluation includes a behavior signature.
The classes construct no providers and execute no candidate code. There is one
scripted provider, in `tests/providers.py`, also used by the reusable skill example.
No shared agent base class, CLI framework, or new dependency was introduced.

This intentionally changes APIs, candidate contracts, and model instructions.
It does not claim prompt equivalence or paper-performance equivalence. Traditional
symbolic GP/random baselines and benchmark answer-generation helpers are outside
the active agent flows. The original infrastructure, restrictions, and tests are
preserved in the [legacy archive](../examples/legacy/README.md): 70 source/template/
configuration files, verified byte-for-byte before removal from active directories.
Its SHA-256 is `9ac565e438c96ea446bdc18c9bc8a16f2f8d1b526b322d5bbec966f91c5c52ce`.
Historical runs, environments, and Scout remain outside this redesign.

## Readability cleanup

Caller types are trusted instead of repeatedly checked with `isinstance`, exact
`type`, or `callable`. Incompatible values fail where operations use them. APEX's
single `isinstance` branch selects its two supported document formats; it is
intentional dispatch. Caller-configuration range, positivity, probability, and finiteness guards have
also been removed from constructors, Config classes, and run preambles. Generated
contracts, measured-score checks, runtime budgets, and rejection rules remain. ReEvo, QUBE, and LLMGP no longer enforce
fresh-instance usage through a `started` guard; their documented lifecycle still
calls for a fresh instance. Evaluator programming errors propagate rather than
being swallowed as candidate rejection.

Large run bodies were split into initialization, selection, generation,
reflection, assessment, and replacement phases. ReEvo's run now shows the sequence
of reflective evolution directly. Fixed-seed APEX and GA/DE results were compared
before and after extraction and matched exactly. Additional LLMGP probes checked
partial-budget histories during initialization, variation, replacement, and final
designation.

Every operation has a separate decorated prompt method and local file. Python
chooses among operations; Jinja only renders supplied data. A QUBE regression verifies that zero temperature reaches sampling and raises the
native division error after seed evaluation. A persistent test
parses all agent and skill-example templates and rejects Jinja `If` or `CondExpr`
nodes. The guide and reusable example now follow these conventions too.

## Verification

- **63 offline unittest checks passed**, using the existing `optimizer/.venv`.
  The latest full run used the repository root; the preceding phase cleanup also
  passed from an unrelated temporary working directory with absolute template roots.
- Tests cover unrelated task contexts, caller-supplied scoring, selection and
  mutation decisions, snapshots, elitism/ties, budgets, finite-score rejection,
  cancellation, malformed responses, and caller-owned Sessions.
- Independent subagent reviews led to regression fixes for AEL provider-timeout
  propagation, LLMGP provider-error ownership and boolean score direction, and
  QUBE overflow-safe offspring quality accumulation.
- An AST inventory confirmed eight owning classes, 28 local prompt methods,
  keyword-only run Sessions, no candidate execution or infrastructure imports in
  agent modules, and exactly one active scripted provider across agents, tests,
  and the skill example.
- `ruff check .` passed; `ruff format . --check` found 38 Python files formatted.
  Recorded runs are excluded.
- Skill frontmatter validation passed. The bundled example is now a generic
  `ProposalDesigner` with injected evaluation and an explicit revision loop;
  its separate proposal/revision rendering and improvement behavior pass through
  the shared provider test suite.

No paid model calls, Docker evaluations, dependency installations, or optimization
benchmarks were required. Tests establish control flow and contracts, not whether
any particular model discovers better candidates. Provider retries, raw response
recording, datasets, persistence, and evaluator execution isolation belong to
callers. Slick's checked template root remains process-global; simultaneous runs
requiring distinct roots need separate processes.

Repeat the current checks from the repository root:

```sh
rtk proxy optimizer/.venv/bin/python -B -m unittest discover -s tests
rtk proxy ../slick/.venv/bin/ruff check .
rtk proxy ../slick/.venv/bin/ruff format . --check
rtk proxy ../slick/.venv/bin/python /Users/stuart/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/slick-development
```

## Earlier comparison and template migration

The [original report](slick-style-report.md) inventoried 26 production files
across the eight applications and Scout: 7,243 lines and 17 inline prompt
functions. The first migration extracted those prompts without changing their
rendered content: 193 rendered cases matched, and all 14 existing offline test
commands passed (including two existing opt-in Docker skips in QUBE).

The initial skill was checked with an independent baseline review and a fresh
agent applying the guide. Both identified the key API and compatibility concerns;
this was not evidence of measured performance improvement from the skill. The
subsequent explicit request for generic classes supersedes the original
extraction-only compatibility constraints and the earlier domain-specific skill
example. The guide and example now reflect those requested boundaries.
