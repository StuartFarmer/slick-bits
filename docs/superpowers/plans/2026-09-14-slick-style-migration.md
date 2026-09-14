# Slick style migration implementation plan

**Goal:** Apply the approved Slick development style to the existing algorithms without changing their experiments.

**Spec:** [Slick development style guide](../../../skills/slick-development/references/style-guide.md).

**Architecture:** Keep current callable and module boundaries unless a concrete simplification warrants a change. Move all 17 model-facing docstrings into each implementation's own `<algorithm>/prompts/<operation>.j2` folder, as requested by the user. Configure that application's absolute Slick template root at application and test startup, preserving library import behavior. Keep parsing, evaluator validation, logs, retries, budgets, and selection in their current ownership boundaries.

**Constraints:** No new dependencies, implicit sessions, human-review gates, schema tightening, generated-artifact edits, or paid provider calls. Preserve exact rendered instructions and schema output. Use the existing per-application offline checks. Apply Ruff formatting at 100 columns and import ordering to maintained source and tests. Retain deliberate `ponytail:` limitation markers as required by the active mode.

- [x] Agent A: EoH, AEL, Optimizer; externalize three proposal prompts, update startup and focused tests, preserve checked source/evaluator contracts.
- [x] Agent B: APEX and EvoPROMPT; externalize six prompts, update startup and focused tests, preserve tagged output and immediate/snapshot search behavior.
- [x] Agent C: ReEvo, QUBE, LLM_GP; externalize eight prompts, update startup and focused tests, preserve reflection, clustering, and fallback policies.
- [x] Parent: Review Scout and shared formatting configuration; check imports and callable compatibility across algorithm groups.
- [x] Compare pre-migration and post-migration renders across branches, run all relevant offline suites, inspect changes, and record results.

Each agent owns only its assigned directories, including their local prompt subdirectories. The parent owns shared files and integration. A pre-edit source snapshot is retained outside the workspace for comparison because this workspace has no Git metadata.
