---
name: slick-development
description: Use when implementing, refactoring, or reviewing Python algorithms built with Slick, including prompt methods, external Jinja templates, generated output contracts, sessions, and workflow orchestration.
---

# Slick development

Write algorithms as ordinary Python with explicit generation boundaries. Keep model instructions in external Jinja templates and make validation, state, and failure policy visible to callers.

Read [the style guide](references/style-guide.md) before designing or reviewing a Slick boundary. It defines the house style, API details, and migration checks. Use the [checked proposal example](assets/checked_proposal.py) and its [template](assets/prompts/heuristic/propose.j2) when a concrete example helps. The example is a new design, not a behavior-preserving patch for an existing algorithm.

## Working contract

Inspect the target algorithm, callers, tests, and actual Slick installation first. Record the current prompt format, return shape, validation location, attempt budget, failure records, and session ownership. Verify API details against that version; this reference was checked against a local checkout declaring Slick 0.3.0.

For new or intentionally migrated prompt boundaries:

| Concern | House convention |
| --- | --- |
| Instructions | Each implementation owns `<algorithm>/prompts/<operation>.j2`; Python docstrings explain behavior. |
| Generated structure | Explicit `output_type=Model`; the return annotation describes the postprocessed result. |
| Postprocessing | Keyword-only `generated`; validate domain facts or transform the parsed result in the body. |
| Input validation | Before generation: constructor, caller, or `@validate_call` outside `@prompt`. |
| Execution | Exactly one call-time `provider=` or `session=`; retries and budgets have an explicit owner. |
| State | Functions for independent operations; a cohesive class for shared context or lifecycle. |

Use `instance` for method-owner attributes in Jinja. Resolve the template root once at application startup; it is process-global in the checked API. Bound method `.render` needs explicit owner binding; see the guide. Preserve raw response logging when moving validation into a decorated body.

Keep text as text when the algorithm consumes prose or requires a paper's delimiter format. Preserve internal dataclasses and pure numerical helpers when they fit. Add sessions for intentional history or tools, and Inbox review for an actual external decision point. Independent concurrent calls cannot share one mutable Session.

## Migration and review

Separate template extraction from prompt rewriting, schema changes, and movement of failure boundaries. Compare rendered prompts first. Preserve RNG order, population snapshots, score direction, call/evaluation counts, fallback policy, evaluator isolation, and held-out data boundaries. A style cleanup must not silently change an experiment.

Verify template branches, typed parsing, domain rejection, and failure accounting with deterministic providers and the relevant existing algorithm tests. Invalid external input should fail before a provider call when the public contract requires that. Test supported launch directories after extracting templates.

Report the changed behavior, evidence from checks, and remaining limits. Distinguish a demo, schema validation, syntax/interface validation, evaluated fitness, and a reproduced result. Apply this guidance within the user's requested scope; it does not authorize wholesale refactoring or new review gates.
