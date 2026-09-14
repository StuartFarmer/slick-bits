---
name: slick-development
description: Use when implementing, refactoring, or reviewing Python algorithms built with Slick, including prompt methods, external Jinja templates, generated output contracts, sessions, and workflow orchestration.
---

# Slick development

Write algorithms as ordinary Python with explicit generation boundaries. Keep model instructions in external Jinja templates and make validation, state, and failure policy visible to callers.

Read [the style guide](references/style-guide.md) before designing or reviewing a Slick boundary. It defines the house style, API details, and migration checks. Use the [checked proposal example](assets/checked_proposal.py) and its [template](assets/prompts/propose.j2) when a concrete example helps. The example is a new design, not a behavior-preserving patch for an existing algorithm.

## Required development patterns

1. **Local prompts.** Each agent owns a `prompts/` folder inside its own directory. Never collect all agents' prompts into a central directory.
2. **Separate prompt operations.** Give initialization, crossover, mutation, revision, and other distinct operations their own decorated methods and prompt files. Choose operations in Python. Jinja must contain no `if`, `elif`, `else`, conditional expressions, or search/control logic. Render supplied data; simple data iteration and serialization are fine.
3. **One owning class per optimizer.** Encapsulate task context, dependencies, prompt methods, and search state in a cohesive class. Keep `async run()` short by calling named logical phases. Do not leave the algorithm as a large procedural script or hide the entire loop in one helper.
4. **Trust caller inputs.** Do not add `isinstance`, exact `type`, or `callable` checks. Assume the annotated inputs and let errors surface where values are used. Do not substitute input-validation decorators, Config validators, or preflight checks for configuration ranges, positivity, probabilities, or finiteness. Generated-output checks and runtime search decisions remain separate concerns.

## Working contract

Inspect the target algorithm, callers, tests, and actual Slick installation first. Record the current prompt format, return shape, validation location, attempt budget, failure records, and session ownership. Verify API details against that version; this reference was checked against a local checkout declaring Slick 0.3.0. Structured templates explicitly request JSON using `{{ schema | tojson }}`; do not assume output instructions are appended automatically.

For new or intentionally migrated prompt boundaries:

| Concern | House convention |
| --- | --- |
| Generated structure | Explicit `output_type=Model`; the return annotation describes the postprocessed result. |
| Postprocessing | Keyword-only `generated`; validate domain facts or transform the parsed result in the body. |
| Execution | Exactly one call-time `provider=` or `session=`; retries and budgets have an explicit owner. |
| Problem boundary | Inject task, provider, and async evaluation; keep benchmarks and execution infrastructure outside the agent. |
| Test generation | One shared scripted provider in `tests/providers.py`; no dummy providers in agent implementations. |

Use `instance` for method-owner attributes in Jinja. Resolve the template root once at application startup; it is process-global in the checked API. Bound method `.render` needs explicit owner binding; see the guide. Preserve raw response logging when moving validation into a decorated body.

Keep text as text when the algorithm consumes prose or requires a paper's delimiter format. Preserve internal dataclasses and pure numerical helpers when they fit. Add sessions for intentional history or tools, and Inbox review for an actual external decision point. Independent concurrent calls cannot share one mutable Session.

## Migration and review

Separate template extraction from prompt rewriting, schema changes, and movement of failure boundaries. Compare rendered prompts first. Preserve RNG order, population snapshots, score direction, call/evaluation counts, fallback policy, evaluator isolation, and held-out data boundaries. A style cleanup must not silently change an experiment. When the user explicitly requests a problem-agnostic redesign, replace domain-specific prompts and APIs deliberately; retain algorithm decisions and move the old benchmark/infrastructure aside. Never replace isolated evaluation with direct execution inside the generic agent.

Verify each operation template, typed generated parsing, candidate rejection, and failure accounting with the shared scripted provider and relevant algorithm tests. Review the four required patterns above before completion. Runtime budget accounting, generated-output contracts, and measured-score checks are separate algorithm behavior. Test supported launch directories after extracting templates.

Extract logical phases rather than moving the whole loop into one helper. Avoid one method per trivial assignment or a shared orchestration framework. Use separate prompt files instead of switching on a Python operation or mode inside Jinja; loops over supplied data are fine.

Report the changed behavior, evidence from checks, and remaining limits. Distinguish a demo, schema validation, syntax/interface validation, evaluated fitness, and a reproduced result. Apply this guidance within the user's requested scope; it does not authorize wholesale refactoring or new review gates.
