# Slick algorithm development style guide

This is the house style for Python algorithms built with Slick. It takes the supplied `PaperPlanner` example as the design reference and preserves the useful experimental discipline in slick-bits. It governs application code using Slick; changes to Slick internals must follow that library's own contracts and tests.

The API notes were checked on 2026-09-14 against the adjacent Slick checkout, whose package metadata declares version 0.3.0, using Python 3.14.0, Pydantic 2.13.4, and Jinja 3.1.6. A version number alone does not identify a local checkout. Recheck the actual source and tests when adopting this guide elsewhere.

**Requirements** below protect correctness or this project's requested external-template convention. **Defaults** are preferences to apply when they fit. **Conditional choices** need a concrete algorithm or application requirement.

## 1. Make the Python tell the algorithm's story

Use domain verbs such as `propose`, `assess_method`, `reflect_pair`, `revise`, and `evaluate`. Use domain nouns for results: `Proposal`, `Assessment`, `Individual`. Prefer `mutate` to `mutation_prompt` when the name is free to change: the decorator already identifies the mechanism. Preserve public names during compatible changes.

Arrange a small module as a concise module docstring, imports, aliases/constants, data contracts, pure checks, related operations, and entrypoint glue. Split cohesive responsibilities when they already have different dependencies or lifecycles. Do not create a module for every class.

A useful starting layout is:

```text
algorithm/
    core.py                    # selection, budgets, state transitions, pure math
    operators.py               # related decorated operations and their checks
    run.py                     # configuration, providers, files, command line
    prompts/
        propose.j2
        revise.j2
    test_algorithm.py
```

This is a default, not a mandatory file count. A small `algorithm.py` and a `prompts/` directory can be sufficient. Keep workers or evaluators separate where execution isolation requires it. Shared provider construction belongs at the application boundary; create a shared CLI helper only when the applications actually share its policy.

Use a class when methods share meaningful context or a run lifecycle, as `PaperPlanner` shares a paper or `ReEvo` owns one evolving population. Independent generation functions and numerical helpers can remain functions. A class should make ownership clearer; a one-method forwarding object usually does not.

## 2. Put model-facing prose in Jinja files

**Requirement for new or intentionally migrated prompts:** each implementation owns its own `prompts/` folder inside its directory. Use `@prompt(template="operation.j2", ...)` relative to that application's configured root. Python docstrings describe the callable to developers. Slick still supports inline prompt docstrings; they are a supported API that this house style chooses not to use for maintained new prompts.

Templates should normally contain, in order:

1. A direct task instruction.
2. The task interface and constraints that affect the answer.
3. Delimited source material, parents, or prior results.
4. The requested response contract, including `{{ output_format }}` for structured decorated calls.

Lead with verbs: “Compare these heuristics and propose one revision.” Role declarations such as “You are an expert” belong only where the experiment requires them. Use sentence case and short paragraphs. Keep uppercase emphasis for a concrete ambiguity, rather than treating it as the default voice.

Use `{{ value }}`, `{% if condition %}`, and `{% for item in items %}` consistently. Keep Jinja branches about presentation and operation-specific instructions; sampling, scoring, validation, retries, and acceptance remain Python decisions. Use `.model_dump() | tojson` for Pydantic data that should be serialized as JSON. Do not call model methods on plain dictionaries or dataclasses.

State which material is task data, the required output, and the limits of what was done. A prompt may request deterministic code or forbid execution, but those instructions do not establish execution isolation. Preserve evaluator controls separately.

**Paper-faithful mode:** keep the source prompt's wording, ordered steps, delimiters, few-shot examples, and word limits. A move to `.j2` should compare the rendered text, including indentation and schema placement. Rewriting a role instruction, replacing tagged text with JSON, or adding output constraints is an experimental change. Label it separately and evaluate it separately.

## 3. Respect template binding and loading

In the checked Slick API:

- `self` is exposed as `instance` in a decorated method's template: `{{ instance.paper }}`. Jinja reserves `self` for its own template object.
- Function inputs and their defaults become template variables. `generated`, `provider`, and `session` are reserved; do not provide them as template inputs.
- `slick.prompts.TEMPLATE_ROOT` defaults to `Path("prompts")`, relative to the working directory. Template files, includes, and imports resolve under that root at render time.
- The loader uses `StrictUndefined`: missing variables should fail loudly. Test every meaningful branch so an unvisited branch does not hide a typo.

For a standalone application with the layout above, set the root once during startup, before any render or provider call:

```python
from pathlib import Path

from slick import prompts

prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
```

This is application configuration, not something to repeat in each decorated operation or assign when importing a library module. The root is process-global: run independent implementations in separate processes when they need different roots at the same time. Programmatic callers configure the chosen implementation's root before rendering. Do not move their templates into a central directory or change the global root concurrently. If distributing a package, include template files in the distribution and verify they resolve after installation.

For an async decorated method, render with the owner explicitly supplied:

```python
text = await HeuristicDesigner.propose.render(designer, feedback)
```

In this implementation, `.render` is attached to the function and is not automatically bound through `designer.propose.render(...)`. An explicit render helper can hide this detail if callers need it. Test preserved `.render` entry points when replacing free functions with methods.

`Prompt("report.j2")` is a renderer, not a provider call. `.render(...)` on a decorated function renders without calling the provider or postprocessing body. Neither operation proves that generated output will pass domain validation. The decorator adds structured output instructions when needed; the plain `Prompt` renderer does not automatically add that decorated-call block.

## 4. Separate input, generated, and domain contracts

There are three boundaries:

| Boundary | Responsibility | Example |
| --- | --- | --- |
| Caller input | Reject unusable inputs before generation | Nonblank feedback, valid operation, positive budget |
| Generated output | Validate the representation | JSON fields, nonblank text, list cardinality, decision literals |
| Domain acceptance | Check what the representation means locally | Quote provenance, expression grammar, known IDs, candidate interface |

For a structured model result, always declare `output_type=...`. Slick does not infer parsing from the return annotation. The annotation describes the result after the function body runs, which may wrap or transform the generated value. Use keyword-only `generated` when the body needs that value, and explicitly return the intended result.

The checked decorator treats a body result of `None` or `Ellipsis` as “return the generated value.” It therefore cannot use `return None` to signal rejection. Raise a suitable exception, return an explicit result type, or handle rejection in an ordinary orchestration function. The reference's undecorated generation orchestrator and workflow review have different return semantics from a prompt body.

Use Pydantic for runtime input/output contracts. For new closed response objects, prefer `class Proposal(BaseModel, extra="forbid")`. The equivalent `ConfigDict(extra="forbid")` remains valid; changing spelling alone is cosmetic. `extra="forbid"` does not imply strict type coercion. Add strict constraints when coercion would be incorrect, for example integer IDs that must not accept booleans or numeric strings.

Reuse constrained text types for prose:

```python
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
```

List length and item validity are separate constraints: `Field(min_length=1)` on `list[str]` does not require each string to be nonblank. Add only the invariants the result actually needs. Empty lists can correctly represent no questions or no assumptions. Treat source code normalization separately from prose normalization; stripping whitespace may change an artifact that must be preserved exactly.

Use a model validator for relationships between fields, such as requiring feedback for a `revise` decision. Keep state-dependent checks in ordinary Python: evidence against a particular paper, selected IDs against this population, or a function against this task signature.

The decorated body runs **after** generation and parsing. Input checks there are too late to prevent a provider call. Use a constructor, a validating outer function, or `@validate_call` placed outside `@prompt`. Verify that decorator composition accepts the intended signature and blocks invalid inputs before the provider runs. Pydantic validates arguments and can coerce them; it does not validate returns by default. See the [official validation decorator documentation](https://docs.pydantic.dev/latest/concepts/validation_decorator/).

Keep genuinely textual outputs as `str`: a ReEvo reflection or a paper-specified tagged instruction need not become a JSON object. Internal immutable configuration and numerical state may remain dataclasses, tuples, arrays, or dictionaries. Types should explain a boundary, not force every object through serialization.

## 5. Keep execution resources and policy explicit

Pass exactly one of `provider=` or `session=` to a decorated call. They are execution arguments, not parameters declared on the decorated function. Store a provider on a cohesive owner if useful, then pass it explicitly at the call site.

Use a direct provider call for independent generation. Use `Session` when an operation intentionally needs conversational history or tool work. Reusing a Session carries state and can change the experiment. A Session without tools is possible; its presence alone does not prove that tools are used.

Do not share one Session between simultaneous calls: the checked implementation rejects overlapping operations. Separate sessions isolate history, but do not establish that a provider with mutable counters or usage records is concurrency-safe. Verify that resource separately.

`max_turns` bounds the Session conversation. It is not a transport retry count, a population budget, or automatic JSON repair. Record transport attempts, generated candidates, and fitness evaluations separately if their counts differ.

The caller owns retry and fallback policy. An algorithm may reject a bad candidate and retain incumbents; a transport adapter may retry transient errors within a stated deadline. Document both decisions. Moving a check into a prompt method must not silently bypass logging, change failure categories, or consume a different number of attempts. Check exception types as well as messages: `ast.parse` can raise `SyntaxError`, which a caller catching only `ValueError` will not handle.

Log raw responses at a boundary that still sees them when parsing or postprocessing fails. Logging only after `await checked_prompt(...)` loses rejected raw output. Retain audit records before acceptance, as the existing implementations already do in several paths.

## 6. Express orchestration with ordinary control flow

Keep selection, generation, scoring, acceptance, and revision visible as sequential Python steps. Short nested functions are useful when they share one run's counters or cache. Extract a collaborator when a distinct policy or lifecycle makes the enclosing function hard to follow.

Parallelize only independent work. The reference uses `gather(..., return_exceptions=True)` to let both assessments finish before raising the first encountered failure in result order. This is a deliberate failure policy, not a default for every loop. Choose and test whether siblings finish or are cancelled. Do not introduce concurrency that changes RNG consumption, population snapshots, ordered logs, or immediate beam updates during a style migration.

Use `@workflow` for an intended Slick workflow boundary. Use `Inbox` when a person or external process must make a decision. They are not mandatory decorations for any async function. A decorator alone does not establish durable recovery; persistent execution requires the appropriate `Workflow` context and stable run inputs/identity in the checked implementation.

For external review, model actions with `Literal`, require feedback for revision, and return a new draft while preserving the previous draft. Publish each revision with a fresh `inbox.request(...)` exchange. Keep channel identity distinct from shared conversation history. State termination behavior: approve returns a result, reject returns the agreed rejection value, revise repeats. The example has no total revision limit; add one only if the application's budget or lifecycle requires it.

## 7. Write precise developer prose

Use a one-sentence module summary that names the operation. Put launch instructions in the README or CLI help. Document public behavior that callers cannot infer from a signature: mutation, ownership, validation, retry policy, score direction, cache assumptions, and failure accounting.

Prefer comments that explain a choice or its limit:

```python
# Let both assessments finish before propagating either failure.
# Each generation reads the population before replacement.
```

Keep useful comments even when they make the file longer. Replace unexplained labels such as `# ponytail:` with the limitation itself and a concrete condition for revisiting it. Preserve mathematical explanations such as APEX's dual ridge formulation. Avoid narrating ordinary assignments.

Distinguish checked properties from stronger claims: a quote's presence establishes provenance, AST inspection checks syntax/interface, a finite fitness validates a measurement, and a deterministic demo exercises plumbing. None alone demonstrates improved heuristics or reproduction of a paper's results.

## 8. Apply a consistent Python format

Default to four spaces, double-quoted strings, two blank lines between top-level definitions, and grouped standard-library, third-party, and project imports. Let the formatter make routine wrapping decisions. Prefer one assignment per line when the values express distinct ownership; compact unpacking is appropriate for a meaningful pair such as `method, evaluation`.

Use modern built-in generics and `X | None` within the supported Python version. Annotate public inputs, return values, and important callbacks. Do not add a protocol only to eliminate a harmless annotation gap. Keep pure mathematical variable names where conventional notation helps understanding.

The adjacent Slick project uses Ruff, a 100-character line limit, and Python 3.10 as its target. Slick-bits adopts these formatting defaults in `ruff.toml`, with default correctness checks and import ordering; recorded runs are excluded. The Python target guides linting and formatting, not a claim that every application's dependencies support Python 3.10. Existing project configuration takes precedence. Formatting changes should be scoped and separate from prompt wording or algorithm changes.

## 9. Verify a migration at its real boundaries

Before editing, record the current render output and public call behavior. For extraction-only work, compare text across every meaningful branch. For an intentional prompt rewrite, check the new contract and report the behavioral change rather than asserting equivalence.

Use deterministic scripted providers to verify valid generation, malformed structured output, domain rejection, and pre-generation input rejection. Exercise external templates from supported working directories and installed layouts. Cover method binding and any public `.render` facade.

Run the existing tests that protect the affected algorithm: fixed budgets, invalid-candidate accounting, selection direction, elitism, tie handling, seeded randomness, snapshot semantics, mutation constraints, finite scores, held-out data separation, and evaluator cleanup. Add checks for a new behavior only where existing coverage does not establish it.

Do not replace isolated evaluators with direct execution to shorten an example. Preserve each implementation's actual execution guarantees; a subprocess is not automatically a security sandbox. Avoid paid model calls for documentation or deterministic interface checks.

## 10. A complete reference boundary

The bundled [Python example](../assets/checked_proposal.py) and [external template](../assets/prompts/heuristic/propose.j2) demonstrate constrained generated data, input validation before generation, a class with shared task context, an ordinary domain check, and explicit provider injection. Run the Python file with an environment containing Slick and Pydantic; its demo uses a canned response and never executes generated source. It intentionally strips outer whitespace from generated code, as EoH currently does; use a separate source contract when exact artifact bytes must be retained.

This example defines a new, deliberately narrow `priority(item, bins)` declaration contract. Its AST check does not resolve runtime name binding: later reassignment, deletion, or another kind of definition can still change what the name refers to. It does not establish return-value properties, safety, or fitness. It does not reproduce a paper or replace EoH's existing worker validation. Moving EoH's checks into this shape would change when failures occur, their logs, and what injected evaluators see; that requires an explicit behavioral migration.
