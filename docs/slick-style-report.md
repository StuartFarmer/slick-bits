# Comparing slick-bits with the PaperPlanner reference

Reviewed 2026-09-14. The supplied example is treated as the user's preferred reference style. “State of the art” here is a design target, not an independently established benchmark result.

This report describes the pre-migration baseline. The implementations have since been migrated to local prompt folders; see [migration validation](slick-style-validation.md#implementation-migration). The baseline inventory has been corrected from 19 to 17 prompt functions (six structured, eleven textual).

The implementations already share the reference's central philosophy: Slick generates; ordinary Python controls the algorithm. The largest opportunity is to make the generation boundary easier to read and review. External templates, explicit generated inputs, consistent response contracts, and clearer ownership would accomplish more than broad class conversions or additional workflow machinery.

The resulting [development style guide](../skills/slick-development/references/style-guide.md) is bundled with the reusable [slick-development skill](../skills/slick-development/SKILL.md). The skill includes a runnable, offline [checked proposal example](../skills/slick-development/assets/checked_proposal.py) and its [Jinja template](../skills/slick-development/assets/prompts/heuristic/propose.j2).

## Scope and evidence

The inventory covers the eight algorithm implementations listed in the root README: AEL, APEX, EoH, EvoPROMPT, LLM_GP, Optimizing the Optimizer, QUBE, and ReEvo. Scout is included as a separate workflow comparison because it uses Slick but is not an LLM optimization algorithm.

An AST inventory examined 26 top-level production Python files across those nine directories, totaling 7,243 lines, with 14 accompanying test files. It excludes `runs/`, generated candidates, captured source snapshots, virtual environments, and caches. Prompt bodies, key orchestration and validation paths, relevant tests, and README explanations were inspected. This is a style and boundary review, not a complete correctness or paper-reproduction audit.

The supplied code also exists as [the adjacent Slick paper-plan example](../../slick/examples/paper_plan.py). The review checked its actual templates, CLI setup, and supporting library source. That checkout declares Slick 0.3.0; the inspected environment resolves `slick` to that source and uses Python 3.14.0, Pydantic 2.13.4, and Jinja 3.1.6. The workspace has no Git repository metadata, so the inventory is a dated filesystem snapshot rather than a commit-pinned audit.

| Surface | Current implementations | Reference |
| --- | --- | --- |
| Prompt operations | 17 decorated functions across eight algorithms | Four decorated methods on one planner |
| Prompt storage | All 17 use inline Jinja docstrings; no `.j2` files in reviewed application sources | Separate `prompts/paper_plan/*.j2` files |
| Parsing | Six prompts specify a Pydantic `output_type`; 11 return text | Every generation method declares a structured `output_type` |
| Prompt body | All 17 contain only the prompt docstring | Explicit keyword-only `generated`, with checks or result composition |
| Execution | Call-time providers; no explicit Slick Session or Inbox usage in the reviewed production files | Provider calls for assessments, Session for plan generation/revision, Inbox for review |
| Workflow boundaries | Four `@workflow` uses in Scout; none in the eight algorithm implementations | Explicit review workflow, plus CLI orchestration in the complete example |
| Data style | Dataclasses, dictionaries, arrays, and six directly declared BaseModel classes; EoH also derives `Heuristic` from `Proposal` | Constrained Pydantic contracts throughout the displayed boundary |

These counts describe choices, not quality scores. A raw-text mutation has no inherent need for an object schema, and a batch optimizer has no inherent need for an external reviewer.

## Differences that affect readability and behavior

### 1. Model prose currently occupies the place of developer documentation

[EoH's `propose`](../eoh/evolve.py) at line 51 contains a 26-line instruction docstring; [optimizer's `propose`](../optimizer/improve.py) at line 59 contains 42 lines. A reader sees task instructions, Jinja branches, and answer formatting before seeing the next Python operation. In the reference, the decorator names the template and output schema, while the body says what happens to the answer.

This is a separation-of-audiences improvement: developers can review the Python contract while prompt authors can review the text that reaches the model. Jinja syntax is already present in all current prompts; externalization is mainly an organizational change, not a new templating technology.

ReEvo has already separated prompt declarations into [prompts.py](../reevo/prompts.py), so its module structure is close to the target. The remaining step is separating instruction files from callable definitions. Avoid simply renaming that Python module into another large collection of embedded strings.

### 2. The reference makes postprocessing part of the public operation

In `PaperPlanner.assess_method`, generation produces an `Assessment`, then `check_evidence` verifies that its quotations occur in the supplied paper. In `generate_plan`, generation produces a `Plan`, while the operation returns a `PaperPlan`. This exposes two different contracts directly in the signature and decorator.

Current applications generally parse and check later. [LLM_GP's `Operators.request`](../llm_gp/operators.py) at line 169 receives a validation callback and a fallback callback. [ReEvo's `score`](../reevo/core.py) at line 160 parses source, evaluates it, and records failure in an `Individual`. [APEX's optimizer](../apex/apex.py) at line 191 checks mutation text inside the search loop.

Putting a local acceptance check in a decorated body can reduce distance between a generated value and its meaning. However, moving an existing check changes which caller sees the error. EoH's worker currently handles code validation; moving it into `propose` could change `evaluation.error` records into generation errors and stop an injected evaluator from receiving the same inputs. An extraction-only refactor must preserve this behavior.

The target rule is therefore “make the boundary explicit,” not “move every check into a decorator.” Preserve raw-response logging when postprocessing can raise before returning to the caller.

### 3. Schema rigor varies, but much of the existing modeling is appropriate

The reference's `Text` alias strips whitespace and requires nonblank text. Its objects forbid unknown keys, list cardinalities express requirements, and `ReviewDecision` has a cross-field invariant. These make the output contract readable without searching for validation elsewhere.

[EoH's `Proposal`](../eoh/evolve.py) at line 33 already forbids extra keys and strips/rejects blank fields using a validator. Replacing it with an `Annotated` alias would primarily centralize a repeated rule; the existing implementation is not unvalidated. By contrast, [optimizer's `Proposal`](../optimizer/improve.py) at line 16 and [LLM_GP's models](../llm_gp/operators.py) at lines 16–30 do not consistently forbid extra keys or constrain every text item. LLM_GP deliberately uses strict validation for choice IDs.

AEL's tagged description plus fenced code, EvoPROMPT's `<prompt>` extraction, and ReEvo's code/prose outputs encode current experimental interfaces. Changing these to JSON would change model instructions and parsing, not just Python style. Internal frozen configuration dataclasses in AEL, APEX, and ReEvo remain a good fit.

Adopt closed models and reusable constraints for new structured boundaries. Review compatibility before tightening an existing contract. Neither `extra="forbid"` nor a return annotation establishes every form of strict validation.

### 4. Ownership is more localized in the reference, but functions are not a deficiency

`PaperPlanner` owns the paper and provider and groups related assessment, planning, and revision methods. Its `run` expresses the order of work without command-line or file-management details. The complete example places CLI and report formatting outside that class.

The current projects have different degrees of separation. [ReEvo](../reevo/core.py) already owns its task, providers, population, RNG, reflections, and run lifecycle in one class. [EvoPROMPT](../evoprompt/evoprompt.py) and [APEX](../apex/apex.py) expose algorithm functions with injected evaluators; this is useful separation. [QUBE's run module](../qube/run.py), [AEL's main module](../ael/ael.py), and [EoH's evolve module](../eoh/evolve.py) combine more of provider construction, prompt definition, execution setup, logging, and CLI work.

Move genuinely separate application plumbing toward `run.py` and preserve pure algorithm modules. Introduce a class only when its shared state or lifecycle helps explain the code. Do not wrap every existing free function in a one-method class to resemble the example.

### 5. Retry and failure policy is already unusually explicit

The example says the caller owns retries and its prompt methods do not catch generation failures. Its assessment gather waits for both outcomes before propagating a failure. Revision is a visible loop driven by typed feedback.

The current implementations already make many comparable choices. EvoPROMPT documents that provider, evaluator, and logging failures propagate without hidden retries. EoH bounds initialization and counts rejected offspring. ReEvo distinguishes aborting provider failures from candidate failures that consume evaluation shots. LLM_GP has an explicit transport retry budget, deadline, logs, and checked fallbacks. These are algorithm policies worth preserving.

A shorter method is not an improvement if it hides a fallback or changes the meaning of a budget. Likewise, copying the reference's `max_turns=20` into a direct-provider algorithm does not implement 20 attempts: the checked Slick API applies that limit to Session conversations.

### 6. Concurrency, sessions, and review solve different problems

The reference runs independent assessments concurrently using provider calls, then uses a Session for generation. Its review loop publishes a draft, waits for a typed decision, and creates a new draft on revision. A fresh review channel and a fresh Session are separate choices: supplying an existing Session makes revisions share history; omitting one creates a fresh Session for each revision.

Sequential evolutionary loops often intentionally preserve population snapshots, random draws, and immediate acceptance. APEX's immediate beam update must not be replaced by batch generation as a style cleanup. LLM_GP explicitly describes its operator adapter as supporting one sequential run.

Scout already uses workflow decorators for data retrieval and owns its interactive selection through local CLI/database behavior. Inbox is a possible feature for external review, not a missing style requirement. Similarly, `@workflow` is not a general replacement for `async def`, and decorating an operation alone does not establish durable recovery.

### 7. The prose differs more in audience and tone than in honesty

Current prompts often use role framing, uppercase sections, and paper-derived instructions: “You are an expert,” “TASK AND FUNCTION CONTRACT,” or numbered crossover and mutation steps. The reference templates begin with direct actions such as “Assess the method” and “Draft a task plan,” then explain what evidence, assumptions, and questions belong in the result.

For newly designed prompts, prefer direct verbs, sentence-case labels, short paragraphs, and explicit output fields. For a paper-faithful operator, preserve the source's role framing and steps until a separately identified prompt change is intended. Brevity is not a reason to weaken the task interface or remove provenance constraints.

Developer docstrings already include strong examples: EvoPROMPT documents snapshot semantics, cache assumptions, and failure propagation; ReEvo states evaluator ownership; APEX explains score direction and held-out separation. Preserve these even when they are longer than the reference's short methods.

Several files use `# ponytail:` to introduce limitations. The limitation is useful; the unexplained label adds vocabulary. Prefer “Sequential calls bound API load; use bounded concurrency if latency becomes a constraint.” Keep mathematical comments and explanations of deliberate constraints.

The current READMEs and demo-provider docstrings carefully distinguish canned execution, validation, actual model discovery, and reproduced results. This matches the reference's “a matching quote does not prove the claim” discipline and should become an explicit guide rule.

### 8. Mechanical formatting is a smaller gap

The implementations generally use normal Python spacing, double-quoted strings, dataclasses or Pydantic models, and explicit async calls. Type coverage varies: EvoPROMPT's public optimizer annotates callback contracts, while many EoH, QUBE, and ReEvo boundaries use untyped arguments or loose dictionaries. Some classes use dense multi-attribute assignment where separate lines would explain ownership better.

As a diagnostic comparison, Ruff was run on the ten Python modules containing all 17 prompts using the adjacent Slick project's 100-column configuration and only `E,F,I` rules. It reported four findings: import ordering in AEL and EoH, AEL's 210-character demo-source line, and one 101-character LLM_GP prompt line. This is a scoped comparison against proposed defaults, not a repository-wide lint failure under existing slick-bits policy. No shared slick-bits `pyproject.toml` was present.

Formatting is worth standardizing, but it is not the primary problem. Avoid broad formatting churn while changing prompt behavior.

## Suggested adoption by implementation

| Implementation | Preserve | Most useful next change | Behavior to protect |
| --- | --- | --- | --- |
| EoH | Five named strategies, closed Proposal model, bounded initialization, attempt logs | Extract `propose` to Jinja first; type its public inputs | Code-only mode, 5-operator schedule, worker error records, generation snapshots |
| ReEvo | Cohesive run owner, separate prompts module, explicit reflection state | Externalize three templates; consider checked source generation only with deliberate failure handling | Raw reflection strings, word bounds, evaluation budget, worse/better ordering |
| LLM_GP | Structured operators, expression grammar checks, transport accounting | Extract four templates; make validation responsibilities easier to locate | Retry/deadline counts, fallback identity, strict IDs, sequential adapter state |
| Optimizer | Complete-source context, explicit revision feedback, saved candidates | Separate propose/revise operations if their contracts diverge; tighten new schemas | Dialogue history, rejected-source retention, heuristic versus performance constraints |
| AEL | Frozen configuration, isolated evaluation options, explicit interface parser | Externalize prompt; separate application/evaluator plumbing where useful | Tagged response protocol, exact reference tours, genetic selection and RNG order |
| QUBE | Search module, clustered selection, bounded evaluator cleanup | Move prompt declaration out of CLI plumbing and externalize its task branches | Island/cluster logic, task interfaces, failed-sample budget, output bounds |
| APEX | Immutable document edits, pure LinUCB logic, injected evaluator | Externalize mutation and prediction templates; isolate mutation acceptance logic if it clarifies the loop | Exact text formatting, immediate beam updates, history reversal, finite scores |
| EvoPROMPT | Typed public optimizer, frozen generations, explicit tagged-text extraction | Externalize GA, DE, variation, and answer templates | Ordered operator steps, single tagged answer, tie rules, held-out separation |
| Scout | Workflow-based retrieval and existing human selection | Apply prose/format guidance when touching relevant code | HTTP concurrency, cancellation, database state, local selection semantics |

These are future migration suggestions. This delivery changes documentation and adds the skill example; it does not refactor the algorithms.

## What to adopt from the reference, and what to qualify

Adopt external Jinja templates, explicit `output_type`, keyword-only generated values for meaningful postprocessing, constrained response contracts, constructor or outer input validation, direct domain names, and short comments explaining ownership or failure behavior.

Qualify the following details before copying the code:

- The displayed imports are used by the full example's CLI and renderer; the excerpt alone should not be judged for unused `sys`, `Path`, `Prompt`, or `Workflow` imports.
- The snippet omits template-root setup. The full [CLI helper](../../slick/examples/_cli.py) sets it at line 32. Copying only the class can produce working-directory-dependent template lookup failures.
- `check_evidence` verifies exact substring provenance, not correctness, completeness, or relevance of the assessment. It does not establish paper quality.
- `revise` constructs a new wrapper but shares the assessment objects. It avoids mutating the draft in that method; it does not make the nested Pydantic objects deeply immutable.
- `extra="forbid"` rejects additional fields but does not enable global strict coercion. Empty question and assumption lists are intentional valid states.
- The review loop has no total revision count or overall deadline. That is a product lifecycle choice to revisit where necessary.
- Explicit `provider` typing and a fully typed gather result could be improved if supported tooling needs them. The example is a design reference, not a reason to reproduce every annotation omission.

API details were confirmed in [Slick's prompt implementation](../../slick/slick/prompts.py), [postprocessing tests](../../slick/tests/test_prompt_postprocessing.py), [Session implementation](../../slick/slick/session.py), and [Inbox implementation](../../slick/slick/inbox.py). Particularly consequential details are `instance` binding, the process-global template root, explicit owner binding for method `.render`, exactly one execution resource, output parsing independent of the return annotation, and `None`/`Ellipsis` passthrough from prompt bodies.

The supporting guide also checked the official [Pydantic validation-decorator documentation](https://docs.pydantic.dev/latest/concepts/validation_decorator/) and [Jinja API documentation](https://jinja.palletsprojects.com/en/stable/api/). Local Slick source is the authority for Slick-specific behavior in this review.

## Delivery validation

The comparison inventory and scoped Ruff diagnostic above were executed. The accompanying skill and example receive structural, deterministic runtime, and independent application checks; their results are recorded in [the validation note](slick-style-validation.md). Existing algorithm tests were inspected for migration invariants, but the full algorithm suites and paid model experiments are outside this documentation change.
