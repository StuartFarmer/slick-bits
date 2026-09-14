# Slick style report and skill validation

Checked 2026-09-14 against the local files described in the [report](slick-style-report.md). The shipped skill is a reference for the requested development conventions, not an enforced repository-wide policy or a claim that existing algorithms were migrated.

## Evidence collected

| Check | Result | Scope |
| --- | --- | --- |
| Python AST inventory | 26 production files, 7,243 lines; 17 prompt functions, six structured outputs, no external templates or generated parameters | Nine application directories; excludes tests, runs, snapshots, environments |
| Existing formatting diagnostic | Four findings: two import-order and two line-length findings | Ten prompt-containing modules; proposed adjacent Slick `E,F,I` configuration |
| Skill frontmatter validator | Passed | Name, description, supported frontmatter, scaffold checks |
| Bundled Python example | Passed with canned response | Renders external Jinja, parses Proposal, checks declared interface, prints result; no model call or generated-source execution |
| Bundled Python Ruff check | Passed | Full adjacent Slick Ruff configuration |
| Focused behavioral exercise | Eight cases passed | Rendering without provider calls; valid typed result; malformed JSON; blank field; unknown field; domain rejection; blank feedback rejected before calling provider; blank constructor input |
| Template location exercise | Passed | Rendered the bundled file from an unrelated temporary working directory using an absolute configured root |
| Existing Slick API tests | 20 passed | `test_prompt_postprocessing.py` and `test_prompt_simplification.py`; no cache writes requested |
| Document links | 45 local links resolved; fenced code blocks balanced | README, report, validation note, skill entrypoint, guide |
| Personal installation | Validated; all four bundled files match the workspace byte for byte | `/Users/stuart/.codex/skills/slick-development` |

The API-test process emitted two interpreter-prefix warnings because the interpreter path contained `slick-bits/../slick`; the tests completed successfully. No full algorithm regression suite, Docker evaluator suite, paid provider call, optimization benchmark, or paper replication was run. The application implementations were not edited.

The focused example exercise verified actual provider call counts and returned values/errors. It also checked explicit owner binding for `HeuristicDesigner.propose.render`, the presence of the task and feedback in the rendered template, and exactly one output-format block. It restored the process-global template root and working directory after running.

## Independent skill application

Before the skill was authored, an independent agent received a realistic EoH migration task without the new guidance. It inspected the code and correctly identified most relevant API and compatibility constraints. Its proposal included:

> I would not add AST validation to the method merely to give its body a check.

It also identified explicit owner binding for a method's `.render`, the global template-root issue, and the need to preserve existing evaluation error records. It kept ReEvo reflections textual and did not introduce Inbox into Scout.

This baseline did **not** demonstrate a failing behavior. The skill was still created because the user explicitly requested reusable development guidance; its purpose is to capture the preferred conventions and verified API details. No claim is made that the skill experimentally improved agent performance. Repeated wording micro-benchmarks were not run for this reference artifact.

A separate fresh-context agent then received the same migration request with the completed skill and the real source. It preserved rendering/calling compatibility, validation ownership, attempt accounting, session policy, and the two counterexamples, and reported no blocking skill error. Findings from that check are incorporated into the reference. In particular, the example now states that checking an AST function declaration does not prove which object a name will refer to at runtime, the guide calls out `SyntaxError` propagation, and source whitespace normalization is explicitly described.

## Repeating the deterministic checks

From the slick-bits workspace, using the inspected adjacent environment:

```sh
rtk proxy ../slick/.venv/bin/python /Users/stuart/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/slick-development
rtk proxy ../slick/.venv/bin/python -B skills/slick-development/assets/checked_proposal.py
rtk proxy ../slick/.venv/bin/ruff check skills/slick-development/assets/checked_proposal.py --config ../slick/pyproject.toml
rtk proxy env PYTHONDONTWRITEBYTECODE=1 ../slick/.venv/bin/python -m pytest ../slick/tests/test_prompt_postprocessing.py ../slick/tests/test_prompt_simplification.py -q -p no:cacheprovider
```

The eight-case exercise was executed as an ephemeral Python script, not added as an application test suite. For future migrations, use the guide's verification criteria and the affected application's existing tests.

## Implementation migration

Completed the subsequent implementation request with three subagents owning separate algorithm groups. Each implementation now owns its own template directory, as explicitly requested:

| Implementation | Local templates |
| --- | --- |
| AEL | `ael/prompts/propose.j2` |
| APEX | `apex/prompts/{mutate,predict}.j2` |
| EoH | `eoh/prompts/propose.j2` |
| EvoPROMPT | `evoprompt/prompts/{ga_offspring,de_offspring,variation,answer}.j2` |
| LLM_GP | `llm_gp/prompts/{initialize,mutate,crossover,choose}.j2` |
| Optimizer | `optimizer/prompts/propose.j2` |
| QUBE | `qube/prompts/generate.j2` |
| ReEvo | `reevo/prompts/{generate,reflect_pair,reflect_long}.j2` |

All 17 prompt bodies now contain developer-facing docstrings and explicit keyword-only generated results. Their public call signatures, six structured output schemas, eleven text outputs, and validation/failure boundaries are preserved. Each CLI sets its absolute local template root once at startup. Programmatic setup is documented in each README; production module imports do not change Slick's global root. No central prompt directory was created.

Ruff now provides consistent 100-column formatting, import groups, and default correctness checks through the root `ruff.toml`. Scout received only formatting/import changes; its nine Python ASTs and import sets were verified unchanged. Existing `ponytail:` limitation markers were retained under the active development mode. Future LLM_GP source snapshots include the four template files needed to reproduce generation.

Final verification:

- All 17 template files exactly match their original cleaned docstrings, plus a file-ending newline. The earlier 19-prompt inventory was an arithmetic error, now corrected in the report.
- 193 rendered cases match the pre-edit snapshot byte for byte, including branches, parents, demonstration variations, and output schemas: 130 EoH/AEL/Optimizer, 13 APEX/EvoPROMPT, 50 ReEvo/QUBE/LLM_GP.
- Public prompt signatures are unchanged. Production algorithm function/class ASTs are unchanged outside prompt boundaries, CLI startup, and LLM_GP's required template snapshot inclusion.
- All 14 offline test commands passed in the final integration run: eight algorithm suites and Scout's six test scripts. EoH ran 12 tests; QUBE ran 10 with its two existing opt-in Docker checks skipped. Other suites use executable assertion checks.
- Offline CLI launches from unrelated temporary working directories passed for every migrated implementation. LLM_GP's new snapshot template contents were checked.
- `ruff check .` passed; `ruff format . --check` reported 41 Python files already formatted. Saved runs are excluded from formatting and were not migrated.

No dependencies were installed, no paid model calls were made, and no paper-performance claims are inferred from these checks. Existing environments supplied the tests: `ael/.venv` for AEL, `evoprompt/.venv` for EvoPROMPT, `llm_gp/.venv` for LLM_GP, `scout/.venv` for Scout, and `optimizer/.venv` for the other algorithms.
