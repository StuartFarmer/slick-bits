# LLM genetic improvement

Slick implementation of **Enhancing Genetic Improvement Mutations Using Large
Language Models** (Brownlee et al., 2023). One optimizer supports independent
random sampling and best-first local search, with SIMPLE, MEDIUM, and DETAILED
LLM mutation prompts. Callers supply the artifact, task, provider, and evaluator.
There is no Java, JCodec, benchmark, or code execution dependency in the agent.

## Use

Configure Slick's template root once at application startup:

```python
import ast
from pathlib import Path

import llm_gi
from slick import prompts
from llm_gi import CandidateRejected, Evaluation, GeneticImprovement

prompts.TEMPLATE_ROOT = Path(llm_gi.__file__).resolve().parent / "prompts"

def check_python(content: str) -> None:
    try:
        ast.parse(content)
    except SyntaxError as exc:
        raise CandidateRejected(str(exc)) from exc

async def optimize(initial: str, provider, evaluate):
    # evaluate: async (complete_content: str) -> Evaluation
    agent = GeneticImprovement(
        task="Reduce runtime while preserving the existing behavior.",
        provider=provider,
        evaluate=evaluate,
        language="python",
        context="A pure function called repeatedly by the application.",
        requirements="Preserve the function signature. Return the complete source.",
        validate=check_python,
        fingerprint=lambda text: ast.dump(ast.parse(text), include_attributes=False),
    )
    best = await agent.run(initial, mode="local", operator="MEDIUM", budget=100)
    return best, agent.attempts
```

`evaluate` measures the **complete artifact**, returning, for example,
`Evaluation(fitness=12.5)` for passing content, `Evaluation(compiled=False,
passed=False)` for a compile failure, or `Evaluation(passed=False)` for a
correctness failure. Lower finite fitness wins; negate a reward to maximize it.
The evaluator owns correctness tests, execution isolation, deadlines, warm-up,
and measurement units. The example's AST check only checks syntax. It does not
execute or establish correctness. For noncompiled artifacts, leave `compiled=True`.

For a different problem, change the task, context, requirements, validation, and
evaluation. The default language/fence label is `text`; default validation only
rejects blank content, and default fingerprints compare exact strings.

The default target is the whole artifact. To mutate only eligible regions, supply
`targets(content) -> Sequence[Target]`, where each `Target(start, end)` uses Python
string offsets into that current content. The agent selects one uniformly and
preserves everything outside it. Supply hot-method block spans for code tasks;
the evaluator still receives the full source. Targets are recomputed from the
current parent each attempt. `validate(full_content)` should raise
`CandidateRejected` for expected syntax or structural failures. Convert parser
exceptions at this boundary; unexpected errors propagate.

Choose `operator="DETAILED"` with `example_before` and `example_after` to give
every mutation the same useful-change example. `SIMPLE` intentionally omits task,
context, examples, and output-format instructions, preserving the paper's weak
prompt condition. Provider model, temperature, timeouts, and transport retries
belong to the caller; the experiment used GPT-3.5 Turbo at temperature 0.7.

## Algorithm and accounting

- Local search defaults to **100 attempt slots: one original evaluation and 99
  mutations**. It accepts only passing candidates with strictly lower fitness.
  Failed, worse, and tied proposals retain the incumbent. A failing baseline
  aborts before generation.
- Random sampling defaults to **1000 mutations of the original**, with no
  baseline evaluation. It returns the lowest-scoring passing candidate, or
  `None` when none pass. Its best candidate is never used as the next parent;
  without measuring a baseline it does not claim an improvement over the original.
- Each LLM mutation requests five suggestions in one independent provider call.
  It scans fences with the configured language label and uses the **first
  replacement that passes validation**. Only that candidate is evaluated; a
  subsequent compile/test failure does not cause another suggestion to be tried.
  Five suggestions are requested, but response cardinality is not enforced.
- No valid suggestion, no eligible target, expected candidate rejection, timeout,
  or nonfinite measured fitness consumes its slot. There are no hidden generation
  retries or model repairs. Provider failures and unexpected callback errors
  are recorded and propagate.
- Content equivalent to the current parent or original is recorded as `no_op`
  without evaluation. `fingerprint` defines equivalence. Duplicate non-no-op
  candidates are **measured again**, allowing repeated timing measurements.
- `classic={"STATEMENT": mutate, "INSERT": insert}` enables the paper's standard
  comparison categories. Each callback takes `(parent_content, rng)` and returns
  a complete changed artifact; select its key with `operator=`. The callback owns
  syntax-aware copy/delete/replace/swap or insertion semantics and can use Gin.
  There are no built-in language-specific classic operators. One category is
  selected per run, matching the experiments; no automatic hybrid is introduced.

`run()` returns an immutable `Candidate(content, fitness)`. `agent.baseline` and
`agent.best` retain the measured original and best candidate. `agent.attempts`
records the baseline and every mutation, including parent, selected target,
raw response before validation, invalid-suggestion reasons, candidate content,
fingerprint/uniqueness, evaluation result, failure status, and acceptance.
`improvement` is baseline fitness minus measured fitness in local runs; it can
be positive even when a candidate fails to beat the current best.

`agent.evaluations` counts actual evaluator calls, including failed calls. Invalid
generations and no-ops explain why this can be smaller than the slot budget.
Uniqueness is tracked for validated candidates, not malformed raw responses;
these records are not directly the paper's all/unique patch tables.

Calls are sequential, with exactly one `provider=` per prompt and no persistent
Session. Each run resets records and its seeded Python RNG. Use one active run
per instance. Slick's template root is process-global: configure it before use,
and use separate processes for applications needing different roots concurrently.

## Official sources and deliberate adaptations

The [official artifact](https://doi.org/10.5281/zenodo.8304433) identifies
[Gin commit `9fe9bdf3ad6115fa26bebbe258f31b4507bac884`](https://github.com/gintool/gin/tree/9fe9bdf3ad6115fa26bebbe258f31b4507bac884).
This implementation uses the following source algorithms and prompt artifacts:

- [LLMReplaceStatement.java](https://github.com/gintool/gin/blob/9fe9bdf3ad6115fa26bebbe258f31b4507bac884/src/main/java/gin/edit/llm/LLMReplaceStatement.java):
  uniform block selection, five requested alternatives, labeled fences, and the
  first usable replacement. Although the paper says "first code block", this
  source skips unparseable suggestions before selecting the first usable one.
- [LocalSearchSimple.java](https://github.com/gintool/gin/blob/9fe9bdf3ad6115fa26bebbe258f31b4507bac884/src/main/java/gin/util/LocalSearchSimple.java)
  and [LocalSearchRuntime.java](https://github.com/gintool/gin/blob/9fe9bdf3ad6115fa26bebbe258f31b4507bac884/src/main/java/gin/util/LocalSearchRuntime.java):
  baseline-inclusive budget, one added mutation per iteration, strict
  best-so-far acceptance, and passing-test runtime minimization. The separate
  `gin.LocalSearch` CLI's add/remove-edit neighborhood is not used here.
- [RandomSampler.java](https://github.com/gintool/gin/blob/9fe9bdf3ad6115fa26bebbe258f31b4507bac884/src/main/java/gin/util/RandomSampler.java):
  independently mutate the original rather than extending previously sampled edits.
- The artifact's `Simple-LLM-prompt.txt`, `Medium-LLM-prompt.txt`, and
  `Detailed-LLM-prompt.txt` define the three prompt conditions. Gin's MIT notice
  is retained in [LICENSE.gin](LICENSE.gin).

This is an intentional problem-agnostic adaptation. JavaParser becomes a caller
validator; Java/project/body instructions become language, task, and requirements;
the fixed JCodec example becomes caller-supplied text. The response stays textual
with labeled fences rather than JSON. Fence headers must end with a newline;
replacement text, including its trailing newline, is preserved.

Accepted artifacts are frozen snapshots. Gin stores edit objects and re-applies
them from the original, potentially issuing new LLM requests for old edits; this
port does not regenerate accepted replacements. No-ops skip evaluation here.
Fitness is not truncated to integer milliseconds, and Python's RNG does not
reproduce Gin's Java random stream or per-patch seed schedule. Profiling, selection
among projects/methods, ten repeated experiments, and test timeouts belong in the
caller harness. No Java runtime or official source checkout is required at runtime.

## Checks

From the repository root, with the existing Slick environment:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_llm_gi
../slick/.venv/bin/ruff check llm_gi tests/test_llm_gi.py
```

The shared scripted provider checks search decisions, budgets, selection of the
first valid suggestion, failures, target boundaries, duplicates, classic callbacks,
repeatability, and external templates. These are offline algorithm/interface
checks, not a reproduction of the paper's runtime improvements or LLM success rates.
