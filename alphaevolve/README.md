# AlphaEvolve

Problem-agnostic implementation of the evolutionary coding algorithm in
[Novikov et al., AlphaEvolve (2025)](https://arxiv.org/abs/2506.13131).
Give it initial source, a task description, a Slick provider, and an async
evaluator. It returns an evaluated candidate and retains the search history.
Source is text: the optimizer does not assume Python or any particular problem.

## Official sources and implementation choices

The official [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results)
contains mathematical constructions and verification code. Its README explicitly
says it does not contain the code to run AlphaEvolve. Those problem-specific
verifiers can be used in an external evaluator, but there is no official agent
implementation to import. This implementation is not a reproduction of Google's
internal system or of its reported scientific results.

The paper cites FunSearch as its predecessor. This implementation adapts the
seed-all-islands and weaker-half reseeding policies from its
[official program database](https://github.com/google-deepmind/funsearch/blob/main/implementation/programs_database.py).
The Python representation, objective archive, sampling, and reset clock differ;
see [NOTICE](NOTICE) and [Apache license](LICENSE.funsearch).
No third-party AlphaEvolve clone is used.

| Paper mechanism | Implementation |
| --- | --- |
| Parent, inspirations, feedback, LLM proposal, evaluation, registration | `sample()` → `mutate()` / `rewrite()` → `_evaluate()` → `_register()` |
| Whole-file and marked-block evolution | Exact text edits; all regions may change together; unmarked files are wholly editable |
| LLM ensemble | Caller-supplied providers sampled by weights; no fixed model names |
| Rich and stochastic prompts | Task, context, measured feedback, inspirations, recent failures, weighted instruction variants |
| Multiple scores | Each final metric is maximized; retain a champion per metric and diversity cell |
| MAP-Elites and islands | Independent dictionaries keyed by `(cell, metric)`; evaluator supplies discrete cells |
| Exploration/exploitation | Uniform archive sampling with probability `exploration`, otherwise a randomly chosen metric's champion |
| Island exchange | Weaker half reseeded from random surviving islands on a completed-attempt interval |
| Evaluation cascade | Early stages must meet all thresholds before the final evaluator runs |
| Meta prompt evolution | Optional extra text call; guidance is selected using average positive offspring improvement |
| Asynchronous pipeline | A bounded worker pool immediately registers completed children without generation-wide barriers |

Section 2.5 does not specify the exact archive descriptors, selection distributions,
replacement rules, or reset schedule. The policies above are explicit local
choices. Default `cell=()` gives one cell per island: supply discrete cells for
diversity within each objective. Each cell retains one champion **per objective**,
not a full Pareto frontier. Ties retain incumbents. All successfully evaluated
programs remain available in `programs`, even when they do not win an archive slot.

## Use with your own problem

Install the adjacent Slick checkout using `pip install -r requirements.txt` from
this directory, or use an environment already containing that checkout. The API
was checked against the adjacent `slick-ai` 0.3.0 source. No new dependencies are
needed beyond Slick.

From an application that can import this repository:

```python
from pathlib import Path

import alphaevolve
from alphaevolve import AlphaEvolve, Config, Evaluation, EvaluationStage
from slick import prompts

# Set once at application startup, before any concurrent prompt calls.
prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).resolve().parent / "prompts"


async def optimize(initial_source, provider, evaluate):
    agent = AlphaEvolve(
        task="Improve the supplied algorithm while preserving its interface.",
        provider=provider,
        evaluate=evaluate,
        context="Describe your interface, constraints, and scoring criteria here.",
        config=Config(meta_interval=10),
    )
    best = await agent.run(initial_source, attempts=100, concurrency=4, seed=7)
    return best, agent
```

`evaluate(source: str) -> Evaluation` is **async**. Its final result supplies all
objectives together, for example:

```python
Evaluation(
    metrics={"quality": 0.93, "negative_runtime": -0.012},
    feedback="Passed the full correctness suite; runtime measured over 100 trials.",
    cell=(2, 5),  # Your discretized behavior descriptors, not objective scores.
)
```

These numbers illustrate the return contract; the caller must actually measure
them. Negate losses, costs, or durations to minimize them. All final candidates
must have the same nonempty set of finite metric names as the seed. Pass
`target_metric="quality"` to `run()` to choose the returned winner; otherwise the
first metric returned by the seed evaluator is used. This also ranks islands at
reset time. `best_by_metric` retains the best ever for every metric independently.

The evaluator owns code execution, compilation, isolation, test data, resources,
timeouts, and cleanup. The agent never imports, executes, or writes candidate
code. Protected text boundaries preserve the source skeleton; they are not a
security sandbox. Put any generated-code execution behind your own isolated
runner, and keep the trusted verifier outside the evolving source.

To carry constructions between successive search heuristics, include their
serialized results or artifact references in `feedback`; they then accompany
the parent in subsequent prompts. The evaluator owns artifact storage.

## Editing contract

Use the markers in comments appropriate to your language:

```python
# Stable imports and interface scaffolding can go outside the block.
# EVOLVE-BLOCK-START
def solve(inputs):
    return initial_solution(inputs)
# EVOLVE-BLOCK-END
```

Marker lines and everything outside them stay exactly unchanged. Multiple blocks
are supported; nested or unbalanced markers fail. Marker tokens are reserved
lexical delimiters, so do not use them as data inside source strings. Without
markers the entire source is editable.

Default `mode="diff"` uses the paper's `<<<<<<< SEARCH`, `=====`,
`>>>>>>> REPLACE` format. Searches must be nonempty and unique, including
overlapping occurrences, and contained within one editable region. Edits apply
sequentially and fail as a whole if any block fails. There is no fuzzy matching.
`mode="rewrite"` requests the whole file as plain text and checks the same
immutable skeleton. Blank and unchanged outputs are rejected. Syntax and
functional correctness are established by the evaluator, not the text parser.

## Optional controls

- **Model ensemble:** pass `ensemble=((fast_provider, 0.8), (strong_provider, 0.2))`.
  This replaces the default single `provider`; model selection is recorded as
  the index in this tuple. Providers must support concurrent calls when enabled.
- **Cascade:** pass `stages=(EvaluationStage(smoke_test, {"valid": 1.0}),)`.
  Each async stage receives the same candidate source. Only the final evaluator's
  metrics become optimization objectives. Earlier metrics gate progression;
  feedback from all successful stages is retained. A stage may delegate work to
  a cluster or use an LLM for qualitative scoring without changing the agent.
- **Stochastic guidance:** pass weighted `prompt_variants`, such as
  `(("Try a different representation.", 1), ("Simplify the algorithm.", 1))`.
  Selection is in Python; Jinja only renders supplied values.
- **Meta evolution:** `Config(meta_interval=10)` creates an extra instruction
  every tenth claimed attempt and immediately tries it on that attempt's parent.
  Instruction reward is positive improvement in the sampled objective divided
  by `max(1, abs(parent_score))`. Failed or non-improving uses receive zero.
  Later instructions are sampled randomly with probability `exploration`, otherwise
  by highest average reward. This separate growing instruction archive is a
  minimal local interpretation of the paper's unspecified co-evolution procedure.
  The default `meta_interval=0` disables its extra model calls.
- **Timeouts:** `generation_timeout` applies to each model call and
  `evaluation_timeout` to each evaluation stage. Both default to the caller's
  timeout policy. Async cancellation is cooperative: external workers must clean
  up their own processes or remote jobs.

## Budgets, failures, and results

Use a fresh agent per run. The seed is evaluated once through the cascade and
registered into every island without a model call. Seed failures propagate.
There are at most `concurrency` in-flight candidate attempts; completed children
are available immediately to subsequent samples. Pending children for a reset
island are admitted to its current archive when they finish.

`attempts=N` spends exactly N candidate attempts on a normal run, including
rejected proposals. `generation_calls`, `meta_calls`, and `evaluations` count
calls separately; `evaluations` counts individual cascade stages, including the
seed and failed stages. Meta calls do not reduce the candidate budget. A failed
optional meta call falls back to an existing instruction. The agent does not retry
provider calls; any internal transport retries are owned by the supplied provider.

Return `Evaluation(error="reason")` or raise `InvalidCandidate` for expected
invalid programs. `ProviderError` and timeouts also reject an attempt. Unexpected
exceptions propagate after sibling workers are cancelled and awaited. Evaluator
programming errors are not silently converted into bad fitness. Failure records
retain raw model text before edit validation, rejected meta text, error reasons,
completed stage results, parent IDs, model IDs, and status.

The return value is a `Candidate` with `content`, `metrics`, `feedback`, `cell`,
`id`, and `parent_id`. Inspect `agent.attempts`, `programs`, `islands`,
`best_by_metric`, `prompt_ideas`, and reset `events` for diagnostics or external
persistence. History is in memory; this module does not implement a durable
database, CLI, cluster scheduler, or benchmark suite.

RNG seeds reproduce selection only when completion order and provider responses
are reproducible. Use `concurrency=1` for deterministic offline experiments.
Slick's template root is process-global; configure it once and avoid concurrently
switching roots to run unrelated agents in the same process.

## Verification

From the repository root, in the Slick environment:

```sh
python -m unittest tests.test_alphaevolve
```

Checks use the shared scripted provider and external deterministic evaluators.
They verify search mechanics, prompts, counters, failures, and immutable edit
boundaries. They do not spend model credits, run generated programs, or establish
the optimization performance or scientific discoveries reported in the paper.
