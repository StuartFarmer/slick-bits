# AdaEvolve

Optimize arbitrary textual artifacts using adaptive exploration, island scheduling,
and model-generated solution tactics. Supply the task, initial candidate, a Slick
provider, and an async evaluator; no benchmark or execution environment is built in.

```python
from pathlib import Path

import adaevolve
from adaevolve import AdaEvolve, Evaluation
from slick import prompts

# Configure once at application startup, independently of the working directory.
prompts.TEMPLATE_ROOT = Path(adaevolve.__file__).resolve().parent / "prompts"

# Your async callback: evaluate(content: str) -> Evaluation.
# Return Evaluation(score=measured_score, feedback=diagnostics) for a valid candidate,
# or Evaluation(error="reason the candidate is invalid") for rejection.
agent = AdaEvolve(
    task="Your objective, candidate format, constraints, and available resources",
    provider=provider,
    evaluate=evaluate,
    evaluator_context="Evaluator source or a precise description of its scoring rules",
)
best = await agent.run(initial_candidate, iterations=100, max_calls=120, seed=42)
print(best.content, best.score)
```

`provider` and `evaluate` are caller-supplied objects. Candidate text can be code,
prompts, plans, or any other artifact your evaluator understands. The agent never
executes it. The evaluator owns correctness checks, execution isolation, timeouts,
and any held-out data boundary. `evaluator_context` is optional but gives tactics
the information needed to reason about scoring. Keep held-out data out of it.
Higher scores win by default; pass `maximize=False` for costs or losses. Scores
returned to callers retain their original sign. Zero and negative scores use the
absolute-denominator normalization from Algorithm 6.
The equations use Python floats without clipping: extreme score ratios can
overflow when squared (for example, a zero baseline followed by `1e150`). Use
numerically sensible score units; finite-score validation alone does not guarantee
that every intermediate arithmetic operation is representable.

Each `run` is intended for a fresh agent. Calls are sequential, use independent
provider contexts, and create no Session. An optional `meta_provider` uses a
different model for tactics; otherwise both operations use `provider`. Slick's
template root is process-global: applications using different template roots
concurrently need separate processes. No root changes occur during import.

## Search behavior

The initial candidate is evaluated once and shared by two initially unvisited
islands. Each mutation selects an island with decayed-reward UCB, samples an
exploration/exploitation mode using its accumulated improvement signal, generates
one candidate, and evaluates it once if generated parsing succeeds. All valid
candidates enter the selected archive, including lower-scoring attempts. The best
candidate is retained; ties favor the incumbent.

Exploration samples uniformly and selects inspirations with the greatest text
distance from the parent. Exploitation samples uniformly from the top quartile
(rounded up), using highest-scoring inspirations. Each prompt receives the task,
evaluator specification, parent, inspirations, and available evaluation feedback.

Ring migration copies each island's best from a snapshot every 15 iterations.
It updates a receiving island's signal and local best without UCB credit, visits,
reevaluation, or a multi-hop cascade. Stagnation can spawn an island seeded from a
uniformly selected unique archived candidate, with fresh adaptive statistics.

Meta-guidance analyzes the global best, evaluator, the last ten mutation attempts
(including invalid outputs), and prior tactic outcomes. It generates one to three
structured tactics with the official `idea`, `description`, `what_to_optimize`,
`cautions`, and `approach_type` fields. Guided prompts apply them round-robin for
five attempts each, then request a fresh batch if stagnation persists. Each tactic
tracks its own uses and directly observed global improvement. Text distance is a
lexical proxy; it does not establish semantic or behavioral diversity.

`Config` exposes experimental controls without per-task tuning requirements:

| Setting | Default |
| --- | --- |
| Initial islands / maximum islands | 2 / 8 |
| Decay / epsilon | 0.9 / 1e-8 |
| Exploration range | 0.1–0.7 |
| Spawn / meta signal thresholds | 0.02 / 0.12 |
| Migration interval / spawn cooldown | 15 / 15 iterations |
| Meta warmup | 15 iterations |
| Uses per tactic / inspirations | 5 / 3 |

Set `migration_interval=0` to disable migration. Spawning requires that existing
islands have each received a mutation attempt. Warmup and cooldown avoid treating
the initialized zero signals as immediate evidence for repeated interventions.

## Budgets and failures

`iterations` bounds mutation attempts. Tactic generation uses additional calls.
`max_calls`, when supplied, caps both mutation and tactic calls together; it does
not include the one seed evaluation. A tactic call requires room for a subsequent
mutation and is skipped after the final mutation. An exhausted call budget can
therefore end a run before its iteration budget. No algorithm-level retries occur.
Provider-level transport retries, if configured by the caller, are outside these
logical call counts.

Malformed JSON, invalid output schemas, blank generated content, and duplicate
ideas within a tactic batch are recorded and rejected. Candidate whitespace is
otherwise preserved. Explicit evaluator errors, nonfinite measured scores, and
evaluator `TimeoutError` reject the candidate. Rejected mutations still decay the
selected island's signal/reward, increment its visits, and consume any active
tactic use. Invalid initial candidates abort. Provider failures and unexpected
evaluator exceptions are recorded and re-raised. Cancellation propagates.

`attempts` contains raw model responses captured before parsing, candidate content,
parent/island IDs, selected intensity, scores, feedback, and errors. `events`
records spawning and migration; `islands` exposes archives and adaptive statistics;
`tactic_history` retains tactic outcomes. `seed_evaluation` records initialization.
`mutation_calls`, `meta_calls`, and `evaluations` count attempted operations.
All state stays in memory; archives and logs grow with the requested budget.
Persistence, checkpointing, distributed workers, and benchmark runners are external.

## Paper and official implementation

Based on Cemri et al., *AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization*,
especially Algorithms 1–6 and Appendix A.2 supplied with this implementation.
The paper identifies [SkyDiscover](https://github.com/skydiscover-ai/skydiscover)
as its official implementation. Source inspected and adapted at commit
[`0d932b690670a7e544388ad362e9876c8bd256a0`](https://github.com/skydiscover-ai/skydiscover/tree/0d932b690670a7e544388ad362e9876c8bd256a0):

- [`adaptation.py`](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/adaevolve/adaptation.py): local/global normalization, decayed UCB statistics, migration credit separation.
- [`archive/diversity.py`](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/adaevolve/archive/diversity.py): ported text-mode token Jaccard and length distance, with weights 0.7 and 0.3.
- [`paradigm/tracker.py`](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/adaevolve/paradigm/tracker.py) and [`generator.py`](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/adaevolve/paradigm/generator.py): adapted rotation/use limits, tactic fields, and evaluator-aware analysis.
- [`database.py`](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/adaevolve/database.py): migration and bounded spawning reference.

This is a focused Slick port, not a wrapper around the full framework. Where the
current official implementation differs, the paper controls the core algorithm:

| Choice | This implementation | Inspected official code |
| --- | --- | --- |
| Improvement signal | Unclipped Algorithm 6; decay on every selected attempt | Clips normalized gains to 1; `record_evaluation` only updates G on improvement |
| Stagnation trigger | All island G values versus paper thresholds | Productivity and windowed binary improvement rates |
| Island warmup | One visit, per paper | Three visits before UCB |
| Parent sampling | Uniform exploration; top-quartile exploitation (Algorithm 3) | Novelty-weighted exploration and archive elite selection |
| Spawning | Random archive seed, 15-step cooldown | Top-program seeds, configurable presets, default 50-step cooldown |
| Archive | Full in-memory history | Bounded novelty/fitness archive with eviction |
| Tactics | Individual outcomes; failed mutations consume uses | Batch-level outcomes; use tracking tied to successful additions |

Appendix Algorithm 3 resolves the paper's prose ambiguity about fitness-proportional
exploitation. Prompt wording has deliberately been generalized from programs to
arbitrary artifacts, and candidate output uses explicit JSON rather than code
diffs. This changes the experiment; no equivalence to the published benchmark
scores is claimed. The warmup, cooldown, migration frequency, maximum islands,
and memory policy are explicit implementation choices, not inferred paper results.
Apache-2.0 attribution is retained in [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Offline verification

Use an environment with this repository's local Slick dependency installed:
`python -m pip install -e ../slick` from the repository root, or
`python -m pip install -r requirements.txt` from this directory. No new dependency
beyond Slick and its existing Pydantic/Jinja dependencies is required.

From the repository root:

```sh
python -m unittest tests.test_adaevolve
```

The tests use the shared scripted provider and external deterministic evaluators.
They check controller mathematics, state transitions, generated contracts, failure
accounting, prompt rendering, and budgets. They do not measure live model quality
or reproduce the paper's optimization results.
