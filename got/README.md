# Graph of Thoughts

Problem-agnostic Slick implementation of [Graph of Thoughts: Solving Elaborate
Problems with Large Language Models](https://arxiv.org/abs/2308.09687), following
the paper's operation graph and the [official implementation](https://github.com/spcl/graph-of-thoughts).
Thoughts are text artifacts; they can represent documents, plans, partial answers,
code, or any other content. The caller defines the task and what makes a result good.

## Use

Use the existing `optimizer/.venv` environment, or install the adjacent Slick
checkout with `python -m pip install -e ../slick` from the repository root.
No additional dependencies are required. Configure templates once at startup:

```python
from pathlib import Path

from slick import prompts
import got

prompts.TEMPLATE_ROOT = Path(got.__file__).resolve().parent / "prompts"


async def solve(task, input_text, provider, evaluate=None):
    agent = got.GraphOfThoughts(task, provider, evaluate)
    result = await agent.run(input_text, max_calls=100)
    return result.final[0].content
```

`task` describes the goal, constraints and output format; `input_text` supplies
the material to work on. The caller configures a Slick provider's model, sampling,
credentials and transport policy. Generated content is never executed by the agent.
Imports do not modify Slick's process-global template root. Configure an absolute
root before calls; simultaneous algorithms needing different roots require separate
processes. Use one run at a time per instance and a provider without implicit chat
history. Calls are sequential and independent, with no shared Session.

The default plan generates five candidates, scores them, keeps three, aggregates
them into three new candidates, and selects the best original or aggregate. It
then generates two independent improvements and keeps the best improvement or
incumbent. That is **10 generation calls plus 10 scoring calls**, or just 10 provider
calls with an injected evaluator. It is a small general-purpose schedule, not a
claim that one graph is optimal for every task.

## Evaluation and validity

An optional async `evaluate(thought, graph) -> float` replaces LLM scoring. `graph`
maps thought IDs to immutable `Thought` values, so an evaluator can inspect parents,
partial results, or the complete reasoning state. Capture task data or external
evaluation resources in your callback. `thought.content` is the candidate text;
`thought.kind` identifies its user-defined class, such as `part` or `solution`.
Higher finite scores are preferred by default; use `higher_is_better=False` for
error or cost minimization. Measured nonfinite scores raise instead of entering ranking.
The snapshot uses the incoming branch's annotations for the evaluated thought;
other nodes carry their latest global annotations at callback time.

An optional async `validate(thought, graph) -> got.Validation` replaces LLM validity
checks. Return, for example, `got.Validation(valid=False, feedback="Missing a required item")`.
Validation is separate from scoring and is used only by `validate_and_improve`.
A failed validity check is an ordinary algorithm result; malformed responses,
transport errors and callback exceptions abort the run.

Without callbacks, the local score/validation templates use the same provider.
`criteria=` sets scoring criteria. LLM scores request a consistent 0–10 scale;
the parsed score must be finite. A model validity judgment is not verified ground
truth. With `Operation("score", ..., count=3)`, three model scores are averaged;
an injected evaluator runs once per thought regardless of `count`, as upstream does.

## Custom operation graphs

Pass a dictionary of named `Operation` values to `run(..., plan=plan)`. Dependencies
can refer to any other named step; dictionary order need not be topological.
The standard library's `TopologicalSorter` schedules each step once after its
predecessors. Cycles raise `CycleError`; missing predecessors raise `KeyError`.

| Operation | Behavior | Meaning of `count` |
| --- | --- | --- |
| `generate` | Independently expand each incoming thought | Calls per input thought |
| `decompose` | Produce complementary text parts in one structured response per input | Parts per input thought |
| `aggregate` | Combine all incoming thoughts into each new result | Independent aggregate calls |
| `improve` | Independently refine each incoming thought | Calls per input thought |
| `score` | Score each thought using the callback or model | Model samples per thought |
| `keep_best` | Stable score ranking, preserving the first item on ties | Maximum retained thoughts |
| `validate_and_improve` | Validate, repair only invalid thoughts, validate again | Maximum repairs per thought |
| `keep_valid` | Keep only thoughts explicitly marked valid | Unused |
| `select` | Route existing thoughts with `selector(inputs)` | Unused |

`instruction=` provides operation-specific requirements or examples. `thought_kind=`
labels new artifacts; scoring, validation, and selection preserve existing labels.
`select` receives an immutable tuple and must return a sequence of existing thoughts.
It can route any subset into subsequent branches. It does not create new artifacts.

For example, decompose a task, explore alternatives within each part, and merge the
best partial solutions (the structure used by the paper's decomposable tasks):

```python
from got import Operation

plan = {
    "split": Operation("decompose", count=2, thought_kind="part"),
}
for index in range(2):
    part, solve, score, best = [f"{name}_{index}" for name in ("part", "solve", "score", "best")]
    plan[part] = Operation(
        "select", ("split",), selector=lambda thoughts, i=index: thoughts[i:i + 1]
    )
    plan[solve] = Operation(
        "generate", (part,), count=3, instruction="Solve only the supplied subproblem."
    )
    plan[score] = Operation("score", (solve,))
    plan[best] = Operation("keep_best", (score,))
plan["merge"] = Operation("aggregate", ("best_0", "best_1"), count=3)
plan["merge_score"] = Operation("score", ("merge",))
plan["final"] = Operation("keep_best", ("merge_score",))

# Inside your async application, after configuring the template root:
result = await agent.run(input_text, plan=plan)
```

For repeated feedback, add `Operation("validate_and_improve", ("final",), count=3)`.
It makes at most three repairs and four validation checks, including a check of the
last repair. It returns the last thought even if still invalid; append `keep_valid`
to discard invalid results. `count=0` performs validation without repair. Generated
revisions start unscored and unvalidated; add `score` before ranking them.

To preserve an incumbent, give `keep_best` both the earlier scored step and the
new scored step as predecessors. Selection prunes the active branch; earlier
thoughts remain addressable through earlier operations. Empty predecessor outputs
stay empty through generation/aggregation, rather than restarting from the input.
A step with no predecessors starts from the original input. An empty plan returns
no final thoughts.

## State and failures

`Result.final` is the union of terminal operation outputs. A custom plan can return
several thoughts or none; use an explicit final `keep_best` when you need one winner.
`Result.outputs` holds immutable per-operation snapshots, including score/validity
annotations. `Result.thoughts` holds all generated artifacts and each artifact's
most recent annotations. Rescoring a shared node does not mutate earlier operation
snapshots. At a join, duplicate IDs contribute once and the first predecessor's
annotations win; put the intended scored/validated branch first.

Each thought records its direct generating parents. ID 0 is caller input and is
included as a parent because the generic templates always supply it. Refinement
adds a new version connected to its predecessor, as in upstream's improvement
state history. Thus feedback is unrolled into bounded steps, not a cycle in the
operation scheduler. `result.volume(thought.id)` counts distinct preceding generated
thoughts, excluding the caller input and the thought itself. Scoring and selection
do not inflate this metric with bookkeeping copies.

`agent.calls` preserves rendered prompts, raw responses, and errors before Slick
parsing/postprocessing. `agent.assessments` records scoring and validation outcomes;
`agent.execution` records completed or failed operations. Generation requires
nonblank content and preserves whitespace. Decomposition validates exact cardinality
and nonblank items; structured response objects reject unknown fields.

`max_calls` is a hard cap checked before every prompt invocation, including model
scoring and validation. Exceeding it raises `CallBudgetExceeded`. Other errors also
propagate without retries or fallback results. Partial thoughts, completed operation
outputs and failure records remain on the agent. Every new `run()` resets state;
there is no implicit resume. Provider-internal retries, tokens, cost limits, local
evaluation limits, persistence, and any execution sandbox belong to the application.

## Official source and scope

The supplied [APET link](https://github.com/daankepel/APET) concerns another algorithm
and already has its own `apet/` folder here. This implementation uses the GoT paper's
official [SPCL repository](https://github.com/spcl/graph-of-thoughts), inspected on
2026-09-14 with `main` at [`3d9d9db`](https://github.com/spcl/graph-of-thoughts/commit/3d9d9dbd8937d47a4441f681b8b40e3c5b054f16).
Its algorithm is re-expressed in Slick; the upstream package is not a runtime dependency.

| Official code used | Corresponding implementation |
| --- | --- |
| [controller.py](https://github.com/spcl/graph-of-thoughts/blob/3d9d9dbd8937d47a4441f681b8b40e3c5b054f16/graph_of_thoughts/controller/controller.py) | Dependency-ready execution, per-operation results, terminal outputs |
| [graph_of_operations.py](https://github.com/spcl/graph-of-thoughts/blob/3d9d9dbd8937d47a4441f681b8b40e3c5b054f16/graph_of_thoughts/operations/graph_of_operations.py) | Static graph separating plan from execution state |
| [operations.py](https://github.com/spcl/graph-of-thoughts/blob/3d9d9dbd8937d47a4441f681b8b40e3c5b054f16/graph_of_thoughts/operations/operations.py) | Per-parent generation, all-parent aggregation, local/model scoring, stable ranking, bounded validation/improvement, selectors |
| [thought.py](https://github.com/spcl/graph-of-thoughts/blob/3d9d9dbd8937d47a4441f681b8b40e3c5b054f16/graph_of_thoughts/operations/thought.py) | Separate content, score and validity state; immutable snapshots here |
| [doc_merge.py:got](https://github.com/spcl/graph-of-thoughts/blob/3d9d9dbd8937d47a4441f681b8b40e3c5b054f16/examples/doc_merge/doc_merge.py#L505) | Default generate/select/aggregate/improve topology, including incumbent edges at both selection stages |

The default uses smaller sample counts than the document-merging experiment.
Prompts are deliberately rewritten for arbitrary tasks, with separate generation,
decomposition, aggregation, improvement, scoring and validation templates. The
paper's task-specific scoring functions, few-shot examples, ground-truth benchmarks,
combined model scoring, provider backends and experiment infrastructure are not
bundled. The upstream license and citation are retained in [UPSTREAM_LICENSE](UPSTREAM_LICENSE).

The static GoO permits arbitrary branching and joins; bounded refinement supplies
recurrence. This does not implement arbitrary in-place cyclic graph editing or
physical removal of archived thoughts. Full input is included in every generic
prompt, so task decomposition does not automatically reproduce the paper's token
or latency savings. Specialize the local templates and evaluator for that experiment.

## Checks

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_got
../slick/.venv/bin/ruff check got tests/test_got.py
../slick/.venv/bin/ruff format --check got tests/test_got.py
```

Tests use the repository's shared scripted provider and the real Slick 0.3.0
decorator. They check graph decisions, raw failure records, score direction,
refinement limits, volume, and template rendering from another working directory.
These are offline algorithm/interface checks, not reproduced benchmark results or
evidence of improvement on an arbitrary task.
