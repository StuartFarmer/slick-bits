# LLM mutations in genetic improvement (2025)

Problem-agnostic Slick implementation of Brownlee et al., **Large language model
based mutations in genetic improvement** (2025). This is separate from
[`llm_gi`](../llm_gi/README.md), which implements the preliminary conference work.
The small `Target`, `Evaluation`, and `CandidateRejected` contracts are reused
from that package; the journal search and its three prompts live here.

```python
from pathlib import Path
from slick import prompts
import llm_gi_2025
from llm_gi_2025 import Evaluation, GeneticImprovement

prompts.TEMPLATE_ROOT = Path(llm_gi_2025.__file__).resolve().parent / "prompts"

async def optimize(initial, provider, evaluate):
    agent = GeneticImprovement(
        task="Your task and constraints",
        provider=provider,
        evaluate=evaluate,
        language="text",
    )
    winners = await agent.run(initial, budget=100)
    return winners["artifact"], agent.history

# Supply your own async evaluate(complete_content, method_id) -> Evaluation:
# Evaluation(fitness=measured_cost)                    # passing, lower is better
# Evaluation(compiled=False, passed=False)            # compilation failure
# Evaluation(passed=False, detail="behavior changed") # correctness failure
```

The default target is the whole artifact. `Candidate.content` is the resulting
text, `fitness` is the measured cost, and `patch` is an immutable tuple of
`Edit(target, replacement)` objects. Negate rewards to maximize them. The agent
never executes generated content. Your evaluator owns datasets, compilation,
tests, isolated execution, deadlines, warm-up, and measurement units. Provider
model, temperature, and transport retries are also caller-owned.

## Targeting arbitrary artifacts

For code, documents, or structured text, supply
`targets(content) -> {method_id: {block_id: Target(start, end)}}`. Spans are Python
string offsets into the supplied content. Method IDs group independent search
regions; block IDs identify replaceable fragments. The callback must keep IDs
stable as replacements change text lengths and recalculate current spans.
It must include all eligible blocks, including nested blocks when relevant.

Local search independently restarts from the original artifact for every method
returned by `targets(initial)`. Supply the top ten profiled methods to reproduce
the paper's outer loop. Each evaluation receives the full artifact plus the
selected method ID so you can run its associated tests. Returned winners are
independent variants; they are not automatically combined.

Random sampling selects a method uniformly, then one of its current blocks
uniformly. Every one-edit patch is generated from the original. Unequal block
counts therefore do not bias method selection. Use `mode="random", budget=1000`.
Unsampled methods, or methods with no passing candidates, have `None` winners.

`validate(complete_content)` checks syntax and the required interface. Raise
`CandidateRejected` for expected invalid generated content; unexpected exceptions
propagate. Default validation only rejects blank content. It establishes neither
syntactic validity for a programming language nor semantic correctness.

## Search and prompt choices

- Local search uses **100 slots per method**, including one baseline evaluation
  and 99 neighbors. It accepts only compiling, test-passing candidates with
  finite fitness strictly below the incumbent. Ties and failures keep the parent.
  A failed baseline aborts the run.
- `neighborhood="paper"` implements Algorithm 1: for a nonempty patch, remove
  one uniformly selected edit with probability 0.5; otherwise add an edit.
  Empty patches always add. Removal replays the remaining edits from the original.
  `neighborhood="artifact"` uses the released experiment runner's add-only search.
- A new LLM edit requests five alternatives and takes the **first parseable
  replacement**, then evaluates it once. A compile/test failure does not cause
  fallback to a later suggestion. The raw response and rejected suggestions are
  recorded. Formatting stays textual, with labeled or unlabeled code fences.
- `prompt_style="BASIC"` is the default. `SMALL_CHANGES` requires
  `examples={"SMALL_CHANGES": [before, copied, deleted, replaced, swapped]}`.
  `STRUCTURAL_CHANGES` requires
  `examples={"STRUCTURAL_CHANGES": [before, alternate_repetition, bulk_transform,
  alternate_traversal]}`. Examples remain fixed throughout a run. Supply examples
  suited to your artifact; the official Java examples are linked below.
- `prompt_style="STATEMENT"` uses a caller-supplied
  `classic(fragment, rng) -> replacement` callback instead of an LLM. The callback
  can perform syntax-aware copy/delete/replace/swap inside the selected block.
  It uses the same search and evaluation loop. No textual approximation of Java
  AST edits is included.

Invalid generations, empty block sets, timeouts, evaluator `CandidateRejected`,
and nonfinite scores consume slots. Unexpected provider/callback errors are
logged and propagate. There are no internal retries or repair calls. Duplicates
and unchanged content are **evaluated again**, matching sampling rather than
introducing a deduplication filter. A patch whose target disappears during replay
is rejected. Use a stable-ID parser to avoid mapping an old edit onto another block.

`agent.history` includes baselines and attempted neighbors, their method, parent,
operation, patch, content, raw generation, validation failures, evaluation result,
status, and acceptance where available. `unique` compares exact frozen patches
per method, not semantic equivalence. `agent.evaluations` counts evaluator calls;
`agent.optimizer_calls` counts LLM call attempts, including failed calls.
`agent.baselines` and `agent.best` retain measured candidates by method.

Each `run()` resets state and its seeded RNG. Use one active run per instance.
All generation uses a direct `provider=` call with no conversational session.
Configure Slick's process-global template root once at application startup;
separate processes are needed for concurrent applications with different roots.

## Official implementation used and discrepancies

The [journal replication artifact](https://doi.org/10.5281/zenodo.13381774)
identifies [Gin commit `f2f6e1018229cf2d8d3220bf1482635af5657806`](https://github.com/gintool/gin/tree/f2f6e1018229cf2d8d3220bf1482635af5657806).
The supplied paper cites `9fe9bdf`, which is the older conference implementation
and lacks the journal's external prompt templates. Both revisions were inspected;
the journal artifact revision supplies this implementation's source reference.

| Official source at the journal revision | Behavior used here |
| --- | --- |
| [LLMReplaceStatement.java](https://github.com/gintool/gin/blob/f2f6e1018229cf2d8d3220bf1482635af5657806/src/main/java/gin/edit/llm/LLMReplaceStatement.java) | Uniform block choice, five alternatives, first parseable replacement |
| [LocalSearch.java](https://github.com/gintool/gin/blob/f2f6e1018229cf2d8d3220bf1482635af5657806/src/main/java/gin/LocalSearch.java) | Add/remove neighborhood and strict passing-runtime acceptance |
| [LocalSearchSimple.java](https://github.com/gintool/gin/blob/f2f6e1018229cf2d8d3220bf1482635af5657806/src/main/java/gin/util/LocalSearchSimple.java) | Per-method baseline-inclusive budgets and optional add-only search |
| [RandomSampler.java](https://github.com/gintool/gin/blob/f2f6e1018229cf2d8d3220bf1482635af5657806/src/main/java/gin/util/RandomSampler.java) | Independent patches and uniform method sampling |
| [BASIC template](https://github.com/gintool/gin/blob/f2f6e1018229cf2d8d3220bf1482635af5657806/examples/llm/base-template-prompt.txt) | Context and replacement-format instructions |
| [SMALL CHANGES template](https://github.com/gintool/gin/blob/f2f6e1018229cf2d8d3220bf1482635af5657806/examples/llm/replicating-statement-prompt.txt) | Ordered copy/delete/replace/swap examples |
| [STRUCTURAL CHANGES template](https://github.com/gintool/gin/blob/f2f6e1018229cf2d8d3220bf1482635af5657806/examples/llm/prompt-with-basic-examples.txt) | Ordered repetition/bulk/traversal examples |

This port deliberately generalizes Java/project/body wording and fixed Java
examples. BASIC and SMALL CHANGES include caller requirements; STRUCTURAL CHANGES
preserves the source's omission of explicit formatting/requirements instructions.
Its task context and examples must therefore carry any needed constraints.
All three use the same response parser. The paper describes unlabeled fences as
accepted; the released regex actually requires `java`. This port accepts both the
configured label and unlabeled fences, requires a newline after the header, and
preserves replacement text verbatim. These are adapted prompts, not exact replicas.

Edits freeze generated replacements so removal does not regenerate earlier LLM
outputs. Gin can call the LLM again when replaying edit objects. Gin also permits
edits to deleted nodes to do nothing; this port rejects missing targets. Python
RNG sequences, callback-defined validity, classic-operator distributions, and
floating-point fitness differ from the Java experiment. No Java runtime is needed;
the search logic and prompt structures are ported from the official sources.
Gin's MIT notice is retained in [LICENSE.gin](LICENSE.gin).

## Verification

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_llm_gi_2025
../slick/.venv/bin/ruff check llm_gi_2025 tests/test_llm_gi_2025.py
```

Offline tests use the repository's shared scripted provider. They check search,
patch removal/replay, first-valid selection, budgets, failures, independent hot
methods, classic callbacks, and all templates from another launch directory.
They do not reproduce benchmark runtime improvements or model success rates.
