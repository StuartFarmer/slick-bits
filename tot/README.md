# Tree of Thoughts

Problem-agnostic implementation of the search algorithms in *Tree of Thoughts:
Deliberate Problem Solving with Large Language Models* (Yao et al., 2023).
`TreeOfThoughts` owns generation, evaluation, and search. You supply the task,
thought granularity, answer requirements, and a Slick provider.

```python
from pathlib import Path

from slick import prompts
import tot

# Application startup: resolve independently of the working directory.
prompts.TEMPLATE_ROOT = Path(tot.__file__).resolve().parent / "prompts"

async def solve(task: str, provider):
    agent = tot.TreeOfThoughts(
        task=task,
        provider=provider,
        thought="One concise planning step; preserve previous decisions",
        criteria="Satisfies the constraints and leads toward a feasible answer",
        answer_format="Return the requested deliverable",
    )
    result = await agent.run(
        search="bfs",
        depth=3,
        breadth=5,
        candidates=5,
        generation="propose",
        evaluation="value",
        evaluation_samples=3,
        max_expansions=100,
    )
    return result

# result.solutions[0].answer is available when search reaches a terminal state.
# Always handle an empty result.solutions and inspect result.budget_exhausted.
```

Use the existing local environment, or install the adjacent Slick checkout with
`python -m pip install -e ../slick` from the repository root. No additional
dependencies are required. Template configuration is process-global in the checked
Slick API; configure it once at application startup. Imports leave it unchanged.
Different template roots in simultaneous applications require separate processes.

## Algorithm

| Choice | Behavior |
| --- | --- |
| `generation="sample"` | Makes `candidates` independent calls with identical context |
| `generation="propose"` | Makes one call proposing up to `candidates` sibling alternatives |
| `evaluation="value"` | Scores each state independently on [0, 1], averaging `evaluation_samples` calls |
| `evaluation="vote"` | Compares candidates jointly; the fraction of votes received is the score |
| `search="bfs"` | Expands the whole frontier and retains the globally highest scoring `breadth` states |
| `search="dfs"` | Visits siblings in descending score order, prunes scores `<= threshold`, and backtracks |

A state is the task (stored on the agent) plus an immutable ordered tuple of
thought strings. Each generation appends one thought; it does not replace earlier
thoughts. Identical siblings are deduplicated in first-occurrence order. Stable
score ties preserve candidate order. A proposal may return an empty list to signal
a dead end. Sample calls require a nonblank thought.

`depth` counts thought expansions; the root has depth zero. At that depth a
separate `finish` call produces the final artifact. BFS finalizes only its best
leaf (Algorithm 1). DFS records every surviving leaf reached within the budget,
in traversal order (Algorithm 2); it does not stop at its first output. Terminal
nodes are not expanded again. `depth=0` makes one direct finalization call.
`breadth` only affects BFS, and `threshold` only affects DFS.

LM values are heuristics, not correctness checks. A terminal state means the
configured depth was reached, not that the problem was solved. `solutions`
contains generated final answers, without an independent correctness claim.

## Custom evaluation and task adaptation

An optional async callback replaces LM scoring, including voting:

```python
async def evaluate(state: tot.State) -> float:
    # score_partial_solution is your own async domain evaluation function.
    return await score_partial_solution(state.thoughts)

agent = tot.TreeOfThoughts(task, provider, evaluate=evaluate)
result = await agent.run(search="dfs", depth=6, threshold=0.2)
```

The callback runs once per candidate and must return a finite score; higher is
better. Choose a threshold on its scale. It receives the entire partial solution,
not just the last thought. `evaluation` and `evaluation_samples` are unused with
this callback. Callers own any execution isolation, external retrieval, datasets,
and checks of the final answer. The agent never executes generated content.

Set `thought` to an equation, edit, plan paragraph, assignment, or other useful
unit. It may describe different requirements for successive step numbers. Put
constraints and any examples in `task`, evaluation guidance in `criteria`, and
final output requirements in `answer_format`. Model choice and sampling
temperature belong to the provider. Independent sampling needs a provider setup
that permits diversity.

## Budgets, records, and failures

`max_expansions` bounds expanded parent states, not provider calls or tokens. BFS
stops before a level if its entire frontier cannot fit in the remaining budget;
it never selects a beam from a partially expanded level. DFS stops when its next
nonterminal state cannot be expanded. Dead ends and pruned subtrees backtrack
normally. Generation errors still consume their attempted expansion.

An expansion costs one proposal call or `candidates` sample calls. LM value
evaluation costs `evaluation_samples` calls per distinct child; voting costs
`evaluation_samples` calls per candidate group. BFS votes over the whole expanded
level, DFS over siblings. Finalization costs one call per returned solution. All
calls are sequential, independent, and have no conversational Session.

`Result` contains:

- `solutions`: tuple of `Solution(state, answer)` values; possibly empty.
- `best_state`: BFS's best retained state, or DFS's first deepest visited state.
  This is a partial-state fallback, not an oracle choice or a verified solution.
- `expansions`: number of expanded parents.
- `budget_exhausted`: whether the budget prevented further exploration.

`agent.history` stores retained BFS frontiers or retained DFS sibling groups.
`agent.calls` records each prompt operation, rendered prompt, raw response, and
any error, including JSON/schema/postprocessing failures. `agent.assessments`
records custom evaluator attempts, scores, and errors. Each run resets these
records. An instance supports one run at a time.

Malformed generations, out-of-range votes, nonfinite scores, evaluator errors,
and provider errors propagate immediately. No retries, repairs, or silent
fallbacks occur. Inspect the records after an exception; retry policy and durable
logging belong to the caller. Caller configuration is trusted rather than
preflight-validated.

## Relationship to the paper and checks

This implements the paper's generic BFS/DFS algorithms, both generation modes,
and both evaluation modes. Generic prompts and typed JSON for proposals, values,
and votes deliberately replace the benchmark-specific prompts. Numerical values
use [0, 1]; DFS orders siblings by those values. Voting with DFS is an extension
using sibling vote shares. Text sampling and final answers remain raw prose.

Thought depth is fixed. The crossword-specific board renderer, variable stopping
rules, confidence aggregation, and benchmark datasets are outside this generic
implementation. There is no A*, MCTS, model training, or reproduction claim for
the paper's reported success rates.

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_tot
../slick/.venv/bin/ruff check tot tests/test_tot.py
```

Offline tests use the shared `tests.providers.ScriptedProvider` through real
Slick rendering and parsing. They check beam selection, pruning/backtracking,
independent sampling, repeated evaluations, budgets, malformed outputs, raw
failure records, and templates rendered from another working directory. They do
not measure model problem-solving performance.
