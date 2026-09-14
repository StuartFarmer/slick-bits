# Mutation Without Variation / LMCA

Problem-agnostic implementation of the supplied paper, **Mutation Without
Variation: Convergence Dynamics in LLM-Driven Program Evolution** (Gurkan,
Stonedahl, Wilensky), based on its [official LMCA repository](https://github.com/can-gurkan/lmca).
The supplied APET URL concerns a different paper; that method lives in
[`../apet`](../apet/README.md).

This algorithm studies a mutation operator: every valid candidate becomes the
next parent, including duplicates and self-loops. There is no fitness selection,
novelty rejection, best-so-far solution, or early stopping on recurrence.
It runs 300 accepted mutations by default and includes the seed as state zero.

## Run on your own representation

Use the repository's `optimizer/.venv/bin/python`, which already has Slick and
NetworkX. For a separate environment, install from this directory with
`python -m pip install -r requirements.txt`. Configure templates once at startup:

```python
from pathlib import Path

from slick import prompts
import lmca

prompts.TEMPLATE_ROOT = Path(lmca.__file__).resolve().parent / "prompts"


async def experiment(task, constraints, seed, provider, validate, skeleton, fallback=None):
    agent = lmca.LMCA(
        task, provider, validate,
        constraints=constraints,
        instruction="Generate a program that includes exploratory modifications.",
        fallback=fallback,
    )
    result = await agent.run(seed, steps=300, retries=5)
    report = lmca.analyze(result.programs, skeleton=skeleton)
    return result, report, agent.calls
```

`task` describes the problem; `constraints` describes the allowed representation,
syntax, primitives, types, and any validity requirements. Inputs can be programs,
expressions, plans, or other text. The neutral experiment does not reward solving
the task or preserving behavior.

Supply an async `validate(candidate: str) -> Validation` function:

```python
async def validate(candidate):
    # Your parser/type checker; no execution is required by this algorithm.
    program, error = parse_and_check(candidate)
    if error:
        return lmca.Validation(None, error)
    return lmca.Validation(render_canonically(program))
```

`parse_and_check` and `render_canonically` above are your domain functions.
Return `None` in `Validation.program` only for an invalid representation, with
a useful repair reason. Return canonical text on success; that text becomes
the next parent and defines exact identity. Exceptions propagate as operational
failures and do not consume additional validation attempts. Blank generated text
is rejected before the validator. Supply a valid, canonical initial seed.

The structural abstraction is a separate synchronous `skeleton(program) -> str`
callback used only by analysis. For the paper's DSL, replace action leaves with
`ACTION`, directions with `DIR`, predicates with `PRED`, and boolean operators
with `BOOL`, preserving control-flow names, arity, and tree structure. For another
language, define its abstraction using its own parser. Structural equivalence
does not imply semantic equivalence.

Each step starts with a fresh mutation prompt on the primary provider. A rejected
candidate triggers a separate repair prompt containing that candidate, its failure
reason, and constraints. **Five retries means six attempts per model**, matching
the official code. After primary exhaustion, the optional fallback starts by
repairing the primary's last invalid candidate, then gets the same retry allowance.
The next accepted step starts on the primary again. With fallback the worst case
is `2 * (retries + 1)` calls per step. With no fallback, only the primary stage runs.

Exhausting validation attempts returns `stop_reason="validation_exhausted"` and
the accepted prefix; no rejected candidate or artificial repeated parent is added.
Reaching the step budget returns `"budget"`. `Result.calls` counts model calls,
while `len(Result.programs) - 1` counts accepted mutations. `steps=0` returns the seed.

Model selection, temperature, token limits, reasoning settings, network retries,
and persistence belong to the caller. The paper generally used temperature 1.0,
1024 output tokens, and five retries; some model conditions differed. No provider
is constructed here. No Session/history, gridworld, or candidate execution is used.
The caller owns isolation if their validator executes anything.

## Analyze convergence

`analyze(programs, skeleton=..., tokenize=...)` returns three entries:

- `programs` and `skeletons`: visit counts, directed transition counts (tuple edge
  keys), cumulative unique counts, revisit fraction, simple-cycle length counts,
  mean degree entropy, and mean successor entropy.
- `successive_distances`: normalized token Levenshtein distances for each mutation.

The seed contributes to unique counts. Revisit fraction is the number of mutations
landing on a previously visited state divided by the number of mutations; it is
zero for empty/single-state chains. All cycle lengths are included, even self-loops.
Parallel transitions contribute to edge frequencies but do not duplicate node cycles.
A cycle in the accumulated graph is not proof of an indefinitely periodic trajectory.

There is a source/prose discrepancy worth preserving explicitly:

- `mean_degree_entropy` matches upstream `graph_analysis.py`: total multigraph
  degrees are normalized across nodes, then `-p * log2(p)` contributions are averaged
  over nodes. This is the metric used for its graph table.
- `mean_successor_entropy` describes successor diversity as discussed in the paper:
  compute each node's entropy over observed outgoing transition frequencies, then
  average across all nodes, assigning zero to sinks. This is an additional metric,
  not a renamed version of upstream's degree metric or its separate weighted
  natural-log transition entropy.

Edit distance divides the number of insertions/deletions/substitutions by the larger
token count; two empty sequences have distance zero. The default tokenizer recognizes
Unicode words and individual punctuation. Supply your language tokenizer explicitly
for comparable experiments. Upstream's DSL tokenizer is
`re.findall(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+|[(),]", text)`.

`pairwise_distances(programs, tokenize=...)` returns a symmetric matrix with zero
diagonal for the paper's heatmaps. It is separate from routine analysis because
pairwise storage is quadratic. Exact simple-cycle enumeration can be exponential
in graph size; analysis does not silently truncate cycles or declare convergence.
Analyze a shorter window when a large graph makes complete enumeration impractical.

## Classical subtree baseline

`lmca.gp` supplies immutable `Primitive` and `Tree` types plus a uniform-node
subtree mutator for arbitrary strongly typed grammars:

```python
import random
from lmca.gp import Primitive, Tree, subtree_mutate

word = Primitive("word", "text")
join = Primitive("join", "text", ("text", "text"))
grammar = (word, join)
tree = Tree(join, (Tree(word), Tree(word)))
rng = random.Random(7)
programs = [tree.render()]
for _ in range(300):
    tree = subtree_mutate(tree, grammar, rng=rng, max_depth=4)
    programs.append(tree.render())
```

The mutator selects any node uniformly, including the root, and replaces it with
a random subtree of the same return type. Root depth is zero; the remaining depth
allowance is `max_depth - selected_node_depth`. Primitive choice is uniform among
those that can be completed within that allowance. The original tree is unchanged.
Supply a well-typed seed within the depth bound and a grammar with feasible typed
replacements. Unchanged mutations are accepted. Analyze the rendered chain with
the same `analyze` function and your grammar's skeleton abstraction.

This preserves the baseline operator's decisions but deliberately replaces the
official gridworld-specific subtree distributions with a generic typed grammar;
it does not reproduce its exact RNG stream or published baseline counts.

## Source mapping and verification

Official source inspected at commit
[`a5e84de6ab98860a75946a303a9a0d8d2883cfb2`](https://github.com/can-gurkan/lmca/commit/a5e84de6ab98860a75946a303a9a0d8d2883cfb2):

| Source under that commit | Used here |
| --- | --- |
| [`src/llm_gp/evolution/llm/operators.py`](https://github.com/can-gurkan/lmca/blob/a5e84de6ab98860a75946a303a9a0d8d2883cfb2/src/llm_gp/evolution/llm/operators.py) | Mutation → validation repairs → fallback repairs, with `retries + 1` attempts per stage |
| [`src/lmca/chains/runner.py`](https://github.com/can-gurkan/lmca/blob/a5e84de6ab98860a75946a303a9a0d8d2883cfb2/src/lmca/chains/runner.py) | Seed plus sequential accepted mutations under the paper's neutral, one-candidate setting |
| [`src/lmca/config/prompts`](https://github.com/can-gurkan/lmca/tree/a5e84de6ab98860a75946a303a9a0d8d2883cfb2/src/lmca/config/prompts) | Separate mutation and repair prompt layouts and caller-selected instruction line |
| [`src/lmca/analysis/graph_analysis.py`](https://github.com/can-gurkan/lmca/blob/a5e84de6ab98860a75946a303a9a0d8d2883cfb2/src/lmca/analysis/graph_analysis.py) | Multigraph transitions, NetworkX simple cycles, degree entropy definition |
| [`src/lmca/analysis/dynamics.py`](https://github.com/can-gurkan/lmca/blob/a5e84de6ab98860a75946a303a9a0d8d2883cfb2/src/lmca/analysis/dynamics.py) | Token edit-distance normalization and representation-level trajectory analysis |
| [`src/llm_gp/evolution/operators.py`](https://github.com/can-gurkan/lmca/blob/a5e84de6ab98860a75946a303a9a0d8d2883cfb2/src/llm_gp/evolution/operators.py) | Random-node, typed subtree replacement with a remaining-depth allowance |

Prompts intentionally replace gridworld wording with task/representation inputs.
The validator owns canonicalization and any code-fence extraction; no heuristic
stripping of program text is performed. The fixed DSL, 50-condition sweep runner,
behavioral evaluation, plots, embeddings, API clients, and checkpoint infrastructure
are outside this algorithm port. The four representative instruction lines in
Table 3 can be supplied directly as `instruction=`; use separate instances to
compare prompts, models, seeds, and replications.

`agent.calls` retains raw responses before validation, prompts, step/stage/attempt,
acceptance, rejection reason, and operational errors. `agent.programs` retains the
accepted prefix even when an exception propagates. Runs reset both lists. Imports
leave Slick's process-global template root untouched; configure it at startup and
use separate processes for algorithms needing different roots concurrently.

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_lmca
../slick/.venv/bin/ruff check lmca tests/test_lmca.py
../slick/.venv/bin/ruff format --check lmca tests/test_lmca.py
```

The shared scripted-provider tests cover recurrence, retry/fallback ordering and
budgets, canonicalization, errors, alternate launch directories, known graph
metrics, distances, and baseline typing/depth/reproducibility. They are offline
algorithm checks, not evidence of LLM convergence or reproduction of paper results.
