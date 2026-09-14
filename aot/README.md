# Algorithm of Thoughts

Problem-agnostic Slick implementation of [Algorithm of Thoughts: Enhancing
Exploration of Ideas in Large Language Models](https://arxiv.org/abs/2308.10379).
The model explores, evaluates, and backtracks **inside one generation**. Python
does not maintain a search tree or make a separate model call for each node.

## Use

Use the existing `optimizer/.venv` environment or install the adjacent Slick
checkout from the repository root with `python -m pip install -e ../slick`.
No additional dependencies are required. Configure the template root once at
application startup:

```python
from pathlib import Path

from slick import prompts
import aot

prompts.TEMPLATE_ROOT = Path(aot.__file__).resolve().parent / "prompts"


async def solve(task, input, provider):
    agent = aot.AoT(task, provider)
    result = await agent.run(input)
    return result.output
```

`task` supplies the goal, allowed actions, constraints, and desired artifact format.
`input` supplies the instance: a specification, document, question, or other text.
The configured Slick provider owns the model, credentials, token limit, decoding,
and transport retries. The paper uses temperature 0 for Game of 24; choose decoding
settings supported by your provider. This agent does not execute generated content.

## Search variants and examples

| `strategy` | Behavior inside one response | Paper reference |
| --- | --- | --- |
| `"plans"` (default) | Propose plans, assess each, choose, produce, refine | Appendix B and F.4; Section 5 QA |
| `"dfs"` | Explore a promising subtree, prune failures, backtrack, reconstruct the solution | Section 3 and Appendix F.1 |
| `"bfs"` | Explore promising candidates level by level, preserving parent paths | Appendix F.1.3 |

The default is the paper's zero-shot plan approach, generalized beyond creative
writing. Set `plans=3` on the constructor for the three-strategy QA variant;
the default is five plans, as in creative writing. The plan count is a prompt
instruction, not a Python-enforced search budget.

For the central few-shot DFS/BFS method, supply **problem, search process, solution**
examples from your task family, including failed branches and recovery. For example,
this illustrative graph task demonstrates how to abandon an attractive dead end:

```python
examples = [aot.Example(
    input="Find a directed route S to G. Edges: S-A, S-B, A-C, B-D, D-G.",
    search=(
        "1. Try S-A; extend to S-A-C. C has no outgoing edge: dead end. "
        "Backtrack to A: no remaining continuation. Return to S. "
        "2. Try S-B; extend to S-B-D, then S-B-D-G. Goal reached. "
        "Check each edge and reconstruct the successful route."
    ),
    answer="S-B-D-G",
)]
agent = aot.AoT("Find a directed route using only the supplied edges.", provider,
                examples=examples)
result = await agent.run("Find S to G. Edges: S-X, S-Y, Y-G.", strategy="dfs")
```

Use BFS-ordered examples with `strategy="bfs"`. Examples are caller-supplied data;
the library does not verify their correctness or synthesize them with extra calls.
Their lengths and quality influence search behavior. Without examples, DFS/BFS
use generic zero-shot instructions, **not** the paper's evaluated few-shot setup.

## Optional initialization and evaluation

`await agent.run(input, warmup=True)` makes one preparation call followed by one
search call. Preparation proposes candidates and selects a compatible starting
set in text. Customize it with `warmup_instructions=` on the constructor.
Alternatively, `warm_start="..."` supplies a starting state or compatible candidate
set directly, skipping preparation even when `warmup=True`. An explicitly supplied
empty string also skips preparation. Both phases retain the original task/input.

This generalizes the two-stage crossword setup in Section 4.2. Unlike its
task-specific compatibility selection, the generic warm-up delegates selection to
the model. A caller needing deterministic compatibility checks should generate
and check candidates outside this agent, then pass `warm_start`.

An optional async `evaluate(output) -> float` constructor argument runs once on
the extracted final artifact. Its finite score is returned as `result.evaluation`;
it does not affect selection, trigger a retry, or certify correctness. The caller
owns domain validation and any required execution isolation.

## Results and failures

`Result` contains `output` (final artifact), `response` (full search response),
`warm_start`, `calls`, and optional `evaluation`. `calls` counts agent-level provider
requests: one normally, two with generated preparation. Provider transport retries
and any model calls made inside an evaluator are outside that count.

Each solve template reserves a standalone `answer:` line for the final artifact.
The parser accepts case variations and CRLF and returns everything after the first
such line without changing artifact whitespace. The marker must not occur earlier
in the search. Missing markers and blank artifacts raise `ValueError`; there is no
repair call, implicit retry, external tree expansion, or fallback answer.
This delimiter is an intentional generalization of the paper's `answer:` and
`Final Passage:` formats. Prose remains prose rather than a forced JSON search tree.

`agent.calls` retains prompts, raw responses, and generation/format errors before
postprocessing. Provider errors and unexpected tool requests propagate. Evaluator
errors or nonfinite scores propagate with `agent.output` and
`agent.evaluation_error` retained. Records reset on each run; use one run at a time
per instance. Calls use explicit context without a mutable Session or hidden history.

The token limit bounds model exploration. Missing finalization is rejected, but
the text-only provider boundary does not expose a finish reason: an artifact
truncated *after* its marker cannot be reliably detected here. Search and constraint
checks are model behavior, not guarantees. Partial answers must identify unresolved
constraints; finding no solution does not establish impossibility.

Templates remain in this folder. Imports do not change Slick's process-global
template root. Configure it before rendering; simultaneous implementations needing
different roots require separate processes.

## Official source status and adaptations

Checked on 2026-09-14:

- The separately supplied [`daankepel/APET`](https://github.com/daankepel/APET)
  repository implements prompt reformulation using a toolbox of prompting
  techniques. Its [`experiment.py`](https://github.com/daankepel/APET/blob/775ab29323d794c6f445156a0c0d344841e7196a/experiment.py)
  was inspected; it is not the official implementation of this AoT paper.
  That algorithm already has its own [APET implementation](../apet/README.md).
- The supplied official URL,
  [`bilgehan-sel/algorithm-of-thoughts`](https://github.com/bilgehan-sel/algorithm-of-thoughts),
  returned HTTP 404 through GitHub's repository API and raw README endpoint.
- The [official project site](https://algorithm-of-thoughts.github.io/) links its
  **Code** button to the [algorithm-of-thoughts account](https://github.com/algorithm-of-thoughts).
  That account lists only the website repository.
- The [website tree at `ab0af8a`](https://github.com/algorithm-of-thoughts/algorithm-of-thoughts.github.io/tree/ab0af8a450f100d1503982f8e2773bf6af618c93)
  was inspected: it contains the site and static assets, with no algorithm implementation.

Consequently no official executable implementation could be reused or audited.
This implementation uses the supplied paper's Section 3, Section 4.2, Section 5,
and Appendix F as its algorithm and prompt references. It does not substitute a
third-party repository for the authors' code.

Deliberate adaptations: generic task text and caller examples, a uniform final-answer
delimiter, model-selected optional initialization, and optional post-run evaluation.
It removes Game of 24 restrictions on negative/fractional values and the assumption
that every problem has a solution. Benchmark runners, fine-tuning, and prompt-length
ablations are outside this algorithm implementation. No benchmark gains are claimed.

## Verification

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_aot
../slick/.venv/bin/ruff check aot tests/test_aot.py
../slick/.venv/bin/ruff format --check aot tests/test_aot.py
```

The offline tests use the shared scripted provider through actual Slick decorators.
They check single-call search, two-call preparation, context/examples, artifact
extraction, evaluation isolation, failure records, state reset, and template loading
from another directory. These verify implementation contracts, not model quality
or reproduction of the paper's results. Checked against the adjacent `slick-ai`
0.3.0 source, including its text parsing and explicit `provider=` boundary.
