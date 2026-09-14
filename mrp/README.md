# Meta-Reasoning Prompting (MRP)

Task-agnostic implementation of **Meta Reasoning for Large Language Models**
(Gao et al., 2024), Algorithm 1 and the method flows in Appendix A.2.
Score each available method from 1–7, select the highest score, then execute
only that method on the original input.

## Use

Use the repository's `optimizer/.venv`, which imports the adjacent Slick checkout
(declared version 0.3.0). Alternatively, install that checkout from the repository
root with `python -m pip install -e ../slick`. No new dependencies are needed.

```python
from pathlib import Path

from slick import prompts
import mrp

# Configure once at application startup, independently of the working directory.
prompts.TEMPLATE_ROOT = Path(mrp.__file__).resolve().parent / "prompts"


async def solve(task: str, input: str, provider):
    agent = mrp.MRP(task, provider)
    result = await agent.run(input)
    return result
```

`task` supplies instructions, constraints, domain context, and output requirements.
`input` is any text: a question, document, code, plan, or other material. The caller
supplies a configured Slick provider, owning model choice, sampling settings,
credentials, token limits, transport retries, and persistence. No evaluator is
required: this algorithm routes by model-estimated suitability, not measured fitness.
External evaluation and execution isolation belong to the application.

`result.output` is the winning method's full response, preserving whitespace.
It may include strategy, critique, or examples before the final deliverable;
there is no task-specific answer parser. `result.method` gives the chosen name,
`result.assessments` retains ordered `Assessment(method, score)` values, and
`result.calls` counts this agent's decorated calls.

## Method pool and execution

Each method is scored in a separate independent call, as in Algorithm 1's loop.
Python computes the argmax; ties select the first method in pool order. Selection
cannot invent a method or disregard the scores. All seven methods use the same
provider, and only the chosen method executes.

| Method | Execution after selection | Calls |
| --- | --- | --- |
| Chain-of-Thoughts | One stepwise solution | 1 |
| Tree-of-Thoughts | Independent strategy-and-answer proposals, then one vote | `proposals + 1` |
| Analogical Prompting | Five solved analogous examples, then the original solution | 1 |
| Self-Refine | Initial answer, then combined critique and revision | 2 |
| Solo Performance Prompting | Personas, profiles, and simulated collaboration in one response | 1 |
| Step-Back Prompting | Extract principles, then solve using them | 2 |
| SimToM | Establish relevant perspective/knowledge, then answer using it | 2 |

The default seven-method pool costs seven scoring calls plus execution: 8, 9,
or 11 calls with `proposals=3`. The proposal count is a local configurable default,
not a recovered MRP experimental setting. Independent proposals have identical
prompts; diversity depends on the caller's model and decoding configuration.
The vote selects an existing response without another generation.

Replace the pool with your own method descriptions:

```python
agent = mrp.MRP(task, provider, methods=[
    mrp.Method("Evidence-first", "Identify supporting evidence, then answer with citations."),
    mrp.Method("Constraint-first", "List the constraints, construct a solution, and check it."),
])
result = await agent.run(input)
```

Without an executor, a custom method applies its description in one solve prompt.
To attach an existing algorithm, supply an async callback:

```python
async def specialized_solver(input: str) -> str:
    # Existing solver supplied by your application; task/provider can be captured.
    return await my_solver(input)

agent = mrp.MRP(task, provider, methods=[
    mrp.Method("My solver", "Describe its actual capabilities and cost.", specialized_solver),
])
```

Callbacks receive the original input, execute once if selected, and return nonblank
text. They own their resources, context, logging, and budgets; their internal calls
are excluded from `result.calls`. Supply a nonempty method pool and a positive
proposal count. Caller configuration is trusted, without preflight validation.

## Prompts, records, and failures

Operations have separate local Jinja files. `methods.json` stores neutral method
summaries; these are paraphrases adapted to the actual bundled execution flows,
not verbatim abstracts. Shared task text is rendered by `prompts/task.j2`.
No Jinja branches select operations. Slick's template root is process-global:
configure it before running and use separate processes for implementations needing
different roots concurrently. Imports do not change it. Calls use the provider
directly, without a conversational Session or implicit history.

Generated scores are strict integers in [1, 7]; votes must be strict in-range
integer indices. Blank text, malformed JSON, unexpected tools, and provider or
callback errors propagate immediately. There are no automatic repairs, retries,
fallback methods, or repeated self-refinement loops.

`agent.calls` retains operation, rendered prompt, raw response when available,
and errors, including responses rejected during parsing. Partial `assessments`,
`selected`, `output`, and `execution_error` remain inspectable after failure.
Each `run()` resets state; use one run at a time per instance. Caller-owned logging
is needed for durable records and callback-internal failures.

## Sources and deliberate adaptations

Sources checked on 2026-09-14:

- [MRP paper](https://arxiv.org/abs/2406.11698): Algorithm 1 defines the scoring
  loop and argmax. Appendix A.2 defines the bundled execution variants. Figure 2
  also presents a joint whole-pool selection prompt; this implementation follows
  the per-method loop instead. Strict score JSON replaces `>> FINAL CHOICE:` so
  Python can enforce the stated selection rule.
- The paper's [GeneralAI link](https://aka.ms/GeneralAI) resolves to the authors'
  [general research site](https://thegenerality.com/agi/). No official MRP code
  repository was verified from that site or the paper. This is an independent
  Slick implementation, not a claimed port of unavailable MRP source code.
- Official [Tree of Thoughts repository](https://github.com/princeton-nlp/tree-of-thought-llm),
  specifically [`bfs.py`](https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/methods/bfs.py):
  used its independent sampling, candidate voting, and selection mechanics as
  implementation references. Here the proposals are complete answers, as in MRP
  Figure 8; this is not the upstream multilevel BFS/DFS search. Vote JSON uses
  zero-based indices instead of the paper's prose delimiter.
- Official [SPP repository](https://github.com/MikeWangWZHL/Solo-Performance-Prompting),
  specifically [`spp_prompt_profile`](https://github.com/MikeWangWZHL/Solo-Performance-Prompting/blob/main/prompts/trivia_creative_writing.py):
  used its participant identification, profiles, critique, and revision sequence
  inside one response. The task-specific examples and trivia wrapper are removed.
- Official [Self-Refine flow](https://github.com/madaan/self-refine/blob/main/src/commongen/run.py)
  was consulted to distinguish the original iterative algorithm from MRP Figure 10.
  MRP's two-call initial-answer/combined-critique-and-revision variant is retained;
  the upstream repeated feedback loop is not imported.

These references inform the implementation; their packages are not runtime
dependencies and no upstream source files are vendored. Existing repository
`tot/` and `self_refine/` implement the fuller original algorithms; MRP's appendix
variants intentionally have different call sequences.

Benchmark-specific few-shot examples, math-only analogies, BigToM sentence
heuristics, and domain-specific output delimiters are generalized. SimToM keeps
both stages and perspective instructions, but does not enforce a factual knowledge
boundary. These are deliberate prompt changes, not exact experimental replication.
No benchmarks, model training, top-k ensembles, or generated-code execution are included.

## Checks

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_mrp
../slick/.venv/bin/ruff check mrp tests/test_mrp.py
../slick/.venv/bin/ruff format --check mrp tests/test_mrp.py
```

Offline tests use the shared scripted provider with real Slick rendering and
parsing. They cover all seven paths, score ordering and ties, call counts, custom
methods, failure evidence, and templates loaded outside the repository root.
They verify orchestration and contracts, not model quality or the paper's results.
