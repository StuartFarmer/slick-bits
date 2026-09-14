# Decomposed Prompting (DECOMP)

Slick implementation of Khot et al., *Decomposed Prompting: A Modular Approach
for Solving Complex Tasks*. The controller generates one sub-question, dispatches
it, appends the actual answer, and repeats until `[EOQ]` returns the last answer.
Programs can call other programs or themselves. Every invocation has its own
numbered answers and history; all invocations share a call budget.

```python
from pathlib import Path

from slick import prompts

import decomp

# Configure once at application startup, before rendering or generation.
prompts.TEMPLATE_ROOT = Path(decomp.__file__).resolve().parent / "prompts"
agent = decomp.paper_agent(provider)
result = await agent.run(
    'Take the second letters of "John Smith" and concatenate using a space.',
    program="letters",
)
print(result.answer)
```

Supply a configured Slick provider. No provider, credentials, model, evaluation,
or retrieval infrastructure is created by this package. The existing local Slick
0.3.0 checkout is used; there are no new dependencies. Run from the repository
root, or add it to `PYTHONPATH` when launching elsewhere.

| Program | Behavior |
| --- | --- |
| `letters` | Split words → foreach `str_position` → merge |
| `str_position` | Split letters → select one-based position |
| `reverse` | Split in halves → recursive reversal → concatenate right then left |
| `context_qa` | Single-hop context QA → foreach with flattening and deduplication |
| `math` | Generate calculation → separately extract its answer |
| `open_qa` | Retrieve and answer each hop → final reading over combined evidence |
| `retrieve_odqa` | External retrieval → prompted single-hop reading |

`paper_agent(provider, context=paragraphs)` supplies context to the coarse QA
handler. To enable open-domain programs, supply an async `retrieve(query)` that
returns JSON-compatible documents containing titles and text. Retrieved content
is available in execution records and passed through the single-hop reader to
the final reader. Copying evidence is prompted, not a provenance guarantee;
validate against the retrieval records when provenance matters.

For a different domain, use `Decomp(task, provider, handlers, programs)`.
`handlers` maps names to async functions taking a resolved question string and
returning a JSON-compatible value. `programs` maps names to `Program(examples)`;
examples use the paper's question/answer transcripts. Select the entry point via
`run(question, program=name)`. The default `decomp` program has no examples.
Callbacks override identically named programs, so a prompted sub-task can be
replaced by a symbolic function without changing the decomposer. The factory's
`handlers=` argument supports the same overrides. Custom programs can also
register fine-grained QA handlers independently.

The parser accepts `QS:` or numbered `Q1:` prefixes (or neither), `[handler]`,
`(select)`, `(project_values)` / `[foreach]`, and
`(project_values_flat.unique)` / `[foreach_merge]`. The underscore spelling
`project_values_flat_unique` is also accepted. It rejects generated answers and
multiple instructions. Reference substitution handles `#1` and `#10` separately
in a single pass. Quoted string references receive JSON escaping; unquoted
strings are inserted as text and other values as JSON. Foreach requires exactly
one referenced list; any other referenced values are held fixed. It preserves
order and duplicates. Foreach-merge requires list-valued results, flattens one
level, and removes duplicates in first-seen order, including structured values.

`Result` contains `answer`, completed `steps` in execution order, and `calls`.
Each step records its program, recursion depth, local index, original instruction,
resolved questions, and answer. `max_calls=256` counts decomposer generations
(including EOQ) and leaf dispatches, including failed attempts. Recursive program
dispatch itself is free; its internal work consumes the shared budget. External
callbacks own any internal retries and costs. `max_depth=16` bounds nesting with
root depth zero. Budget exhaustion raises rather than returning a partial answer.

Provider, parsing, and handler errors propagate without retries. Partial
`agent.steps`, raw decomposer `agent.generations`, and `agent.calls` survive a
failure. Structured leaf parsing is owned by Slick; configure provider logging
to retain raw leaf responses, including rejected JSON. Each run resets records;
use one run at a time per instance. Calls use explicit providers and independent
prompts; conversation sessions are unnecessary because the controller owns history.
Slick's template root is process-global and is never changed during import.

This implements the method, not an experimental reproduction. Decomposer outputs
retain the paper's textual protocol. Leaf responses deliberately use a typed
JSON `answer` envelope, position pairs use JSON arrays, and reversal uses lists
and separate split/base handlers. Few-shot examples are shortened and normalized;
math's two stages are selected by the decomposer rather than a fixed controller.
The bundled QA examples use the coarse scheme. Original datasets, baseline
experiments, historical GPT-3 models, and published accuracy claims are not
reproduced. Binary reversal has logarithmic recursion **depth** and linear total
work/calls for fixed-size leaves.

Offline checks (real Slick rendering/parsing with the shared scripted provider):

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_decomp
../slick/.venv/bin/ruff check decomp tests/test_decomp.py
```
