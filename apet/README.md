# APET

Problem-agnostic Slick implementation of **Autonomous Prompt Engineering in Large
Language Models**, by Daan Kepel and Konstantina Valogianni. It uses the supplied
paper and the authors' [official implementation](https://github.com/daankepel/APET).

APET makes one zero-shot rewrite using a toolbox of Expert Prompting, Chain of
Thought, and simulated three-expert Tree of Thoughts prompting. The model chooses
the techniques and their combination. A fresh call answers the rewritten prompt.
There is no candidate search, training data, tree-search controller, or iterative
optimization loop in this algorithm.

## Use

Use the repository's `optimizer/.venv` environment, which already has Slick, or
install the adjacent checkout from the repository root with
`python -m pip install -e ../slick`. No new dependencies are required.

Configure the local templates once at application startup:

```python
from pathlib import Path

from slick import prompts
import apet

prompts.TEMPLATE_ROOT = Path(apet.__file__).resolve().parent / "prompts"


async def solve(question: str, provider):
    agent = apet.APET(question, provider)
    result = await agent.run()
    return result.answer
```

`question` is the complete task prompt: instructions, source material, constraints,
and desired output format. It can ask for analysis, writing, code, classification,
or any other textual task. The agent returns text and never executes it.
The caller provides a stateless Slick provider and owns model selection,
credentials, decoding settings, transport retries, and persistence.

To obtain just the rewritten prompt, use `await agent.optimize()` (one model
call). `await agent.run()` rewrites and answers (two calls).
`await agent.run(compare=True)` follows the official experiment order: rewrite,
original answer, optimized answer (three calls). Both answers are independent;
neither receives the other answer or the optimizer conversation.

For exact input retention, identify source text already included in the question:

```python
async def summarize(document: str, provider):
    question = f"Summarize this document in three bullets:\n\n{document}"
    agent = apet.APET(question, provider, input_text=document)
    return await agent.run()
```

As in the official code, when `input_text` is absent as an exact substring of the
rewrite, it is appended under `Input for question:` with triple-quote delimiters.
Empty `input_text` disables this safeguard. This protects only the supplied text;
preservation of other instructions relies on the model. Supply the entire original
question as `input_text` if all of it must remain verbatim. The optimizer receives
`question`, so include all source material there even when also protecting it.

## Optional comparison and scoring

Inject an async `evaluate(answer) -> float` callback to score generated answers:

```python
async def compare(question, provider, evaluate):
    agent = apet.APET(question, provider, evaluate)
    return await agent.run(compare=True)
```

The evaluator can close over a reference answer, rubric, or task context. It runs
after generation, on the original answer first (when requested), then the optimized
answer. References and scores never enter the optimizer prompt. Without `compare`,
only the optimized answer is scored. Scores must be finite; their direction and
interpretation belong to the evaluator. No score affects which answer is returned.

`Result` contains `original_prompt`, `optimized_prompt`, `answer` (the optimized
answer), `original_answer`, `original_score`, `optimized_score`, and `calls`.
Absent baseline answers or evaluations are `None`. APET can make an answer worse;
the comparison reports both outcomes without silently selecting a winner.

## Official source and adaptations

Source checked on 2026-09-14 at commit
[`775ab29323d794c6f445156a0c0d344841e7196a`](https://github.com/daankepel/APET/commit/775ab29323d794c6f445156a0c0d344841e7196a).
The reference is
[`experiment.py`](https://github.com/daankepel/APET/blob/775ab29323d794c6f445156a0c0d344841e7196a/experiment.py):

| Official function | Used here |
| --- | --- |
| `prompt_optimization` | Toolbox wording, one rewrite, exact-substring input restoration |
| `run_Benchmark` | Rewrite → original answer → optimized answer ordering for comparison |
| `generate_response` | Same provider across independent generations |

`prompts/reformulate.j2` preserves the official optimizer's wording, ordered
techniques, and four-quote delimiters. Slick's portable provider API accepts one
context string, so its system and user instructions are concatenated with a blank
line. This changes message roles relative to upstream; it is not a byte-for-byte
API reproduction. Slick also strips the rendered context's surrounding whitespace.
Generated outputs otherwise retain their original whitespace; protected input
is embedded verbatim inside its delimiters.

The benchmark-specific answer system prompt, single-expression requirement,
`>> FINAL ANSWER` extraction, Excel/CSV I/O, and benchmark evaluators are omitted.
Put output requirements in your task and interpret answers in your evaluator.
Upstream's retry path and exception-to-empty-string fallback are replaced by
explicit error propagation; there are no hidden repair generations here.
The official script uses `gpt-4-turbo-2024-04-09` at temperature 0. This library
does not impose model settings; configure the injected provider for your experiment.

The paper's Tree of Thoughts option is simulated expert discussion within a
prompt, not Yao et al.'s executable BFS/DFS algorithm. No external agents or tools
are spawned by APET. The implementation references the official source without
importing its dataset and OpenAI experiment infrastructure.

## Failures and verification

`agent.calls` retains each operation, rendered context, raw response (when
received), and generation error. Raw text is saved before validation, including
blank responses and unexpected tool requests. Such outputs and provider failures
stop the run immediately. Evaluator failures and nonfinite scores also propagate;
completed generation records remain available. There is no result on failure.
Call counts exclude evaluator work and transport attempts internal to a provider.

`optimize()` and `run()` reset call records. Use one operation at a time per agent.
Imports do not change Slick's process-global template root; configure it before
calls, and use separate processes for concurrent algorithms needing different roots.

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_apet
../slick/.venv/bin/ruff check apet tests/test_apet.py
../slick/.venv/bin/ruff format --check apet tests/test_apet.py
```

Tests use the shared scripted provider through real Slick templates and decorators.
They check call isolation/order, verbatim input restoration, arbitrary tasks,
evaluation isolation, worse optimized answers, failures, and template loading from
the repository and algorithm directories. These are offline algorithm and interface
checks, not reproduced benchmark scores or evidence of improved answer quality.
