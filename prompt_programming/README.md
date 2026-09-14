# Prompt programming beyond few-shot

Problem-agnostic implementation of Reynolds and McDonell's
[Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm](https://arxiv.org/abs/2102.07350).
The paper offers prompting methods, not a trained optimizer. `PromptProgrammer`
implements task specification, serializing metaprompts, multipart generation,
and optional counterfactual selection of fragment insertion points.

## Use

Use the repository's Python environment with the adjacent Slick checkout
(checked against local Slick 0.3.0). No additional dependencies are needed.

```python
from pathlib import Path

import prompt_programming
from slick import prompts

# Configure once at application startup, before any render or generation.
prompts.TEMPLATE_ROOT = Path(prompt_programming.__file__).resolve().parent / "prompts"


async def solve(task, provider, evaluate=None):
    agent = prompt_programming.PromptProgrammer(task, provider, evaluate)
    result = await agent.run()
    return result.answer
```

Supply any task, constraints, source material, and desired answer format as text.
The caller constructs the Slick provider and controls temperature, output-token
limits, stop sequences, timeouts, and transport retries. The paper used
`temperature=0` for its metaprompt examples. For literal paper-style behavior,
use a provider that **continues the supplied text**; a chat provider's turn
formatting changes the experiment.

The default makes two sequential generations:

1. Append the paper's serializing seed to the task and generate a procedure or analysis.
2. Append `\nThus, the correct answer is` and generate the answer separately.

`result.answer` contains only the final call's text; `result.fills` contains the
selected intermediate slots. `result.transcript` contains the full retained
prompt and answer. Text whitespace is preserved. An optional async
`evaluate(answer) -> float` runs once on the final answer; its finite measurement
is returned as `result.score`, with no ranking or assumed score direction.
Without it, `score` is `None`. No candidate code is executed by this package.

## Prompt variants

```python
agent = prompt_programming.PromptProgrammer(your_task, your_provider)

direct = await agent.run(mode="direct")
proxy = await agent.run(mode="proxy", proxy="a patient teacher")
examples = await agent.run(
    mode="demonstration",
    examples=(("source example", "target example"),),
    input_label="Input",
    output_label="Output",
)

multipart = await agent.run(fragments=(
    "In order to solve this problem, we will ",
    " Let's begin.\n",
    "\nFinal answer: ",
))
```

`mode="demonstration"` with no examples is the paper's simple-colon zero-shot
format. Set labels to `French` and `English` to reproduce Figure 1's formatting;
the implementation contains no translation-specific logic. Demonstrations are
used in supplied order, without retrieval or assumptions of independence.

`fragments` applies to metaprompt mode and must contain at least one item. Every
fragment is followed by a generation; the last generation is the answer. Each
fragment includes its own spacing and delimiters. This supports task-specific
procedure generation and expert-generator narratives using the same loop.
Generated instructions become part of subsequent prompts without executing code
or creating an additional agent. The built-in teacher proxy generalizes §4.4;
it is not a verbatim reproduction of the master-translator narrative.

The final fragment can specify a label or output format, but a prompt is not a
parser: this implementation does not guarantee that the model obeys it. Callers
can parse or validate final text in their evaluator or after `run()`.

## Counterfactual insertion

With an ordinary provider, the whole response fills each intermediate slot.
This is a bounded-generation adaptation, **not** the paper's likelihood-based
stopping method. Configure the provider's output limit for these slots.

To use likelihood-based insertion, supply both callbacks:

```python
agent = prompt_programming.PromptProgrammer(
    your_task,
    your_provider,
    token_boundaries=your_token_boundaries,
    score_suffix=your_async_suffix_log_probability,
)
result = await agent.run(max_calls=512)
```

- `token_boundaries(generated_text) -> Sequence[int]` returns Python character
  offsets at the scoring model's token boundaries, in ascending order. Include
  `0` to permit an empty slot and `len(generated_text)` to consider the full
  response. Supply real tokenizer offsets, not word splits or byte offsets.
- `await score_suffix(prefix, suffix) -> float` computes the conditional log
  probability of the **whole** supplied suffix given the exact prefix. Sum
  conditional log probabilities of suffix tokens; do not score the prefix,
  average token scores, or ask a chat model to invent a probability. The adapter
  owns context/suffix tokenization at their boundary. Valid values are nonpositive;
  negative infinity denotes an impossible continuation.

For each intermediate response, the agent scores all supplied boundaries,
retains the highest-likelihood prefix (first supplied boundary wins ties),
discards the remaining generated tail, and injects the next fragment. A first
local maximum does not terminate the scan. No possible boundary raises
`ValueError`. This follows the author's bounded-passage counterfactual parsing
procedure. It is not an unbounded online peak detector.

Slick's checked provider interface exposes text and tool requests, not arbitrary
suffix likelihoods or tokenizer offsets. Those model-specific callbacks stay
outside the algorithm. Its renderer also strips surrounding whitespace, so
`run()` restores the exact composed context when forwarding `fill` and `answer`
calls to the provider; this keeps scoring and continuation prefixes identical.
Decorated `.render` remains Slick's normal stripped preview. For example:
`await PromptProgrammer.direct.render(agent)` explicitly binds the owner.

## Sources and implementation decisions

- The supplied paper, especially §§4.2–4.7 and Figures 1, 3–6, defines the prompt
  methods. The default serial seed and verdict phrase retain its wording.
- The author's [Methods of prompt programming](https://generative.ink/posts/methods-of-prompt-programming/)
  links to [Parsing by counterfactual](https://generative.ink/posts/parsing-by-counterfactual/#code).
  We use its published `conditional_logprob` and `substring_logprobs` procedures:
  score a target suffix at candidate token positions, then select a prefix.
  The code is published under the page's CC0 notice. This implementation adapts
  the algorithm to async callbacks rather than copying its historical API calls.
- On 2026-09-14, the paper page and author materials did not identify a repository
  as an official implementation of this paper. The author's linked
  [Loom repository](https://github.com/socketteer/loom) is a separate tree-writing
  interface; it is not presented here as the paper's official implementation.

There is no prompt search, training, benchmark harness, or automatic model-based
fitness estimate. The paper's incorrect `f(f(3)) = 27` example is an observed
model output, not an expected algorithm result. Offline checks do not reproduce
the translation BLEU scores or establish better task accuracy.

## Failures and checks

`max_calls` bounds all generation, suffix-scoring, and evaluator attempts, including
failed attempts. An unscored program of N fragments uses N generation calls,
plus one optional evaluation. Scoring adds one call per candidate boundary per
intermediate slot. Tokenization is local and not counted. Exhaustion raises
`RuntimeError`; no partial answer is returned. Errors propagate without internal
retries. Blank generated responses and invalid measured scores are rejected.

`agent.calls` retains prompts, raw responses (including rejected output), scores,
and operation errors. It resets at the start of each run; use separate instances
for concurrent runs. Providers own transport-level attempts inside a logical
call. No mutable Session history or tools are used.

The absolute template root above works from another launch directory when the
repository is importable. Imports do not mutate Slick's process-global root;
concurrent applications needing different roots require separate processes.

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_prompt_programming tests.test_prompt_layout
../slick/.venv/bin/ruff check prompt_programming tests/test_prompt_programming.py
```

Tests use the shared `tests.providers.ScriptedProvider` with real Slick rendering
and postprocessing. They check ordered insertion, likelihood peaks and ties,
tail removal, whitespace fidelity, zero/few-shot formats, budgets, and failures.
They make no paid model calls.
