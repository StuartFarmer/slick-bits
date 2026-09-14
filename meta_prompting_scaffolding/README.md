# Meta-Prompting: task-agnostic scaffolding

Slick implementation of Algorithm 1 from [Suzgun and Kalai (2024)](https://arxiv.org/abs/2401.12954):
one model conducts a sequence of isolated expert consultations, checks their
responses, and returns a final answer. Supply any task as text, including its
context and output constraints. There are no benchmark-specific rules or personas.

This is distinct from this repository's `meta_prompting/`, which implements
Fu's later *Meta-Prompting Protocol* for optimizing reusable instructions.

## Use

Use the repository's `optimizer/.venv` environment or install the adjacent Slick
checkout (`python -m pip install -e ../slick`). No new dependencies are required.
The implementation was checked against the adjacent Slick source declaring 0.3.0.

Configure the process-global template root once at application startup:

```python
from pathlib import Path

from slick import prompts
import meta_prompting_scaffolding
from meta_prompting_scaffolding import MetaPromptingScaffolding

prompts.TEMPLATE_ROOT = (
    Path(meta_prompting_scaffolding.__file__).resolve().parent / "prompts"
)


async def solve(task: str, provider):
    agent = MetaPromptingScaffolding(task, provider)
    result = await agent.run(max_rounds=15)
    return result
```

Pass a configured, stateless Slick provider. The same provider handles conductor
and expert calls; configure the model, sampling parameters, and token limit there.
The paper reports temperature 0, top-p 0.95, and max tokens 1024; the official
runner defaults to temperature 0.1. These settings are not hardcoded here.
Run implementations requiring different template roots in separate processes.

`result.answer` is the first complete final-answer block, with its internal
whitespace preserved, or `None` if the budget was exhausted. Check
`result.stop_reason` (`final_answer` or `round_limit`). The result also contains
`rounds`, the number of provider `calls`, and an immutable `history` snapshot.
`agent.calls` retains prompts, raw responses, and errors; `agent.executions`
retains submitted code and executor outputs/errors. Each `run()` resets records;
use one run at a time per instance.

## Optional Python execution

```python
async def solve_with_python(task, provider, sandbox_execute):
    # sandbox_execute: async (code: str) -> str, supplied by your application.
    agent = MetaPromptingScaffolding(
        task, provider, execute_python=sandbox_execute
    )
    return await agent.run()
```

The executor receives generated source code and returns stdout/diagnostics as
text. It owns the sandbox, permissions, timeout, and output limits. Expected code
errors can be returned as diagnostic text for the conductor to correct; raised
executor exceptions abort the run and remain logged. This package never executes
generated code itself.

With the callback supplied, only a call to exactly `Expert Python` can request
execution. Its response must contain a complete Python or unlabelled fenced code
block followed by `Please run this code!`. As upstream does, we execute the last
such block before the first marker, once per expert call, and feed the code/output
back to the conductor. A malformed request returns repair feedback without
execution. Without the callback, all experts produce text only, and the conductor
receives the official prompt without Python execution capabilities.

## Algorithm and failure policy

1. Initialize history with the task and the official invitation to select experts.
2. Ask the conductor using its instructions and the full accumulated history.
3. If the first `Expert Name: """instruction"""` block is nonblank, call that
   expert with only its name and delegated instruction, plus the generic expert
   prompt. Append its response and the official verification feedback.
4. Otherwise, return the first nonblank `>> FINAL ANSWER: """answer"""` block.
5. Otherwise, append the official formatting reminder and continue.

Each round consumes one conductor call and at most one expert call, so the model
call budget is at most `2 * max_rounds`. Even malformed conductor output consumes
a round. The final round includes the official last-round reminder; no additional
generation occurs after exhaustion. Zero rounds makes no provider calls.

The conductor chooses experts dynamically and communicates all shared context.
Experts cannot call each other and receive no prior conversation. Verification
is requested in the official prompt, not enforced by a hardcoded extra judge;
Algorithm 1 permits immediate final answers. A returned answer is not evidence
of correctness. There is no evaluator or training loop in this inference method.
Transport/executor errors propagate without hidden retries. Native provider tool
requests are rejected; Python uses the explicit callback protocol above.

## Official implementation and adaptations

Adapted from the authors' [official repository](https://github.com/suzgunmirac/meta-prompting),
pinned at `40422564938d772c3e3e6b9614b1df48b8dd6a08`:

- [`utils/meta_scaffolding.py`](https://github.com/suzgunmirac/meta-prompting/blob/40422564938d772c3e3e6b9614b1df48b8dd6a08/utils/meta_scaffolding.py):
  expert routing, fresh context, history feedback, and the Python execution protocol.
- [`prompts/`](https://github.com/suzgunmirac/meta-prompting/tree/40422564938d772c3e3e6b9614b1df48b8dd6a08/prompts):
  both official conductor instruction files are copied verbatim as `system.j2`
  and `system_python.j2`; the baseline JSON supplies the error reminder and
  generic expert instruction.
- [`run_experiments.py`](https://github.com/suzgunmirac/meta-prompting/blob/40422564938d772c3e3e6b9614b1df48b8dd6a08/run_experiments.py):
  initialization suffix, intermediate verification feedback, and Python instruction.

The upstream MIT license is included in [LICENSE](LICENSE).

This port uses Slick decorated text operations and external Jinja templates.
It renders history with explicit role labels into one text prompt instead of
upstream's chat-message API. No shared mutable Session is used: Python owns the
history, and each provider call is independent. This preserves information flow,
but does not claim identical API-role semantics or model outputs.

Algorithm 1 specifies the first expert block and a fixed bound; upstream executes
multiple blocks and uses recursive calls with a 16-round cutoff. This port follows
Algorithm 1, defaults to the prompt's 15 rounds, and explicitly reports exhaustion.
Unlike upstream's marker-only stop, final answers require a complete, nonblank
triple-quoted block. Callback-owned execution replaces upstream execution machinery;
transport retry loops, benchmark evaluations, and optional multi-sample summarizing
are excluded. The official instruction text still mentions 15 rounds when a caller
overrides the runtime limit.

## Checks

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_meta_prompting_scaffolding
../slick/.venv/bin/ruff check meta_prompting_scaffolding tests/test_meta_prompting_scaffolding.py
```

The shared scripted provider exercises context isolation, history retention,
delimiter precedence, malformed responses, budgets, Python opt-in, exception
records, reruns, and templates from another working directory. These are
deterministic integration checks; no live-model benchmark results are claimed.
