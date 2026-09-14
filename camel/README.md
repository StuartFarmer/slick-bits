# CAMEL

Problem-agnostic Slick implementation of **CAMEL: Communicative Agents for
“Mind” Exploration of Large Language Model Society** (Li et al., 2023).
An AI user plans one instruction at a time; an AI assistant supplies solutions.
Both receive the specified task, assigned roles, and the entire selected conversation.

## Use

Use the repository's `optimizer/.venv` environment, or install the adjacent Slick
checkout with `python -m pip install -e ../slick` from the repository root.
No additional dependencies are needed.

```python
from pathlib import Path

from slick import prompts
import camel

prompts.TEMPLATE_ROOT = Path(camel.__file__).resolve().parent / "prompts"


async def solve(task, provider):
    agent = camel.CAMEL(
        task,
        provider,
        assistant_role="Subject specialist",
        user_role="Project owner",
    )
    return await agent.run(max_messages=40, extract=True)
```

Supply any textual task, source material, constraints, and desired output format.
Roles are caller-selected strings. The algorithm contains no benchmark, domain
logic, generated-code execution, or provider construction. The caller owns model
selection, credentials, decoding settings, transport retries, and persistence.
Use stateless Slick providers; the same provider may serve both roles.
`user_provider=` and `specifier_provider=` optionally use different models.

By default, one task-specifier call makes the idea concrete using the official
50-word prompt. This is a prompted limit, not truncation or a validation rule.
For a fully specified problem whose scope and inputs must stay intact, use
`await agent.run(specify_task=False)`. Task specification deliberately invites
imagination and can change the scope of a vague idea.

`Result` contains the effective `task`, immutable `messages` and `selections`,
`stop_reason`, optional extracted `solution`, and actual provider `calls`.
Messages retain raw generated text and identify their `speaker` as `user` or
`assistant`. The done token and protocol-violating final message are retained.
`extract=True` adds one call using Appendix H's full-solution extraction prompt
when an assistant message exists. Otherwise `solution` is `None`; the transcript
is the primary output. An extracted result from an interrupted conversation may
be incomplete. Extraction does not change the conversation's stop reason.
External evaluation belongs to the caller, for example
`score = await evaluate(result.solution)` after checking that it exists.
No evaluator or numerical fitness search participates in CAMEL's base algorithm.

## Termination and budgets

| Stop reason | Condition |
| --- | --- |
| `task_done` | The selected AI user reply, stripped, equals `<CAMEL_TASK_DONE>` |
| `user_no_instruct` | Three consecutive selected user replies lack a nonempty `Instruction:` line |
| `assistant_instruct` | The selected assistant reply contains a nonempty `Instruction:` line |
| `message_limit` | The selected conversation reaches `max_messages`, default 40 |
| `token_limit` | The rendered prompt reaches the supplied token budget, or an adapter raises `TokenLimitError` |

An instruction resets the missing-instruction count. The first two missing
instructions still receive assistant replies. `Input:`, `Solution:`, and
`Next request.` are prompted conventions; they are not mandatory parse schemas.
The instruction detector is a case-sensitive line heuristic: quoted/fenced
`Instruction:` lines can match, and implicit instructions can go undetected.
The exact standalone done check avoids stopping when the token is merely mentioned
inside an instruction. An assistant's done token never declares success.

The message limit counts **individual user and assistant messages**, including
termination messages. An odd limit can leave an unanswered instruction. The
default is at most 20 pairs, not 40 pairs. Specification, candidate alternatives,
critic reviews, and optional extraction are outside that message count.
With no critic, generation costs at most `max_messages + 1` calls when specifying,
plus one if extracting. `max_messages=0` generates no conversation messages;
specification still runs unless disabled.

For model-specific context control, supply `count_tokens(text) -> int` and
`token_limit=` together. The counter receives the exact rendered prompt; set the
limit to the usable input allowance after reserving completion tokens and any
provider-added framing. A blocked call is recorded with `attempted=False` and
does not increase `Result.calls`. No tokenizer is guessed. Without this pair,
the provider controls its context limit. Its adapter can raise `camel.TokenLimitError`
for context overflow or a truncated completion, which returns partial conversation
state. Slick's text-only `acall` interface does not expose finish reasons itself.

## Optional critic

Pass `critic=critic_provider, candidates=3` to sample independent alternatives
at **each** user and assistant turn. The critic sees all proposals, the selected
conversation, role assignments, and `criteria` (default: improving task performance).
Only its selected proposal enters the conversation. The next turn cannot see
rejected alternatives. `Result.selections` preserves proposals and explanations.
This is the paper's local expansion/selection procedure, without MCTS rollouts,
backpropagation, or numerical tree search.

Alternatively supply an async human/external reviewer:

```python
async def review(speaker, proposals, history):
    # Your application obtains a decision from its UI or other review service.
    option, explanation = await choose_in_application(speaker, proposals, history)
    return camel.Choice(option=option, explanation=explanation)
```

Pass it as `review=review`; it takes precedence over `critic=`. Options are
one-based. Choices must identify an existing proposal and include an explanation.
`candidates=1` bypasses selection; without either critic, exactly one candidate is
generated regardless of the candidates setting. With an AI critic and `k > 1`,
each selected message costs `k + 1` provider calls. With an external reviewer,
it costs `k` provider calls plus one callback. Critic choices are JSON-validated;
invalid output fails immediately, without random selection or hidden retries.

## Official source and adaptations

The paper explicitly identifies [camel-ai/camel](https://github.com/camel-ai/camel)
as its official implementation. This implementation uses release `v0.1.0`, commit
[`bda4bf43bb03092936c2c0316aee9ff96aa551bf`](https://github.com/camel-ai/camel/commit/bda4bf43bb03092936c2c0316aee9ff96aa551bf),
checked on 2026-09-14:

| Official source | Used here |
| --- | --- |
| [`prompts/ai_society.py`](https://github.com/camel-ai/camel/blob/bda4bf43bb03092936c2c0316aee9ff96aa551bf/camel/prompts/ai_society.py) | Task specification and assistant/user/critic inception prompts, adapted verbatim to local Jinja variables |
| [`agents/role_playing.py`](https://github.com/camel-ai/camel/blob/bda4bf43bb03092936c2c0316aee9ff96aa551bf/camel/agents/role_playing.py) | User-first turn ordering and committing only selected messages |
| [`agents/critic_agent.py`](https://github.com/camel-ai/camel/blob/bda4bf43bb03092936c2c0316aee9ff96aa551bf/camel/agents/critic_agent.py) | Numbered proposal menus and critic selection with explanations |
| [`agents/chat_agent.py`](https://github.com/camel-ai/camel/blob/bda4bf43bb03092936c2c0316aee9ff96aa551bf/camel/agents/chat_agent.py) | Full retained history and pre-call token exhaustion |
| [`agents/task_agent.py`](https://github.com/camel-ai/camel/blob/bda4bf43bb03092936c2c0316aee9ff96aa551bf/camel/agents/task_agent.py) | One task-specification call and default 50-word limit |

The supplied [daankepel/APET](https://github.com/daankepel/APET) link implements a
different prompt-optimization algorithm, already available in [`../apet`](../apet/README.md).
It is not used as the CAMEL implementation.

Slick accepts a single context string, so inception prompts and labeled history
are rendered together instead of passed as separate native system/chat messages.
The owning class explicitly maintains this transcript rather than using a mutable
Slick Session: this preserves both sides' inputs and keeps rejected candidates
out of shared history. Source prompt wording is retained, but native message-role
priority is not reproduced. The task specifier uses the paper's prompt without
the upstream extra system sentence. The unused assistant warm-up generation in
upstream `init_chat()` is omitted. The user starts directly from its inception
prompt, as in the paper's equations.

The 40-message and protocol guards follow Section 4.1; upstream's early example
instead uses 50 turns and tests the done token after generating both sides.
Here termination is checked immediately after the relevant selected reply.
The critic's free-text integer extraction, retry loop, and random fallback are
replaced by a typed `Choice`. Alternatives are separate stateless calls rather
than provider-specific `n` completions; the critic receives the full current
conversation rather than its own prior review history. These are explicit
adaptations, not a byte-for-byte reproduction of the API experiment.

The generic role-playing algorithm, optional critic, and solution extraction are
implemented. Dataset factories, fine-tuning, benchmark evaluation, task-planner
ablations, and embodied tools are outside this folder's scope. Source-derived
prompts are covered by the included [Apache-2.0 license](LICENSE) and [NOTICE](NOTICE).
There is no runtime dependency on the modern CAMEL framework; this local package
uses the same `camel` import name, so use separate environments if you need both.

## Failure records and checks

`agent.calls` retains operation, role, rendered prompt, attempt status, raw
response when received, and error before parsing or rejection can discard it.
`agent.reviews` retains external review attempts and failures. Partial selected
messages and completed selections remain inspectable after exceptions. Provider,
tool-request, blank specification/extraction, parsing, and invalid-choice errors
propagate without algorithm retries. Extraction errors propagate even after a
completed conversation; `agent.stop_reason` remains available. Empty conversational
messages remain protocol observations. A run resets all records.

Configure the process-global template root once before running. Imports do not
change it; concurrent algorithms needing different roots require separate processes.

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_camel
../slick/.venv/bin/ruff check camel tests/test_camel.py
../slick/.venv/bin/ruff format --check camel tests/test_camel.py
```

Tests use the shared scripted provider through real Slick decorators and templates.
They exercise history, termination, budgets, rejected branches, critic validation,
failure records, and loading from different working directories. These offline
checks establish algorithm/interface behavior, not task accuracy or reproduced
paper evaluation scores. `task_done` is the AI user's judgment only.
