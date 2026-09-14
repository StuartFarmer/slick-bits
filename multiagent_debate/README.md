# Multiagent debate

Problem-agnostic Slick implementation of Du et al., *Improving Factuality and
Reasoning in Language Models through Multiagent Debate*. Agents independently
answer a task, then revise their answers using other agents' previous responses.

```python
from pathlib import Path

from slick import prompts

import multiagent_debate
from multiagent_debate import MultiAgentDebate

# Configure once at application startup, before rendering or generation.
prompts.TEMPLATE_ROOT = Path(multiagent_debate.__file__).resolve().parent / "prompts"

async def solve(task: str, provider):
    agent = MultiAgentDebate(task, [provider] * 3, debate_rounds=2)
    return await agent.run()

# In your async application, with a configured Slick provider:
# result = await solve("Your task, context, constraints, and output format.", provider)
# print(result.responses)
```

The task is the complete initial prompt, so it can request prose, an analysis,
code, a plan, or any other textual artifact. Put any desired explanation and
answer-format instructions there. The algorithm treats responses as text and
does not parse numbers, multiple-choice letters, or task-specific delimiters.
It does not execute generated content. There is no evaluator in the debate loop;
applications own external evaluation and any necessary execution isolation.

Supply one provider entry per agent. `[provider] * 3` samples the same model
three times with separate conversations; `[provider_a, provider_b]` mixes models.
Configure sampling, credentials, model selection, and transport policy on the
providers. This package adds no dependencies beyond the local Slick installation
(checked against the adjacent 0.3.0 source). Run from the repository root, or
put that root on `PYTHONPATH` when launching elsewhere.

| Setting | Meaning |
| --- | --- |
| `providers` | Nonempty sequence; entry order is the stable agent order |
| `debate_rounds=2` | Number of revisions **after** independent initialization |
| `style="long"` | Treat peer responses as additional advice |
| `style="short"` | Encourage agreement based on peers' opinions |
| `summarizer=None` | Concatenate peers directly; supply a provider to summarize peers before each revision |

Caller settings are trusted: use at least one provider and a nonnegative integer
round count. Zero debate rounds returns independent samples. One agent performs
self-reflection on each round and never calls the summarizer. There is no early
stopping or automatic consensus detection. Agreement does not establish correctness.

`Result.responses` contains all final responses. `Result.rounds` contains the
initial responses followed by each completed revision round; both use provider
order. `Result.calls` counts attempted generation boundaries, including summaries.
Choose a final response or apply a task-specific answer extractor/selector outside
the agent. The implementation does not add a judge call or equate wording differences
with substantive disagreement.

For `N` agents and `R` revisions, direct debate makes `N * (1 + R)` calls.
Summarization adds `N * R` calls when `N > 1`: each agent receives a separate
summary of only its peers, excluding itself. The summary is generated without
conversation history and is then passed to the selected short/long revision
operation. This compresses peer context, but does not truncate an agent's own
history or guarantee a token limit.

Calls run sequentially, matching the official scripts and avoiding concurrent use
of caller-owned providers. Every revision reads a fixed previous-round tuple;
an earlier agent's new answer cannot leak into a later agent's current round.
Each agent owns a fresh Slick `Session`. Native Slick providers carry conversation
continuations; custom `acall`-only providers receive Slick's portable transcript
of earlier assistant outputs. That fallback omits earlier user messages from the
provider input, so the original task is repeated in every revision. Full native
chat replay requires a provider with Slick's continuation support.

There are no internal retries, fallbacks, tools, or failure suppression. Each
session operation allows one provider turn. Empty generated text raises
`ValueError`; nonempty text is returned unchanged. On failure, `agent.calls`,
`agent.generations` (raw outputs, including rejected blank responses),
`agent.sessions` (conversation records), and `agent.rounds` (only completed
rounds) remain available. Provider errors propagate as received; provider-internal
retries are outside this call count. A new `run()` resets all records and sessions.
Use one run at a time per instance. Template configuration is process-global;
imports and constructors do not change it.

## Source correspondence

The implementation was developed against the
[official repository](https://github.com/composable-models/llm_multiagent_debate)
at commit `9846749350eb917ae5bfaaff4c645fc705b8d3af`, alongside the supplied paper
([arXiv:2305.14325](https://arxiv.org/abs/2305.14325)).

| Source | Algorithm retained |
| --- | --- |
| [math/gen_math.py](https://github.com/composable-models/llm_multiagent_debate/blob/9846749350eb917ae5bfaaff4c645fc705b8d3af/math/gen_math.py) | Independent conversations, peer exclusion, fixed `2 * round - 1` snapshot, single-agent reflection |
| [gsm/gen_gsm.py](https://github.com/composable-models/llm_multiagent_debate/blob/9846749350eb917ae5bfaaff4c645fc705b8d3af/gsm/gen_gsm.py) | Initialize all agents, repeatedly revise using peers, retain final responses |
| [mmlu/gen_mmlu.py](https://github.com/composable-models/llm_multiagent_debate/blob/9846749350eb917ae5bfaaff4c645fc705b8d3af/mmlu/gen_mmlu.py) | Peer advice combined with the agent's earlier answer |
| Paper §2.2 and §3.3 | Short/long prompt variants, heterogeneous models, optional peer summarization |

This is a fresh Slick implementation of the algorithm, not a wrapper around the
benchmark scripts. Initial prompts are entirely caller-supplied; revision prompts
deliberately remove domain-specific instructions while retaining the paper's
short/long distinction and upstream fenced peer-response format. The summarization
prompt is new: the checked official repository does not provide that variant.

Upstream `rounds` includes initialization (`rounds=2` means one revision).
Here, `debate_rounds=2` explicitly means initialization plus two revisions, or
nine calls for three agents. Use `debate_rounds=1` to match upstream `rounds=2`.
Benchmark datasets, numeric answer parsers, majority-vote evaluation, biography
early stopping, historical models, and recursive API retries stay outside this
generic implementation. No published accuracy result is claimed or reproduced.

## Offline verification

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_multiagent_debate tests.test_prompt_layout
../slick/.venv/bin/ruff check multiagent_debate tests/test_multiagent_debate.py
```

The tests use real Slick rendering and sessions with the shared
`tests/providers.py` scripted provider. They check round synchronization, peer
exclusion, separate histories, fresh runs, mixed providers, reflection,
summarization, failure propagation, and generation counts without paid model calls.
