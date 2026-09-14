# Multi-Agent Debate (MAD)

Problem-agnostic Slick implementation of Liang et al., *Encouraging Divergent
Thinking in Large Language Models through Multi-Agent Debate*. Two debaters
speak in fixed order, and a judge decides whether to stop after each round.
After the round limit, a final judge extracts candidate answers and selects an
answer. Defaults: two debaters, three rounds, moderate disagreement.

## Use

```python
from pathlib import Path

from slick import prompts

import mad

# Configure once at application startup, before any prompt calls.
prompts.TEMPLATE_ROOT = Path(mad.__file__).resolve().parent / "prompts"

# provider is any configured Slick provider supplied by your application.
agent = mad.MultiAgentDebate(
    task="Design a text file format for the following requirements: ...",
    provider=provider,
    criteria="Satisfy the requirements; make parsing and manual editing simple",
    answer_format="A format specification with one example document",
)
result = await agent.run(max_rounds=3)
print(result.answer)
print(result.reason, result.rounds, result.stop_reason)
```

Put the question, evidence, constraints, and context in `task`. `criteria` guides
the judge; `answer_format` describes the desired artifact. There is no built-in
translation, arithmetic, dataset, retrieval service, or code execution. Optional
`negative_provider=` and `judge_provider=` select different role backbones; both
default to the supplied provider. Configure temperature and model on the provider
(the paper used temperature zero). The paper reports judge bias with mixed models.

The algorithm's evaluator is its judge. External evaluation is intentionally
outside the debate and does not change the paper's stopping rule:

```python
# Your async evaluator owns domain validation and any execution isolation.
score = await evaluate(result.answer)
```

Use the existing environment with the adjacent Slick checkout (inspected version
`slick-ai` 0.3.0); no new dependencies are required. Run from the repository root,
or put it on `PYTHONPATH` when launching elsewhere. The template root is
process-global; configure it once, never change it concurrently for different
algorithms. Importing `mad` does not change that root.

## Flow, records, and failures

1. Affirmative proposes an answer; negative sees it and challenges it.
2. The discriminative judge reads all arguments and its earlier judgments.
   A nonempty `answer` ends the debate; an empty answer continues it.
3. Later rounds let affirmative respond to the latest negative argument, then
   negative respond to the new affirmative argument. Both see all prior arguments.
   Judge assessments are not fed to debaters, matching the official controller.
4. If still undecided after the last judge call, extract candidates from **all**
   debate rounds and select a nonblank final answer. This final judge does not
   inherit earlier discriminative judgments.

Each round costs three logical provider calls; fallback adds two. A run with
limit `R` therefore uses at most `3 * R + 2` calls. There are no internal retries
or hidden baseline calls. Transport retries and token limits belong to the
provider. `max_rounds=0` is an extension that directly proposes candidates and
selects an answer in two calls.

`Result` exposes `answer`, `reason`, completed `rounds`, `stop_reason` (`judge` or
`round_limit`), immutable `turns`, and the logical `calls` count. Each `Turn`
contains its round, speaker, and verbatim content. `agent.judgments` retains
per-round judge decisions. `agent.calls` retains the operation, speaker, round,
rendered prompt, raw response, and any error, including malformed JSON and
postprocessing rejection. The final extraction and selection are in those call
records; they are not debater turns.

Blank debate/candidate text, malformed judgment objects, a blank final answer,
provider errors, and tool requests abort immediately. Exceptions propagate;
partial records remain inspectable. No silent fallback or JSON repair occurs.
Judge answers strip surrounding whitespace; debater prose is preserved. Caller
configuration is trusted. Each run resets state; use one run at a time per instance.
Histories are rendered explicitly, so no mutable Slick Session is shared by roles.

## Official implementation and adaptations

The implementation was developed against the paper's
[official repository](https://github.com/Skytliang/Multi-Agents-Debate), pinned to
commit `e58d146033568ccf737d8031a93748597907d7d3`. The source files inspected were:

- [interactive.py](https://github.com/Skytliang/Multi-Agents-Debate/blob/e58d146033568ccf737d8031a93748597907d7d3/interactive.py):
  initialization, ordered turns, adaptive break, and two-call final judging.
- [config4all.json](https://github.com/Skytliang/Multi-Agents-Debate/blob/e58d146033568ccf737d8031a93748597907d7d3/code/utils/config4all.json):
  generic role instructions, disagreement policy, and judge operations.
- [agent.py](https://github.com/Skytliang/Multi-Agents-Debate/blob/e58d146033568ccf737d8031a93748597907d7d3/code/utils/agent.py):
  per-role chat histories and transport retry behavior.
- [debate4tran.py](https://github.com/Skytliang/Multi-Agents-Debate/blob/e58d146033568ccf737d8031a93748597907d7d3/code/debate4tran.py)
  and its config: translation-specific baseline and runner, excluded here.

The algorithm and prompts are adapted from those sources under
[GPL-3.0-or-later](LICENSE), with upstream attribution retained. No upstream
package installation or legacy OpenAI client is required.

Deliberate differences from that checkout:

- Its fallback reads `memory_lst[2]`, the opening arguments. Here extraction sees
  all rounds, following Section 2's definition of the extractive judge over the
  whole history.
- Its `eval` parsing becomes Slick/Pydantic JSON parsing of `{answer, reason}`.
  The redundant preference/side fields are omitted; the official stop condition
  already uses only a nonempty answer. Final abstention now raises.
- Per-role native chat messages become explicitly labeled text histories. Prompts
  are adapted to task criteria and output requirements in local Jinja files;
  they are not byte-identical reproductions of upstream prompts.
- Provider construction, legacy retries, CLI, persistence, translation baseline,
  and benchmarks remain outside the algorithm. The main two-debater method is
  implemented; extra-debater and forced-disagreement ablations are not included.

This implements the algorithm, not the paper's reported accuracy or token costs.
The judge's answer is a model decision, not independently measured correctness.

## Checks

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_mad
../slick/.venv/bin/ruff check mad tests/test_mad.py
../slick/.venv/bin/ruff format --check mad tests/test_mad.py
```

Offline tests use the repository's shared `tests.providers.ScriptedProvider`
through actual Slick rendering/parsing. They exercise sequential visibility,
adaptive stopping, full-history fallback, role routing, state reset, failed
generations, raw failure records, and all templates from another working directory.
They make no paid model calls and do not measure reasoning quality.
