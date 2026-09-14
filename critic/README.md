# CRITIC

Problem-agnostic implementation of Algorithm 1 from **CRITIC: Large Language
Models Can Self-Correct with Tool-Interactive Critiquing** (Gou et al., ICLR 2024).
The caller supplies the task, criteria, demonstrations, provider, and external
tools. The algorithm generates an output, verifies it through tool interaction,
then corrects it using the critique and actual tool results.

```python
from pathlib import Path

import critic
from slick import prompts

# Configure once at application startup; Slick's template root is process-global.
prompts.TEMPLATE_ROOT = Path(critic.__file__).resolve().parent / "prompts"

async def solve(provider, input_text, lookup):
    # lookup is your annotated async function, with a descriptive docstring.
    agent = critic.CRITIC(
        task="Answer using the supplied source collection.",
        provider=provider,
        tools=[lookup],
        criteria="Check factual accuracy, relevance, and source support.",
        examples="",  # Supply task-specific few-shot demonstrations here.
    )
    return await agent.run(input_text, max_iterations=3)

# result.output: latest answer
# result.history: outputs, critiques, and tool requests/results for each verification
# result.stop_reason: "verified", "budget", or "unchanged"
```

Tools can be ordinary annotated sync/async functions or `slick.tools.Tool` objects.
Use async wrappers for blocking tools. Tool selection and follow-up queries happen
inside a fresh Slick `Session` for each verification. The provider must support
Slick's `acall(context, tools=..., tool_results=...)` interface and tool requests.
Search, test execution, simulation, scoring, and retrieval all fit the same API.
The tools own credentials, permissions, timeouts, caches, and any execution isolation.

`initial_output=` skips generation and allows correction of an existing artifact.
`criteria=` defines what acceptance means. `examples=` supplies demonstrations to
all three operations; the local templates can be edited for more specific prompts.
Generated artifacts remain text and retain whitespace. Critiques use explicit JSON
with a strict boolean and nonblank feedback; the model still supplies the judgment.

## Budgets and failures

- `max_iterations=N` permits N verification/correction cycles. As in Algorithm 1,
  the final correction is returned without another verification. Zero returns the
  initial output, generating it if necessary.
- Each verification allows eight provider turns: up to seven rounds of tool
  requests and one final critique. A round can contain multiple tool calls; this
  is a conversation-turn budget, not an exact individual-tool-call budget.
  Exhaustion raises with pending work left unexecuted in `agent.verification_session`.
- `unchanged_patience=2` optionally stops after two consecutive byte-identical
  corrections, following the official QA script's stagnation rule. It is disabled
  by default. `unchanged` and `budget` do not mean the output is verified.
- A `correct=true` critique requires at least one successful, nonblank tool result.
  This prevents acceptance based solely on model self-feedback; it does not prove
  that the evidence is relevant or that the model interpreted it correctly.
- Tool execution errors become visible feedback through Slick. Malformed/blank
  generation, transport errors, and exhausted conversations propagate without
  automatic retries or silently accepting the current answer. `agent.output`,
  `agent.history`, raw `agent.calls`, and the latest session preserve partial work.
  `result.calls` counts provider calls, including tool-interaction turns.
- Runs reset state. Use one active run per agent, and separate processes for
  simultaneous algorithms that need different Slick template roots.

## Official implementations used

This is an intentional generic adaptation of the paper and the authors' code,
not a reproduction of the benchmark prompts or reported scores:

- [Microsoft's CRITIC source](https://github.com/microsoft/ProphetNet/tree/master/CRITIC):
  the [QA loop](https://github.com/microsoft/ProphetNet/blob/master/CRITIC/src/qa/critic.py)
  informed interleaving evidence and the optional two-unchanged stopping rule;
  the [program loop](https://github.com/microsoft/ProphetNet/blob/master/CRITIC/src/program/critic.py)
  informed separate critique and correction calls with execution feedback;
  the [toxicity loop](https://github.com/microsoft/ProphetNet/blob/master/CRITIC/src/toxicity/critic.py)
  was checked for its threshold and rejection behavior.
- [The authors' web tools](https://github.com/ZubinGou/llm-agent-web-tools):
  `official_tools.official_google` directly wraps its `Search.search` implementation,
  preserving its cache use and selecting a result by rank, with a 400-character
  evidence excerpt by default. It does not reimplement the crawler.

To use the official search tool, install that repository's requirements in the
application environment and put its checkout on `PYTHONPATH`, as its README's
`src.tools...` import requires. Then pass its engine to the included adapter:

```python
from src.tools.web_tools.core.engines.google import Search
from critic.official_tools import official_google

search = official_google(Search(proxy=None))
agent = critic.CRITIC(task, provider, tools=[search], criteria=criteria)
result = await agent.run(input_text)
```

The adapter creates a worker event loop because the official crawler calls
`run_until_complete` internally. Cancellation does not terminate its worker;
configure network deadlines in the caller's tool environment. No crawler or
benchmark dependencies are required by the core algorithm.

The generic version uses structured critique judgments and native Slick tool
requests instead of parsing the QA script's search delimiters. It follows Algorithm
1's correctness stop, rather than always revising as some benchmark scripts do.
Task-specific answer extraction, interpreter-result equality, toxicity thresholds,
oracle labels, and score-based rejection remain outside this algorithm. Express
task checks through tools and criteria; the core does not assume scores or labels.

## Validation

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_critic
../slick/.venv/bin/ruff check critic tests/test_critic.py tests/providers.py
```

Checked against the adjacent Slick 0.3.0 source. Tests exercise real local tools
with the shared scripted provider, structured parsing, raw failure records,
iteration and conversation limits, evidence propagation, state reset, adapter
event-loop handling, and rendering from another working directory. They validate
algorithm plumbing; they do not establish benchmark improvement or live search
availability. No paid model calls or generated-code execution are used.
