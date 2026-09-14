# HMAW

Problem-agnostic Slick implementation of Liu et al., *Towards Hierarchical
Multi-Agent Workflows for Zero-Shot Prompt Optimization*. The CEO generates
instructions for the Manager, the Manager generates instructions for the Worker,
and the Worker answers. All three stages receive the original query verbatim;
each downstream stage receives only its immediate predecessor's instructions.

## Use

```python
from pathlib import Path

from slick import prompts

import hmaw

# Configure once at application startup, before any prompt calls.
prompts.TEMPLATE_ROOT = Path(hmaw.__file__).resolve().parent / "prompts"

# Supply any configured Slick provider; put the problem and context in task.
agent = hmaw.HMAW(task="Your question, source material, and constraints", provider=provider)
result = await agent.run()
print(result.answer)
print(result.optimized_prompt)  # Exact prompt sent to the Worker.
```

`task` can contain arbitrary text: questions, code, documents, conversation history,
and output requirements. `provider` supplies all three generations by default.
Optional `ceo_provider=` and `manager_provider=` override those roles; the Worker
always uses `provider`. Configure models, sampling, token limits, and transport
retries on the providers. The official runner defaults to temperature 0.2 and
seed 0; this implementation imposes neither setting.

`Result` contains the unmodified `answer`, `ceo_instructions`,
`manager_instructions`, `optimized_prompt`, and logical `calls` count. A successful
run uses exactly three calls. There is no training, few-shot dataset, evaluator,
baseline generation, task execution, or refinement loop inside HMAW. If needed,
evaluate the answer separately with your own async evaluator:

```python
score = await evaluate(result.answer)
```

Use the existing environment with the adjacent Slick checkout (verified
`slick-ai` 0.3.0); no new dependencies are required. Launch from the repository
root or add it to `PYTHONPATH`. The template root is process-global: configure it
once and use separate processes for concurrent algorithms with different roots.
Importing this package does not set the root.

## Failures and state

The three generations have no shared Session or implicit conversation history.
`run()` clears `agent.calls` before starting; use one run at a time per instance.
Each record holds its operation, rendered prompt, raw response, and any error.
Responses are recorded before generated-output validation. Blank text, tool
requests, and provider exceptions abort the run immediately; completed and failed
call records remain inspectable. Errors propagate without retries or fallback.
Caller inputs are trusted. These checks validate usable text, not answer quality.

## Official source and adaptations

The [project page](https://liuyvchi.github.io/HMAW_project/) links to the
[official implementation](https://github.com/liuyvchi/HMAW). This port uses
commit `e3968263f04e1988722b58a46e2128fe64ac933f`, specifically
[`get_m2prompting_output` in prompter.py](https://github.com/liuyvchi/HMAW/blob/e3968263f04e1988722b58a46e2128fe64ac933f/prompter.py#L218).
The three local Jinja templates were extracted directly from that function's
string literals. Its stage ordering, company roles, prompt wording, delimiters,
and two query skip connections are retained. The upstream MIT notice is in
[LICENSE](LICENSE). Paper: [arXiv:2405.20252](https://arxiv.org/abs/2405.20252),
Section 3.2 and Appendix A.1.

Deliberate adaptations:

- Remove Python source indentation from the templates; do not apply upstream's
  `clean_system_message` to interpolated text. Its repeated-space collapse can
  corrupt code indentation and other task data.
- Return Worker text verbatim instead of splitting at the last
  `**Response for the User**:` occurrence, which could truncate a valid artifact.
- Replace synchronous, hardcoded OpenAI clients and key-file reads with injected
  Slick providers. Each role remains one independent text call.
- Reject blank output and retain raw failure records. Expose the actual optimized
  prompt as well as both intermediate instructions and the answer.
- Keep benchmark-specific formatting (including the math final-result suffix),
  scoring, datasets, and infrastructure outside the algorithm. Extra-layer and
  alternate-context ablations are outside this default three-layer implementation.

This is an algorithm implementation, not a reproduction of the paper's reported
preference scores, accuracy, or latency.

## Checks

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_hmaw
../slick/.venv/bin/ruff check hmaw tests/test_hmaw.py
../slick/.venv/bin/ruff format --check hmaw tests/test_hmaw.py
```

Offline tests use the shared `tests.providers.ScriptedProvider` through real Slick
rendering. They check stage order, verbatim skip connections, provider routing,
independent runs, failure propagation and raw records, answer preservation, and
all templates from another working directory. They make no paid model calls.
