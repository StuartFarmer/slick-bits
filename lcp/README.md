# LCP

Task-agnostic paper reconstruction of **Learning from Contrastive Prompts: Automated
Optimization and Adaptation** (Li et al., 2024), [paper, §§2.2–2.3](https://arxiv.org/html/2409.15199v1).
No author-maintained implementation was located; this is not an official-code port.

The optimizer explains each failed training case, repeatedly samples explanations
to induce diverse instructions, scores and accumulates those instructions in a
permanent pool, then contrasts the highest and lowest scoring instructions to
produce the next iterate. The next iterate may score below the best-ever prompt;
both are returned. Adaptation mode retains only target-model failures that the
source model answered correctly.

```python
from pathlib import Path
from slick import prompts
from lcp import LCP

prompts.TEMPLATE_ROOT = Path("/absolute/path/to/lcp/prompts")
agent = LCP(task, provider, evaluate, failures)
result = await agent.run(initial_prompt, diversity=10, top_k=3)
instruction = result["best"].prompt
```

`evaluate(text)` asynchronously returns a finite higher-is-better score on the
fixed search split. `failures(text)` asynchronously returns incorrect training
observations as dictionaries, including input, expected answer, and observed
response. Adaptation additionally requires a boolean `source_correct` field. The
caller performs both task-model inference and any source-model comparison.

The templates are task-neutral rewrites of the paper stages, not verbatim
experimental prompts. Candidate text is stripped and deduplicated, with scores
cached for the run; use a fixed evaluator. If the pool has fewer than `2 * top_k`
unique prompts, equally sized disjoint high/low groups are used. A one-prompt pool
ends the search rather than contrasting a prompt with itself. This is an explicit
generated-collapse policy. Sampling is seeded in Python; provider temperature and
model sampling are caller-owned. Empty generated text and nonfinite measurements
raise; transport, parsing, and evaluation failures propagate without retry.

The result includes the best and current candidates, complete unique archive,
contrast history, optimizer-call count, and actual evaluation count. Dependencies
are Slick and its normal dependencies plus the standard library. No dataset,
embedding model, benchmark runner, or provider credentials are bundled. Tests use
the shared scripted provider to check ranking, archive reuse, adaptation filtering,
collapse termination, and invalid output. They do not reproduce paper results.
