# PACE

This is a **paper-based reconstruction**. No official implementation was located
in the paper, the survey bibliography, or searches for its title/authors. It is
not presented as a port of an unavailable codebase.

`PACE(task, provider, evaluate, actor_provider=None)` samples demonstrations,
generates actor responses under the current instruction, asks a critic to review
each response against ground truth, and aggregates those critiques into a revised
instruction. Each candidate receives its own sampled actor/critic evidence;
all candidates in one iteration start from the same current instruction.

```python
from pace import PACE, Example
result = await PACE(task, provider, evaluate).run(
    initial_prompt, [Example(input_text, target_text)],
    iterations=1, candidates=2, actors=4,
)
```

`evaluate(prompt)` returns finite validation fitness, higher is better. Optional
`actor_provider` separates the task model from the optimizer. Demonstrations are
training data; held-out evaluation is caller-owned. The best candidate becomes
the current prompt, while a separate best-ever result protects earlier winners.
An unchanged chosen instruction ends iteration. Results retain actor responses,
ground truths, critiques, and separate actor/optimizer/evaluator counts. An empty
initial prompt supports the paper's generation-from-scratch setup.

Sources inspected: [original paper](https://arxiv.org/abs/2308.10088),
[final ACL paper](https://aclanthology.org/2024.findings-acl.436/), Algorithm 1,
Equations 1-3, and Appendix B's actor, critic, and update templates.

Adaptations: local Slick templates condense the appendix's wording and include
explicit caller task context. Python RNG samples with replacement. The paper's
experimental multi-candidate setting is exposed as independent complete candidate
trials; actor and critic calls run sequentially. The paper leaves convergence
unspecified; this port uses unchanged text plus the explicit iteration budget.
Scalar evaluation replaces benchmark-specific scoring. No generic rewrite-only
loop is substituted for the actor/critic evidence generation.

Configure Slick's process-global template root before use:

```python
from pathlib import Path
from slick import prompts
import pace
prompts.TEMPLATE_ROOT = Path(pace.__file__).parent / "prompts"
```

Requires the repository's Slick installation. Provider/model construction,
benchmarks, credentials, persistence, and execution isolation belong to callers.
All model calls are sequential, and provider/evaluator failures propagate without
retries. Blank optimizer prose raises before scoring. Use one active run per
agent instance. Deterministic tests check algorithm mechanics, not benchmark
performance or reproduction of the published numerical results.
