# Optimizer

`Optimizer` proposes a textual candidate, evaluates it, then revises it once for
each supplied feedback item. It returns the best candidate measured in that run.

```python
from pathlib import Path
from slick import prompts
from optimizer import Optimizer

prompts.TEMPLATE_ROOT = Path("optimizer/prompts").resolve()
agent = Optimizer(task="Your task and candidate requirements", provider=provider,
                  evaluate=evaluate)
best = await agent.run(feedback=["Simplify the approach", "Address edge cases"])
```

`evaluate(content: str) -> float` is async and must return finite fitness. Higher
fitness wins unless `maximize=False`. Generated `Proposal` objects contain
nonblank `description` and `content`; the returned `Individual` adds `fitness`.

The first proposal is required to succeed. Every feedback item consumes one
revision attempt; only strict improvements replace the incumbent. Each revision
sees the current best, the actual previous proposals, scores, acceptance decisions
and failures. Invalid JSON, invalid/nonfinite scores and timeouts reject a
revision. Other exceptions propagate. `agent.history` records all turns in memory
and resets on each run. There is no implicit retry or open-ended review loop.

Pass a caller-owned Slick Session with `run(session=...)`, or use the constructor
provider. Calls are sequential, and an instance supports one run at a time.
Configure Slick's process-global template root once before running; concurrent
applications with different roots need separate processes.

Evaluation, any required execution isolation, provider construction and
persistence belong to the caller. This agent never executes candidates. Previous
domain-specific code is in `examples/legacy/2026-09-14-problem-specific.tar.gz`.
From this directory, `python -m pip install -r requirements.txt` installs the
existing local Slick dependency. From the root, run
`python -m unittest tests.test_optimizer` for offline verification.

`run` orchestrates the initial proposal and feedback turns; `_assess` scores
candidates and `_turn` records acceptance. Revision prompts receive the previous
turns as JSON data.

Caller inputs follow the annotated API without blanket runtime type checks.
Generated proposals retain their schema constraints, and evaluation requires finite scores.
