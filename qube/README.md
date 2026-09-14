# QUBE

Quality-uncertainty evolution of caller-scored text. Candidates with equal
behavior signatures share a cluster; islands select parents using observed
child quality (maintained as an incremental mean) plus an exploration bonus.
Higher scores win.

```python
from pathlib import Path
from slick import prompts
from qube import Evaluation, QUBE

prompts.TEMPLATE_ROOT = Path("qube/prompts").resolve()
agent = QUBE(
    task="Write a concise product description.",
    provider=provider,
    evaluate=evaluate,  # async (candidate: str) -> Evaluation
    seed_candidate="A simple everyday notebook.",
    samples=20,
    islands=4,
    reset_interval=10,
)
result = await agent.run()  # or run(session=your_session)
print(result.best.candidate)
```

The caller supplies `provider` and `evaluate`; the evaluator returns
`Evaluation(score=finite_float, signature=(finite_float, ...))`. Signatures must
be nonempty tuples with the seed's dimension. Their meaning belongs to the
caller. Candidate text is passed through unchanged, never executed here.

`run` evaluates the seed once, then makes at most `samples` generation calls.
Rejected candidates consume samples but do not update parent offspring quality.
A shared parent cluster receives one visit per sample. `k` controls exploration,
`temperature` controls the preference for shorter candidates within clusters,
and `seed` controls random selection. Every `reset_interval` samples the weaker
half of islands are replaced from stronger islands; zero disables resets.

`run` separates initialization, parent selection, generation, assessment,
recording, and periodic resets. Seed evaluation and provider failures abort.
Evaluator `ValueError` rejections and invalid candidate text are recorded in
`result.samples`. Cancellation propagates.
`result.best` retains the best candidate across island resets, while
`result.accepted` and `result.resets` expose acceptance and replacement records.
Use a fresh QUBE instance for each run. Caller types are trusted; other errors
propagate.

Configure Slick's process-global template root before calling this agent.
Independent roots need separate processes. A supplied Session intentionally
carries conversation history and uses its own provider.

Run offline checks from the repository root:
`optimizer/.venv/bin/python -m unittest tests.test_qube`.
The former benchmark and Docker application is archived in
`examples/legacy/2026-09-14-problem-specific.tar.gz`.
