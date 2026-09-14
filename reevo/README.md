# ReEvo

Reflective evolution of caller-scored text. The task can describe any candidate
format; the agent never executes candidates or loads a benchmark.

```python
from pathlib import Path
from slick import prompts
from reevo import Config, ReEvo

prompts.TEMPLATE_ROOT = Path("reevo/prompts").resolve()
agent = ReEvo(
    task="Write a concise product description.",
    provider=provider,
    evaluate=evaluate,  # async (candidate: str) -> finite float
    config=Config(max_evaluations=20, initial_size=4, maximize=True),
    seed_candidate="A simple everyday notebook.",  # optional
)
result = await agent.run()  # or run(session=your_session)
print(result.best.candidate if result.best else result.stop_reason)
```

The caller supplies `provider` and `evaluate`. Lower scores win by default;
set `maximize=True` for higher scores. Candidate text is passed through unchanged.
Short reflections compare worse/better pairs; crossover combines them, long
reflections retain fewer than 50 words, and mutation uses the best candidate
including current crossover offspring. Rates determine offspring counts per
population. The random seed controls parent selection.

`run` reads as seed, initialize, select parents, reflect, cross, update memory,
and mutate. Initial generation, crossover, and mutation each have a dedicated
decorated method and template; prompts contain no stage switches.

`max_evaluations` counts all supplied seed and generated candidate attempts,
including blank text and evaluator rejections. Reflection calls are separate.
Invalid seeds stop with `invalid_seed`; no accepted parents or no distinct-score
pairs stop early. Provider failures abort; evaluator `ValueError` rejections are
recorded in `result.individuals`; cancellation propagates. Use a fresh ReEvo
instance for each run. Caller types are trusted; other errors propagate.

Configure Slick's process-global template root before calling this agent.
Independent roots need separate processes. Supplying a Session intentionally
carries conversation history and uses its provider instead of `provider`.

Run offline checks from the repository root:
`optimizer/.venv/bin/python -m unittest tests.test_reevo`.
The former TSP application is archived in
`examples/legacy/2026-09-14-problem-specific.tar.gz`.
