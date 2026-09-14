# EvoPROMPT

`EvoPrompt` evolves caller-supplied task instructions using a genetic algorithm
(GA) or differential evolution (DE). The class owns its task context, decorated
generation methods, and async selection loop. Supply a Slick provider and your
own async evaluator; there is no built-in dataset, evaluator, or runner.
`run` composes initialization, a GA or DE generation phase, and score snapshots.

```python
from pathlib import Path

from slick import prompts

import evoprompt
from evoprompt import EvoPrompt

# Configure Slick's process-global template root once before rendering.
prompts.TEMPLATE_ROOT = Path(evoprompt.__file__).parent / "prompts"

agent = EvoPrompt(task, provider, evaluate)
result = await agent.run(initial_prompts, algorithm="de", iterations=10)
best_instruction = result["best"]["prompt"]
```

`task` describes the problem and output requirements. `evaluate(instruction)`
must asynchronously return finite, nonnegative fitness, with higher scores
preferred. Set `cache=False` when repeated evaluation is intentionally noisy.
Generated instructions retain the textual `<prompt>...</prompt>` protocol:
decorated methods check that exactly one nonempty final prompt is present before
evaluation. Invalid output raises `ValueError` containing the raw response.
Provider and evaluator failures propagate without automatic retries.

- GA requires at least two members, samples parents with replacement using
  fitness-proportional roulette, and retains the best N parents and offspring.
  All-zero fitness falls back to uniform sampling. Stable ties prefer incumbents.
- DE requires at least three members, selects distinct donor indices excluding
  the target, and uses the current generation's best prompt. Each trial replaces
  its own target only on strict improvement. Every trial reads the same generation.
- A larger `population_size` fills missing members with checked variations of
  the supplied initial prompts. `seed` controls the local random generator.

The result includes the best and initial-best prompts, final population,
per-generation score history, generation/evaluation counts, and cache hits.
Pass `session=your_session` to `run` for an intentional sequential conversation;
otherwise every generation uses the injected provider independently. Different
applications need separate processes to use different template roots concurrently.
Each run resets the agent's search state; use one run at a time per instance.
Caller types are trusted; population size, generated instructions, and fitness
remain checked algorithm boundaries.

Run offline checks from the repository root with Slick installed:
`python -m unittest tests.test_evoprompt`.
The previous benchmark CLI and demo dataset are preserved in
`examples/legacy/2026-09-14-problem-specific.tar.gz`; historical `runs/` remain in place.
