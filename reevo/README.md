# ReEvo

Reflective evolution of caller-scored text. The task can describe any candidate
format; the agent never executes candidates or loads a benchmark.

Implements Section 4 of [ReEvo (NeurIPS 2024)](https://arxiv.org/abs/2402.01145),
using the [official implementation](https://github.com/ai4co/reevo) linked by
the [authors' project page](https://ai4co.github.io/reevo/).

```python
from pathlib import Path
import reevo
from slick import prompts
from reevo import Config, ReEvo

prompts.TEMPLATE_ROOT = Path(reevo.__file__).resolve().parent / "prompts"
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
For code evolution, put the function signature, semantics, constraints, and
required output format in `task`. The evaluator owns parsing, interface checking,
execution isolation, timeouts, and scoring on a fixed validation set (for example,
mean performance over problem instances). Translate expected candidate failures
to `ValueError`. Evaluate the final winner separately on held-out data.

Short reflections compare randomly selected unequal-score worse/better pairs;
crossover combines them, long reflections retain fewer than 50 words, and mutation
uses the best candidate including current crossover offspring. All mutations in
one generation use the same elite snapshot. The next population consists of
crossover and mutation offspring; the all-time elite remains available for
selection even when offspring regress. The random seed controls parent selection,
not the LLM or evaluator.

Default settings follow Table 8: 30 initial generations, population size 10,
100 candidate evaluation attempts, crossover rate 1, and mutation rate 0.5.
Each generation produces `int(population_size * crossover_rate)` crossover
offspring and `int(population_size * mutation_rate)` mutations, capped by the
remaining budget. An optional seed consumes an additional initialization slot
within that same total budget. `short_reflection` and `long_reflection` switch
off the corresponding reflection operations; zero rates disable operators.

For black-box search, supply an opaque task description and set
`Config(black_box=True)`. Its separate reflection prompt asks the model to infer
objective structure from comparative performance. Following official code,
when a seed is provided, only candidates strictly better than that seed can be
parents. Without a seed, all successfully evaluated candidates are eligible.
This option does not redact revealing names or explanations from your task/seed.

Pass `reflector_provider=...` to use a different model for both reflection stages.
Pass `initial_provider=...` for more diverse initialization. Both default to
`provider`. Model choice, temperature, and transport retries belong to those
providers; to match the paper, use temperature 1 for generation/reflection and
1.3 for `initial_provider`. The agent does not mutate provider settings.

`run` reads as seed, initialize, select parents, reflect, cross, update memory,
and mutate. Initial generation, crossover, and mutation each have a dedicated
decorated method and template; prompts contain no stage switches.

`max_evaluations` counts all supplied seed and generated candidate attempts,
including blank text and evaluator rejections. Reflection calls are separate.
Invalid seeds stop with `invalid_seed`; no accepted parents or no distinct-score
pairs stop with `no_valid_individuals` or `no_distinct_parents`. A generation
producing no offspring stops with `no_offspring`; exhausted budgets stop with
`budget`, including zero-budget runs. Provider failures abort; evaluator `ValueError` rejections are
recorded in `result.individuals`; cancellation propagates. Use a fresh ReEvo
instance for each run. Caller types are trusted; other errors propagate.

`result.best` is the best accepted `Individual` (or `None`), and `best_history`
records the incumbent score after each candidate attempt. Individuals preserve
raw text, scores, rejection errors, parent IDs, stage, and generation. Reflection
records preserve pair IDs, raw short reflections, bounded long-term memory, and
`raw_long_term` before truncation (`None` when no long reflection was requested).

Configure Slick's process-global template root before calling this agent.
Independent roots need separate processes. Supplying a Session intentionally
carries conversation history and overrides all three providers. Default calls
are independent and sequential, as in the paper's prompt contexts. No session is
created or shared implicitly.

## Official source correspondence

Reviewed and adapted from commit
[`6dce18257da5e11db2d138e417a2fffc5c72d05f`](https://github.com/ai4co/reevo/tree/6dce18257da5e11db2d138e417a2fffc5c72d05f).
The upstream MIT notice is retained in [LICENSE](LICENSE).

| Official source | Local implementation |
| --- | --- |
| `reevo.py:init_population` | Optional seed evaluation, `initialize`, separate initialization provider |
| `reevo.py:random_select` | Uniform unequal-fitness pairs, elite reinsertion, black-box seed filter |
| `reevo.py:short_term_reflection`, `crossover` | `reflect`, `cross_population`, local pair/crossover templates |
| `reevo.py:long_term_reflection`, `mutate` | `update_reflections`, `mutate_population`, local long/mutation templates |
| `reevo.py:evolve`, `update_iter` | `run`, score direction, population replacement, all-time elite |
| `prompts/common/*.txt` | Task-agnostic adaptations in `prompts/*.j2`, with 20/50-word reflection instructions |
| `cfg/config.yaml` and paper Table 8 | `Config` search defaults |

Deliberate adaptations: candidate strings and an injected evaluator replace
hard-coded Python functions, version renaming, benchmark subprocesses, and Hydra.
Prompts are rewritten for arbitrary artifacts rather than copied verbatim;
they retain relative performance, pairwise hints, accumulated guidance, and
elitist mutation. The search uses Python's RNG and exact uniform pair enumeration
instead of NumPy rejection sampling. Budgets are strictly capped (upstream checks
between full batches and can overshoot); early stops return records rather than
raising. Long memory is capped at 49 words while retaining its raw response.
An available elite can keep a run alive after a failed offspring batch. These
choices make this an algorithm adaptation, not a byte-for-byte reproduction.

Run offline checks from the repository root:
`optimizer/.venv/bin/python -m unittest tests.test_reevo tests.test_prompt_layout`.
Checks use the actual local Slick 0.3.0 prompt decorator with the shared scripted
provider. They exercise search accounting, rejection, both score directions,
black-box selection, memory, elite snapshots, provider routing, and template
resolution from another working directory. No live LLM performance or paper
benchmark reproduction is claimed.
The former TSP application is archived in
`examples/legacy/2026-09-14-problem-specific.tar.gz`.
