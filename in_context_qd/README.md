# In-context Quality-Diversity

A problem-agnostic implementation of [Large Language Models as In-context AI
Generators for Quality-Diversity](https://arxiv.org/abs/2404.15794), Lim, Flageat,
and Cully (ALIFE 2024). `InContextQD` owns the archive, prompt operations, and
search loop. Supply a Slick provider, initial candidates, an async evaluator,
and a grid defining the behaviors you want to diversify.

## Use it on your problem

Candidates are strings: designs, prompts, programs, or encoded parameter vectors.
Your evaluator returns `Evaluation(fitness, features)` with **larger fitness
better** and features in the grid's original units. Negate a loss to minimize it.
The evaluator owns candidate decoding, feasibility checks, timeouts, and any
required execution isolation. The optimizer never executes generated code.

```python
from pathlib import Path

from slick import prompts
import in_context_qd
from in_context_qd import Config, Evaluation, Grid, InContextQD


async def optimize(provider, evaluate, initial_candidates):
    # Configure once at application startup, before any calls or rendering.
    prompts.TEMPLATE_ROOT = Path(in_context_qd.__file__).resolve().parent / "prompts"
    agent = InContextQD(
        task=(
            "Propose one candidate for my task. Return only the candidate text. "
            "Examples have fitness : encoded behavior features : candidate. "
            "Each feature is encoded from its lower/upper bound to 0/1000. "
            "Complete the final row's missing candidate."
        ),
        provider=provider,
        evaluate=evaluate,  # async (str) -> Evaluation
        grid=Grid(bounds=((0, 1), (0, 10)), bins=(20, 20)),
        config=Config(batch_size=10, context_size=30, seed=42),
    )
    result = await agent.run(initial_candidates, generations=100)
    return result
```

Replace the task description and grid with your actual task and behavior ranges.
Sample initial candidates from your domain; they are evaluated rather than
trusted as pre-scored elites. The sampler is outside the algorithm so it can
generate whatever your domain requires. For multiline artifacts, use a reversible
single-line serialization in the task, seeds, and evaluator to preserve the
paper's one-example-per-line convention.

`result.archive` maps cell-index tuples to `Elite(candidate, fitness, features)`.
`coverage` is the occupied fraction, `qd_score` the raw sum of elite fitnesses,
and `max_fitness` the best measured fitness. `history` includes initialization and
each generation. Negative fitness is supported; filling an empty cell with a
negative score can decrease the raw QD score even though coverage improves.

## Numeric black-box problems

The optional `NumericSpace` implements bounded parameter encoding as integer
CSV. It is independent of the task and maps each nonconstant coordinate to
`[0, precision]`, with `precision=1000` by default. Constant coordinates encode
as zero. The evaluator decodes generated strings to the original units.

```python
import random

from in_context_qd import NumericSpace


async def optimize_vectors(provider, objective, parameter_bounds, feature_bounds, bins):
    # objective: async (tuple[float, ...]) -> Evaluation, in original units.
    prompts.TEMPLATE_ROOT = Path(in_context_qd.__file__).resolve().parent / "prompts"
    space = NumericSpace(tuple(parameter_bounds))

    async def evaluate(candidate):
        return await objective(space.decode(candidate))

    rng = random.Random(42)
    initial = [space.sample(rng) for _ in range(100)]
    agent = InContextQD(
        task="",  # Bare numeric pattern completion, as in the paper.
        provider=provider,
        evaluate=evaluate,
        grid=Grid(tuple(feature_bounds), tuple(bins)),
    )
    return await agent.run(initial, generations=1000)
```

For a chat model, supply task instructions requesting only the comma-separated
integer parameters, without a repeated fitness/features prefix. Provider/model
selection and decoding settings remain with the caller. The paper used
Mistral-7B-v0.1; this code does not download a model or construct a provider.

## Algorithm and options

1. Evaluate seeds and retain the strictly best candidate in each measured cell.
2. Select a batch of target cell centroids. `feature_query="empty"` samples empty
   cells without replacement if there are at least `batch_size`; otherwise it
   samples the entire grid with replacement. `"uniform"` always uses the latter.
3. Independently sample up to `context_size` occupied elites without replacement
   for each query, using the archive as it stood at the start of the generation.
4. Order context by `order="distance"` (farthest to nearest, Euclidean distance
   in original feature units), `"fitness"` (ascending), or `"random"`.
5. Query a fitness above the context maximum and the selected features, generate
   one candidate, evaluate it, and compete in its **measured** cell.

`Config.template` selects separate local templates:

| Template | Context row | Query |
| --- | --- | --- |
| `qd` (default) | `fitness : features : candidate` | `fitness : features :` |
| `fitness` | `fitness : candidate` | `fitness :` |
| `feature` | `features : candidate` | `features :` |
| `lmx` | `candidate` | No fitness/feature query |

Feature values are encoded as integers only for prompting; cell assignment and
context distances use the original measured values. Each feature's encoding
precision is controlled by `Grid.precision`. This is separate from archive bins.
Context size zero uses `initial_fitness` (default 1.0) and no examples.

The context fitness target is
`best + max(abs(best) * improvement, minimum_improvement)`, with defaults 0.2
and 1e-6. This matches a 20% increase for ordinary positive fitness and extends
the paper's rule to negative/zero fitness. The quantization precision, positive
floor, zero-context target, sampling replacement details, and one-generation
default are explicit implementation choices where the paper does not specify
an exact convention. Set `generations=1000` for its experimental iteration count.

## Failures, budgets, and ownership

- Each generation consumes exactly `batch_size` provider calls unless an
  unexpected error aborts. Rejected candidates consume attempts; no hidden
  repair, mutation, replacement sampling, or model retry occurs.
- `evaluations` counts evaluator calls, including seeds and rejected evaluations.
  Blank generations are rejected before evaluation. `model_calls` counts attempted
  provider calls. `attempts` records raw output, query/context, measured values,
  acceptance, and errors; records remain on the agent if the run aborts.
- Raise `CandidateRejected` from your evaluator for expected infeasibility or
  invalid generation. Nonfinite fitness, invalid feature dimensions, and features
  outside the grid are rejected. Numeric decoding rejects malformed, wrong-length,
  and out-of-range vectors. Ties keep the incumbent. Other exceptions propagate.
- There must be at least one surviving seed to start search. With
  `generations=0`, an empty archive can be returned (`max_fitness=None`).
- One active run per instance; `run()` resets the archive, RNG, counters, and
  records. No mutable Slick Session or conversational history is shared.
- Caller settings are trusted. Use finite, increasing feature bounds, matching
  positive bin counts, nonnegative context sizes, and positive batch sizes.
  A rectangular grid has a product-of-bins cost; keep its total size practical.
- Slick's template root is process-global. The absolute path above works from
  different launch directories. Configure it outside the library and do not
  change it concurrently for different algorithms.

## Sources and verification

Algorithm source: [paper, Method and template ablations](https://arxiv.org/html/2404.15794v1).
No official implementation link was found in the paper, the
[first author's publication page](https://limbryan.github.io/), or the public
repository lists of [the author](https://github.com/limbryan) and
[the authors' lab](https://github.com/adaptive-intelligent-robotics), checked on
2026-09-14. This is an independent implementation from the supplied paper,
not a port of an unverified third-party repository. No official code was available
to reuse from those sources.

The text-candidate interface and caller task prefix are deliberate generalizations;
bare numeric CSV with an empty task preserves the paper's delimiter-based prompt
structure. Training, benchmark environments, baseline experiments, and paper
performance reproduction are outside this algorithm implementation.

Run offline checks from the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_in_context_qd
../slick/.venv/bin/ruff check in_context_qd tests/test_in_context_qd.py
```

These use the shared scripted provider to check search decisions, numeric
decoding, failure accounting, and actual Slick prompt calls. They do not establish
optimization performance or reproduce the paper's reported results.
