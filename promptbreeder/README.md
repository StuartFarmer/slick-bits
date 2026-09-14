# Promptbreeder

Evolve task prompts **and the mutation prompts that modify them**, with an
arbitrary task description, Slick provider, async evaluator, and embedding
similarity callback. All model instructions live in this folder's `prompts/`.

## Sources

- [Fernando et al., Promptbreeder, sections 3–4 and appendices C/D/J/L](https://arxiv.org/abs/2309.16797);
  [ICML publication](https://proceedings.mlr.press/v235/fernando24a.html).
- [Co-author Dylan Banarse's minimal implementation](https://github.com/dylski/promptbreeder/tree/e707e1e71fa89c27c26728b1af44836d700214d9)
  (commit `e707e1e71fa89c27c26728b1af44836d700214d9`, May 2025).
  Its 20 mutation prompts and 20 thinking styles are reused verbatim as default
  seed data in `prompts/seeds.json`, under the included [MIT license](LICENSE).
  Its `ga.py` and `mutator.py` informed pairing, coupled genomes, and the
  evaluator boundary.

The co-author repository calls itself a **minimal implementation inspired by the
paper**. It is not established as the original PaLM experiment release; no such
release was found linked from the publication pages. The full paper governs this
implementation's operator set. The later repository's hypermutation quick-check
acceptance gate and restricted operators are not substituted for the paper's
algorithm.

## Supply your task

```python
from pathlib import Path

import promptbreeder
from promptbreeder import Evaluation, PromptBreeder
from slick import prompts

# Configure once at application startup, independent of the launch directory.
prompts.TEMPLATE_ROOT = Path(promptbreeder.__file__).resolve().parent / "prompts"

# evaluate(unit) -> Evaluation(score, correct_workings)
# similarity(a, b) -> BERT embedding cosine similarity; both callbacks are async.
agent = PromptBreeder(task_description, mutation_provider, evaluate, similarity)
result = await agent.run(population_size=10, generations=5, context_size=3, seed=42)

best = result["best"]
outputs = await agent.predict(best.unit, new_input, provider=inference_provider)
answer = outputs[-1]
```

`evaluate` receives an immutable `Unit` with a tuple of task `prompts`, one
`mutation` prompt, a tuple of verified `context` examples, and elite `lineage`.
It must score the **whole prompt sequence with that exact context**, returning
`Evaluation(score, correct_workings=())`. Higher finite, nonnegative scores win;
convert losses to rewards in your evaluator. Correct workings are optional,
self-contained input/output demonstrations actually generated during evaluation
and verified by your scorer, never merely supplied reference labels.

`predict` returns all generated outputs in order. It prepends demonstrations,
executes prompt 1 plus the input, then appends each remaining prompt to the
accumulated transcript and generates again. It never receives an expected answer.
Pass an inference provider configured separately from the mutation provider.
For different sampling settings at each stage, your evaluator can call `start`
and `continue_solution` with different providers directly.

A complete evaluator wiring example, using your own training pairs and checker:

```python
import random

async def optimize(task, mutation_provider, inference_provider, train, is_correct, similarity):
    # train: sequence of (input_text, expected_answer)
    # is_correct(input_text, generated_answer, expected_answer): async -> bool
    batches = random.Random(7)
    context_inputs = {}

    async def evaluate(unit):
        # Do not score an input whose answer is already in the few-shot context.
        shown = {context_inputs[working] for working in unit.context}
        eligible = [(question, target) for question, target in train if question not in shown]
        batch = batches.sample(eligible, min(100, len(eligible)))
        if not batch:
            raise ValueError("No training examples remain outside the few-shot context")
        correct = []
        for question, target in batch:
            outputs = await agent.predict(unit, question, provider=inference_provider)
            if await is_correct(question, outputs[-1], target):
                working = "Input: " + question + "\nOutputs:\n" + "\n".join(outputs)
                context_inputs[working] = question
                correct.append(working)
        return Evaluation(len(correct) / len(batch), tuple(correct))

    agent = PromptBreeder(task, mutation_provider, evaluate, similarity)
    return await agent.run(population_size=10, generations=5, context_size=3)
```

The application owns data loading, random training-batch sampling (the paper uses
100 examples), correctness checks, embedding computation/caching, provider
configuration, and any required execution isolation. Keep validation/test data
outside evolution; evaluate the selected `best.unit` on held-out data afterwards.
For objectives other than accuracy, replace the example evaluator's aggregation.
No dataset, model name, answer parser, generated-code execution, or provider SDK
is hard-coded into this optimizer.

## Evolution and budgets

Defaults: 50 units, two sequential task prompts, 20 generations, zero-shot
(`context_size=0`). Set `context_size=3` for few-shot evolution. Supply custom
`mutations` and `styles` to `run()` to replace the author repository's seed pools,
including the paper's full 56/39 pools if desired.

Each generation shuffles the population indices into disjoint pairs. Stored
fitness selects a winner; a mutated copy replaces the loser even if its measured
score is worse. Replacements become visible immediately to later events. Ties
favor the first member of the shuffled pair; odd populations have one random bye.
The winner's evaluated genome remains unchanged.

One of these nine operations is sampled uniformly per event:

| Operation | Mechanism |
| --- | --- |
| `zero` | Generate fresh hints from the task description; extract the first hint |
| `first` | Apply the unit's mutation prompt to one task prompt |
| `distribution` | Continue a shuffled population list, excluding similarity > 0.95 |
| `ranked` | Continue the filtered list in ascending fitness order, retaining the paper's deliberately contradictory descending-order heading |
| `lineage` | Continue the chronological elite ancestry for the selected prompt slot |
| `hyper_zero` | Generate a mutation instruction from task and thinking style, then apply it |
| `hyper_first` | Improve the mutation instruction, then apply it |
| `lamarckian` | Infer an instruction from verified successful workings |
| `context` | Shuffle the order of verified few-shot examples |

An additional 10% crossover replaces the selected task prompt with a random
prompt from another individual, selected in proportion to fitness (uniform if
all donor scores are zero). Mutation prompts do not cross over. EDA uses supplied
embedding similarities; ranked filtering retains the fitter of near duplicates.
Fitness numbers are not exposed to the LLM.

Verified workings fill offspring contexts; an already full context receives at
most one new example from the parent's evaluated batch. With 10% probability,
the context is resampled from available verified workings. Context changes are
made **before offspring evaluation**, keeping archived scores attached to the
exact prompts/context that produced them. Zero-shot runs still retain successful
workings for Lamarckian induction.

For N units and G generations, successful completion invokes the evaluator
`N + G * floor(N / 2)` times. Explicit `tournaments=T` overrides the generation
budget, for `N + T` evaluations. Initialization uses `N * prompt_count` model
calls; mutation uses zero (context), one, or two (hypermutation) calls per event,
plus evaluator-owned inference calls. Default evolution therefore makes 550
fitness evaluations; it does not automatically stop on a plateau.

Results contain `best` (highest measured fitness across the run), `population`,
`history` (all valid evaluated individuals), `evaluations`, `operators`, `pairs`,
and `fallbacks`. The archive is a best observed training measurement, not a
statistical guarantee. `agent.raw_responses` retains mutation-generation output
before validation; provider-side logs can capture transport attempts.

## Explicit choices and limits

- Text generation preserves the paper's continuation delimiters; there is no
  JSON wrapper. Surrounding whitespace is stripped and blank generated prompts
  raise `ValueError`. The zero-order hint parser accepts numbered/bulleted
  multiline hints; unlisted prose is treated as one hint. Zero-order
  hypermutation includes an explicit request for a mutation instruction.
- One randomly selected task-prompt slot changes per event. Lineage records
  complete elite prompt sequences; lineage generation uses the matching slot.
- The paper says nine uniformly sampled operators while describing crossover as
  an additional event. Here context shuffling is the ninth, with crossover
  applied separately; zero-shot context mutations can be no-ops.
- Whole-context resampling is underspecified in section 3.2.5. This implementation
  selects each available verified working with probability `1/context_size`,
  caps the list at that size, and permits an empty list.
- Missing Lamarckian evidence triggers an explicitly recorded 50:50 zero/first
  mutation fallback (using the paper's ablation replacement convention).
- Optional appendix J stagnation interventions (random-character prefixes,
  fitness sharing, evolving sampling temperatures) are not included. Provider
  sampling settings remain caller-owned. This is an implementation of the core
  algorithm, not a reproduction of the benchmark results.
- Errors from providers, scorers, similarities, and generated-output validation
  propagate without retries; no invalid score becomes fitness zero.
  `evaluations` counts evaluator invocations even when they fail; `history`
  includes valid results only. Operator/pair records include attempted events.
  Use a fresh agent per run. There are no hidden sessions.
- Slick's template root is process-global: configure it once and use separate
  processes for simultaneous algorithms requiring different roots.

This completes the earlier local mechanism port deliberately: plain text replaces
JSON generation, disjoint stored-fitness tournaments replace repeated independent
rescoring, full lineage replaces first-prompt-only history, and context resampling,
author seed defaults, sequential inference, and an evaluation archive are added.

## Checks

From the repository root:

```sh
rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_promptbreeder -v
rtk proxy ../slick/.venv/bin/ruff check promptbreeder tests/test_promptbreeder.py
rtk proxy ../slick/.venv/bin/ruff format promptbreeder tests/test_promptbreeder.py --check
```

The tests use the shared `tests/providers.py` scripted provider. They check
evolution decisions, all operation boundaries, failures, and template loading
from other working directories. They do not measure real-model fitness or
establish the paper's reported gains.
