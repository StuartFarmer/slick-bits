# GEPA

`GEPA` optimizes a dictionary of named prompts for any system you can execute
and score. It implements reflective mutation, instance-wise Pareto selection,
candidate ancestry, and optional system-aware crossover from **GEPA: Reflective
Prompt Evolution Can Outperform Reinforcement Learning**. Model weights and your
system's control flow stay under your evaluator's control.

```python
from pathlib import Path

from slick import prompts

import gepa
from gepa import GEPA, Evaluation

# Configure once at application startup, before any reflection calls.
prompts.TEMPLATE_ROOT = Path(gepa.__file__).resolve().parent / "prompts"

async def optimize(task, reflection_provider, evaluate, seed_prompts, train, validation):
    agent = GEPA(task, reflection_provider, evaluate)
    return await agent.run(
        seed_prompts,      # e.g. {"planner": "...", "writer": "..."}
        train,
        validation,
        budget=1000,
        minibatch_size=3,
        max_merges=5,      # 0 (default): GEPA; 5: GEPA+Merge
        seed=0,
    )

# evaluate is your async callback:
# evaluate(prompts: Mapping[str, str], instance: YourInstance) -> Evaluation
# Return, for example:
# Evaluation(
#     score=0.75,
#     output="The system's final response",
#     feedback="Three of four checks passed; the last check failed because ...",
#     trace="Module inputs/outputs, tool calls, compiler diagnostics, ...",
#     module_feedback={"planner": "The plan missed a required step ..."},
# )
# result = await optimize(task, reflection_provider, evaluate,
#                         seed_prompts, train, validation)
# optimized_prompts = result["best"].prompts
```

Use one key for a single prompt, or any number of keys for a compound system.
Your callback receives a fresh prompt dictionary and one instance from either
dataset. Run the complete system with those prompts, compute the score, and
return `Evaluation`. Scores must be finite; larger is better. Negate a cost for
minimization. The optimizer places no restrictions on instance types; training
instances are displayed to reflection with `str(instance)`. Give custom objects
an informative string representation, or use ordinary strings/dictionaries.

The callback owns model configuration, tools, metric computation, retries,
execution isolation, and any domain knowledge surfaced in feedback. It may use
Slick or any other system. Generated code is never executed by this optimizer.
Every callback invocation counts as one rollout, regardless of the number of
internal model/tool calls. Reflection calls have their own counter and are not
included in the rollout budget.

## Search behavior

1. Evaluate the seed on the full validation set.
2. Collect all candidates tied for the best score on each validation instance.
   Following the official code, remove the lowest-mean redundant candidates
   whose wins are covered by other remaining candidates. Sample a survivor
   proportionally to its number of instance wins.
3. Rotate to the selected candidate's next module. Evaluate the parent on a
   random training minibatch; reflect on its current instruction, inputs,
   outputs, full execution traces, score, and global/module feedback.
4. Replace only that module and evaluate the child on the **same minibatch**.
   A strict improvement earns full validation and admission to the pool.
   Children inherit their parent's next-module position.
5. When enabled, after accepting a mutation, try to merge complementary
   frontier candidates with a common ancestor and no direct ancestry. The
   ancestor must score no higher than either parent. Preserve changes from one
   branch where the other kept the ancestor's prompt; resolve conflicting
   changes in favor of the higher-mean parent, randomly breaking ties.
   Screen merges on up to five validation instances drawn across each parent's
   wins and ties. Accept a score at least as high as the better parent's score
   on that subset, then finish validation. Record both parents.
6. Return the highest mean-validation candidate from the entire pool; equal
   means prefer the earliest candidate. Keep other candidates for ancestry and
   inspection even when they leave the selection frontier.

Training feedback alone enters reflection prompts. Validation is used only for
scoring, selection, and merge screening. Supply disjoint datasets for adaptation;
keep the final test set outside the optimizer. For the paper's inference-time
search setting, deliberately pass the same instances as both datasets.

Supply nonempty prompt mappings and datasets, a positive minibatch size, and a
budget sufficient for initial validation. Caller settings are trusted. Each
mutation reserves `2 * min(minibatch_size, len(train)) + len(validation)` rollouts
before starting; a merge reserves `len(validation)`. Insufficient tail budget
remains unused. The initial evaluation raises `RuntimeError` before any calls if
it cannot fit. There is no cross-candidate evaluation cache; a merge reuses its
own screening measurements when completing validation.

The return dictionary contains `best` (`Candidate`), `population`, `pareto_weights`,
`history`, `raw_responses`, `rollouts`, `reflection_calls`, and `merge_attempts`.
Candidate scores follow validation order; parent IDs index `population`.
History records accepted, non-improving, unchanged, and invalid proposals.
Raw reflection output is retained before extraction. Empty or unfenced output
is rejected; provider errors, evaluator errors, and nonfinite measurements
propagate, with attempted calls already counted on the agent. There are no
automatic retries or fallback model calls. Every `run()` resets search state;
use one run at a time per agent. No shared conversational Session is used.
Slick's template root is process-global; different template roots need separate
processes when running concurrently.

## Official implementation and scope

This is a small Slick port using the [official GEPA repository](https://github.com/gepa-ai/gepa),
reviewed at commit [`15ee314f9c7d34ec153b809d401f42f55c4dcd76`](https://github.com/gepa-ai/gepa/tree/15ee314f9c7d34ec153b809d401f42f55c4dcd76).
The source's [MIT license](LICENSE) is retained. The implementation adapts:

- [`gepa_utils.py`](https://github.com/gepa-ai/gepa/blob/15ee314f9c7d34ec153b809d401f42f55c4dcd76/src/gepa/gepa_utils.py): coverage pruning and frequency weights.
- [`component_selector.py`](https://github.com/gepa-ai/gepa/blob/15ee314f9c7d34ec153b809d401f42f55c4dcd76/src/gepa/strategies/component_selector.py): per-candidate round-robin module selection.
- [`instruction_proposal.py`](https://github.com/gepa-ai/gepa/blob/15ee314f9c7d34ec153b809d401f42f55c4dcd76/src/gepa/strategies/instruction_proposal.py): reflection instructions, rendered locally with task/module context and evaluation records.
- [`merge.py`](https://github.com/gepa-ai/gepa/blob/15ee314f9c7d34ec153b809d401f42f55c4dcd76/src/gepa/proposer/merge.py): ancestry eligibility, module combination, and stratified merge screening.

The official coverage rule can prune a candidate that no single score vector
strictly dominates; this port follows that code rather than substituting a
textbook Pareto algorithm. Appendix D's prose describes disjoint changes, but
Algorithms 3–4 and the code permit overlapping changes when at least one module
has a complementary edit; this port preserves that rule too.

Deliberate differences from the full upstream engine: hard budget reservation;
uniform training minibatches without replacement (clamped for small datasets);
weighted `random.choices` for parent sampling; uniform choice among eligible
common ancestors; unique merge-screening instances for small validation sets;
at most `max_merges` **evaluated attempts**, including rejected merges;
no-op/duplicate merge suppression; and strict nonempty fenced-output extraction.
Merge search makes ten pair draws per opportunity, and reflection does not skip
perfect-scoring batches (as in the paper's Algorithm 1).
These choices are reproducible locally with a seed but do not promise identical
upstream random trajectories. There is no DSPy adapter, checkpointing engine,
benchmark harness, or reinforcement-learning implementation. The folder is a
local `gepa` package, not a drop-in replacement for the upstream package's API.

## Offline checks

From the repository root, use the existing environment with Slick 0.3.0:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_gepa
../slick/.venv/bin/ruff check gepa tests/test_gepa.py
../slick/.venv/bin/ruff format gepa tests/test_gepa.py --check
```

Alternatively, from this folder install `requirements.txt` into your environment;
it references the adjacent Slick checkout. The tests use the repository's shared
scripted provider. They verify algorithm decisions and prompt integration,
including rendering from another working directory. Pareto weights were also
compared directly with upstream on 1,000 generated score matrices. These are
offline correctness checks, not a reproduction of the paper's benchmark gains.
