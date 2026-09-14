# PE2

`PE2(task, provider, evaluate, observe, full_template=...)` performs the paper's
two-stage prompt engineering: analyze concrete model outputs against authoritative
labels, then revise the prompt using that analysis. The first stage explicitly
shows the actual task-model input template and distinguishes an incorrect model
answer from an incorrect task description. Optional momentum summarizes edits
and carries each parent's own history into subsequent revisions.

```python
agent = PE2(task, provider, evaluate, observe,
            full_template="Question: {input}\nInstruction: {instruction}")
result = await agent.run(initial_prompts, iterations=5, beam_size=4,
                         children=2, batch_size=2, momentum=True)
```

`evaluate(prompt)` returns a finite validation score, higher is better.
`observe(prompt)` returns training observations with `input`, `output`, `label`,
`score`, and optionally `reasoning`. Hard-example sampling selects `score == 0`;
set `hard_examples=False` to sample all observations. Validation examples are
never passed to the optimizer by the agent. Global backtracking selects parents
from all evaluated candidates; `backtrack=False` selects the current generation.
The final result always preserves the best historical candidate. Repeated prompt
text is rejected before evaluation; empty candidate sets end the search.

Pass an empty initial list with `demonstrations=[...]` to induce `initial_count`
starting prompts. Results report candidates, ancestry-specific history, duplicate
proposals, validation evaluations, training observation calls, and optimizer
calls. Each `iterations` unit is a full expansion; initial scoring is additional.

Sources inspected: [paper](https://arxiv.org/abs/2311.05661),
[official repository](https://github.com/INK-USC/PE2),
[`trainer/pe2_trainer.py`](https://github.com/INK-USC/PE2/blob/main/trainer/pe2_trainer.py),
[`trainer/default_trainer.py`](https://github.com/INK-USC/PE2/blob/main/trainer/default_trainer.py),
and [`meta_prompts/pe2/proposer.md`](https://github.com/INK-USC/PE2/blob/main/meta_prompts/pe2/proposer.md).

Adaptations: the original multi-turn Guidance program becomes explicit separate
Slick calls for analysis, revision, and optional history summary. Templates retain
the three distinguishing components (detailed two-stage task description,
full-context specification, and per-example reasoning questions), but condense
wording and omit optional generic prompting tutorials/demonstrations. Caller
observations replace task-specific DataFrame packing and early error-collection
logic. Prompt history carries measured validation scores, rather than silently
claiming they are upstream training accuracy. No accuracy==1 early stop is
imposed on a task-agnostic score scale. Word-count edit limits and legacy
weighted-hard batching are not implemented.

Configure Slick's process-global template root before use:

```python
from pathlib import Path
from slick import prompts
import pe2
prompts.TEMPLATE_ROOT = Path(pe2.__file__).parent / "prompts"
```

Requires the repository's Slick installation. Provider/model construction,
benchmarks, credentials, persistence, and execution isolation belong to callers.
All model calls are sequential, and provider/evaluator failures propagate without
retries. Blank optimizer prose raises before scoring. Use one active run per
agent instance. Deterministic tests check algorithm mechanics, not benchmark
performance or reproduction of the published numerical results.
