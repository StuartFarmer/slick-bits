# MoP

Slick port of Mixture-of-Prompts using the official automatic clustering,
compensating instruction generation and region-based joint assignment variant.

Official sources inspected:

- [Paper](https://arxiv.org/abs/2407.00256)
- [MoPTrainer](https://github.com/ruocwang/mixture-of-prompts/blob/main/mop/src/algo/mixture_of_experts/mop_trainer.py)
- [Expert-count objective](https://github.com/ruocwang/mixture-of-prompts/blob/main/mop/src/algo/mixture_of_experts/utils.py)

`MoP(task, provider, evaluate, embed)` receives input/target `Example` objects.
`embed(input_strings)` returns fixed numeric input embeddings, never labels.
`evaluate(instruction, demonstrations, examples)` executes an expert on the
supplied evaluation examples and returns a finite higher-is-better score. The
caller owns all task-model execution and isolation.

The agent clusters demonstration inputs and selects the expert count minimizing
`within_cluster_inertia / total_inertia + 0.02 * number_of_experts`, matching the
source criterion. Each expert's demonstration allotment is capped at
`len(demonstrations) // max_experts`; small clusters repeat examples to fill it,
while trimmed examples remain available for instruction generation.

Instructions are generated from the other experts' demonstration partitions
and the trimmed examples, following the source's `compensate` option. The merged
candidate pool is scored without demonstrations and the strongest instructions
are shared across experts. Each expert then tests every retained instruction on
evaluation queries closest in cosine similarity to its allotted demonstration
centroid. It retains its best regional instruction together with its assigned
demonstrations. At inference, `route(input)` selects the nearest original cluster
center and `predict(input)` executes only that expert's prompt.

```python
from pathlib import Path
from slick import prompts
from mop import Example, MoP

prompts.TEMPLATE_ROOT = Path("mop/prompts").resolve()
agent = MoP(task, provider, evaluate, embed)
result = await agent.run(demonstrations, validation_examples)
answer = await agent.predict(new_input)
```

Intentional adaptations: local NumPy k-means++/Lloyd iterations replace scikit-
learn; automatic clustering uses one initialization per candidate count. Identical
embeddings collapse to one expert, avoiding the source zero-inertia/empty-cluster
failure. Demo allotment has a minimum of one. A single expert with no complement
uses its own demonstrations for candidate generation. Stable ties and negative
finite scores are supported; the source initializes regional best scores at zero,
which can leave no chosen prompt when every score is nonpositive. The complement
construction also corrects the source's partition-count indexing typo. Random/fixed
partition, independent assignment and unconditional-generation variants are not
implemented. Candidate generation uses task-neutral local templates.

Regional scores are not a held-out mixture accuracy estimate. Results expose
experts, cluster centers, candidate instructions, measurements and counts.
Malformed instructions, nonfinite measurements/embeddings and dependency errors
propagate. Configure the process-global template root once before training or
prediction. No benchmark execution, paid calls or reproduction claim.

Check: `rtk proxy optimizer/.venv/bin/python -m unittest tests.test_mop`.
