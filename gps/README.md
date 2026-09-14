# GPS

`GPS(task, provider, evaluate, accept=...)` implements Genetic Prompt Search:
score the initial prompts, select the best `top_k` in each generation, paraphrase
each selected parent, and use valid unique offspring as the next population.
When all offspring are rejected, keep the selected parents. Final selection
returns the best unique prompts over every evaluated generation, so an earlier
winner survives even if the current generation performs worse.

```python
agent = GPS(task, provider, evaluate)
result = await agent.run(initial_prompts, generations=9, top_k=4,
                         offspring_per_parent=10)
```

`generations` includes the initial evaluated generation. `evaluate` returns a
finite score, higher is better, and is not cached. `accept(parent, child)` checks
task/template constraints; the default preserves the exact multiset of literal
`{{...}}` input placeholders. Candidate templates are data and are never executed
by this agent. Blank, duplicate, and rejected offspring consume a model call
without evaluation and are recorded with their raw response.

Sources inspected: [paper](https://aclanthology.org/2022.emnlp-main.559/),
[official repository](https://github.com/hwxu20/GPS),
[`ga_processer_t0.py`](https://github.com/hwxu20/GPS/blob/main/ga_processer_t0.py)
(top-k selection, generation fallback, final cross-generation merge), and
[`dino/use_dino_to_generate_template_t5.py`](https://github.com/hwxu20/GPS/blob/main/dino/use_dino_to_generate_template_t5.py)
(paraphrase generation, template reconstruction/filtering, and deduplication).

Adaptations: a local Slick paraphrase call replaces T5/DINO batched generation.
Caller-owned validation replaces per-dataset handwritten template reconstruction
and lexical filters; pass equivalent constraints when reproducing a dataset.
Task-specific top-k values become a caller setting. This implementation does not
introduce crossover: the released GPS pipeline generates paraphrases. Historical
best selection and fresh-generation replacement remain separate decisions.

Configure Slick before running (the template root is process-global):

```python
from pathlib import Path
from slick import prompts
import gps
prompts.TEMPLATE_ROOT = Path(gps.__file__).parent / "prompts"
```

Requires the repository's Slick installation; no model, dataset, credentials,
or execution runner is constructed. Provider/evaluator exceptions propagate;
transport retries belong to the caller. Use one active run per agent instance.
These are source-grounded algorithm ports with adapted generation boundaries,
not reproductions of the papers' benchmark results.
