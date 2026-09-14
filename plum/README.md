# Plum

`Plum(task, provider, evaluate, spans)` runs the official modular trainer
families with constituent edits. Call `await agent.run(initial_prompt,
algorithm="ga")`; supported algorithms are:

- `ga`: tournament parent selection from a growing archive; append the best
  neighbor even when it fails to improve the global incumbent (GA-M).
- `hc`: greedy best-neighbor hill climbing.
- `sa`: best-neighbor hill climbing with exponential-temperature acceptance.
- `tabu`: incumbent-centered search, a FIFO record of neighborhood winners,
  and probabilistic admission of exact repeats (upstream default 0.5).
- `hs`: segment recombination from harmony memory, optional paraphrase pitch
  adjustment, and top-score truncation of old plus new harmonies.

The parser's `spans(text)` returns disjoint constituent character offsets.
`evaluate(text)` asynchronously returns a finite score; larger is better.
Substitution uses the injected Slick provider. Other edits use ordinary Python.
The pure edit operation is reused from GrIPS, matching Plum's original reuse.
The deletion bank changes only after accepted edits (retained harmonies in HS).
Results include current/best prompts, population, history, taboo rejections,
evaluation counts, and generation counts. Scores are not cached.

Sources inspected: [paper](https://arxiv.org/abs/2311.08364),
[official repository](https://github.com/research4pan/Plum),
[`GA_trainer.py`](https://github.com/research4pan/Plum/blob/main/trainers/GA_trainer.py),
[`HC_trainer.py`](https://github.com/research4pan/Plum/blob/main/trainers/HC_trainer.py),
[`TB_trainer.py`](https://github.com/research4pan/Plum/blob/main/trainers/TB_trainer.py),
[`HS_trainer.py`](https://github.com/research4pan/Plum/blob/main/trainers/HS_trainer.py),
and [`base_trainer.py`](https://github.com/research4pan/Plum/blob/main/trainers/base_trainer.py).

Adaptations: SuPar/Pegasus and task scoring are injected; provider paraphrases
replace Pegasus beams. Empty edits consume a bounded attempt rather than
upstream unbounded regeneration. HS preserves each partition's full segment
instead of the upstream `end-1` slice that drops a constituent. Python RNG,
character-offset edits, and no benchmark-specific normalization change exact
trajectories. Best-ever and current results are separated for annealing.
The separate older `run_search_modified_GA_C_add.py` GA-with-crossover experiment
is not implemented by `ga`; this port intentionally exposes the modular GA-M.

Configure Slick before running (the template root is process-global):

```python
from pathlib import Path
from slick import prompts
import plum
prompts.TEMPLATE_ROOT = Path(plum.__file__).parent / "prompts"
```

Requires the repository's Slick installation; no model, dataset, credentials,
or execution runner is constructed. Provider/evaluator exceptions propagate;
transport retries belong to the caller. Use one active run per agent instance.
These are source-grounded algorithm ports with adapted generation boundaries,
not reproductions of the papers' benchmark results.
