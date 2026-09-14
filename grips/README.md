# GrIPS

`GrIPS(task, provider, evaluate, spans)` implements constituent edit search:
freeze the incumbent, compose randomly selected delete/swap/substitute/add edits,
score the neighborhood, accept its best strict improvement, and stop after
patience is exhausted. Optional annealing accepts a non-improving neighborhood
winner with `exp((new-old)/T)`, `T = temperature * exp(-step/cooling)`.
Addition restores phrases from accepted deletions only. Return values distinguish
best-ever and current prompts, accepted history, rejections, attempts, model calls,
and evaluator calls. Finite scores are maximized and not cached.

```python
agent = GrIPS(task, provider, evaluate, spans)
result = await agent.run(initial_prompt, candidates=5, iterations=10)
```

`spans(text)` supplies disjoint constituent `(start, end)` character offsets in
that exact text. Supply a constituency parser to recover the paper's phrase
segmentation; token/sentence spans deliberately change the search granularity.
Substitution uses a local Slick paraphrase prompt. All other edits are ordinary
Python and do not need prompts. Blank edited candidates consume attempts and
are skipped; a blank generated paraphrase raises before evaluation.

Sources inspected: [paper](https://arxiv.org/abs/2203.07281),
[official repository](https://github.com/archiki/GrIPS), and
[`run_search.py`](https://github.com/archiki/GrIPS/blob/main/run_search.py)
(`perform_edit`, deletion tracker, candidate loop, greedy/annealed acceptance).

Adaptations: caller-owned parser and score replace SuPar/Pegasus and the
classification-specific balanced-accuracy/entropy score. A single provider
paraphrase replaces Pegasus's random pick among ten beams. Character-offset
edits replace upstream global string substitutions, avoiding unintended edits
to repeated phrases. Python's seeded RNG is reproducible locally but does not
reproduce NumPy's upstream sequence. Empty-neighborhood handling is bounded,
and best-ever tracking remains available when annealing moves downhill.

Configure Slick before running (the template root is process-global):

```python
from pathlib import Path
from slick import prompts
import grips
prompts.TEMPLATE_ROOT = Path(grips.__file__).parent / "prompts"
```

Requires the repository's Slick installation; no model, dataset, credentials,
or execution runner is constructed. Provider/evaluator exceptions propagate;
transport retries belong to the caller. Use one active run per agent instance.
These are source-grounded algorithm ports with adapted generation boundaries,
not reproductions of the papers' benchmark results.
