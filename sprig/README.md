# SPRIG

`SPRIG` searches an ordered sequence of system-prompt components. It enumerates
insertions at every position, deletions, pairwise swaps, and component paraphrases.
UCB limits insertion choices; adding a component earns its measured improvement,
while deleting it earns the negated improvement. Paraphrases retain their original
component ancestry for this credit. Only edited candidates enter the next beam.
Repeated components and empty prompts are allowed.

Sources: [paper §3](https://arxiv.org/html/2410.14826v1),
[official source inspected](https://github.com/orange0629/prompting/blob/main/scripts/sprig.py).
The source enumerates these edits, records addition/deletion credit, and aggregates
paraphrase credit by source component.

Adaptations: this port replaces Pegasus with a Slick paraphraser and rewritten
JSON instructions. It always applies the configured UCB cap, including the first
round (the release initially considers the entire corpus). It remeasures parents
alongside children on the same batch before calculating credit, whereas the release
uses each parent's latest stored score. Ties preserve enumeration order. These
changes are explicit experimental choices, not a reproduction of reported scores.

```python
from pathlib import Path
from slick import prompts
from sprig import SPRIG

prompts.TEMPLATE_ROOT = Path("sprig/prompts").resolve()
agent = SPRIG(task_collection_description, provider, evaluate_batch, component_corpus)
result = await agent.run(rounds=10, beam_size=10)
system_prompt = result["best"].prompt
```

`evaluate_batch(prompts, generation)` is async and returns one finite, larger-is-better
score per input, in order. It owns target-model inference and a shared sampled
training subset across all candidates, ideally balanced across tasks. Generation
`-1` evaluates the initial prompt. The caller supplies the component corpus, model
providers, benchmark data, aggregation, and held-out evaluation. No model weights or
datasets are downloaded. `evaluations` counts scored prompts including rescoring;
`credit` exposes component rewards. Generated count/text errors, score alignment,
nonfinite measurements, provider exceptions, and evaluator exceptions stop the run
without retries. Caller configuration is trusted. Configure the process-global Slick
template root before use; do not switch roots concurrently.

Validation: `python -m unittest tests.test_sprig` uses the shared scripted provider
to check edit boundaries, UCB credit, common batches, and failure propagation.
