# SCULPT

`SCULPT(task, provider, evaluate, errors, render=...)` refines a hierarchy of
`Section(title, body, examples, children)` values. `await agent.run(sections)`
performs preliminary structural assessment, applies its actor actions, collects
observed errors on that revised prompt, assesses sampled error batches, groups
feedback by referenced top-level section, and creates one actor revision per
group. Old beam members and new revisions compete by measured score.

The actor emits local rephrase/create/delete/merge/reorder/example actions.
Actions operate on a copied tree; absent/ambiguous paths and invalid reorder or
merge instructions reject the whole action batch, retaining its parent. Example
updates have a separate prompt and enforce the source's six-example ceiling.
An empty feedback list does not trigger an actor call. Stable score ties prefer
incumbents. Results expose beam trees/text/scores, rejected action records, scalar
evaluation count, error-collection count, and optimizer call count.

```python
from sculpt import SCULPT, Section
agent = SCULPT(task, provider, evaluate, errors)
result = await agent.run([Section(title="Instructions", body=initial_instruction)],
                         iterations=5, beam_size=4)
```

`evaluate(text)` returns a finite scalar, larger is better, cached per run.
`errors(text)` returns actual failure records such as dictionaries with `input`,
`expected`, and `prediction`; caller sampling must use search/training data,
not held-out test data. `render(sections)` defaults to hierarchical Markdown.
Callers supply a parsed hierarchy, avoiding lossy guessing about arbitrary
markup. Nest sections to retain the meaningful structure of a long prompt.

Sources inspected: [paper](https://arxiv.org/abs/2410.20788),
[final ACL paper](https://aclanthology.org/2025.acl-long.730/),
[official repository](https://github.com/Sshanu/SCULPT),
[`src/sculpt/optimizers.py`](https://github.com/Sshanu/SCULPT/blob/main/src/sculpt/optimizers.py)
(`run_critic`, `aggregate_feedbacks`, `run_actor`, `apply_actions`, `expand_candidates`),
[`batch_critic_template.md`](https://github.com/Sshanu/SCULPT/blob/main/src/our_data/batch_critic_template.md),
and [`batch_actor_template_agg.md`](https://github.com/Sshanu/SCULPT/blob/main/src/our_data/batch_actor_template_agg.md).

Adaptations: supplied trees replace the benchmark Markdown parser; JSON actions
use typed title paths. Preliminary and error assessment use separate Slick
operations, and errors are recollected after structural revision instead of
reusing stale pre-revision predictions. Explicit feedback grouping and ordinary
beam scoring implement the core mode. Optional upstream whole-prompt rephrase,
crossover augmentation, implicit clustering, and bandit evaluators are not
implemented. Local actions are atomic and propagate transport/parsing failures
rather than reproducing upstream broad catches and partially applied edits.

Configure Slick before running (the template root is process-global):

```python
from pathlib import Path
from slick import prompts
import sculpt
prompts.TEMPLATE_ROOT = Path(sculpt.__file__).parent / "prompts"
```

Requires the repository's Slick installation; no model, dataset, credentials,
or execution runner is constructed. Provider/evaluator exceptions propagate;
transport retries belong to the caller. Use one active run per agent instance.
These are source-grounded algorithm ports with adapted generation boundaries,
not reproductions of the papers' benchmark results.
