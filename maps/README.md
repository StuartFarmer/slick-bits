# MAPS

Task-agnostic adaptation of **The Prompt Alchemist: Automated LLM-Tailored Prompt
Optimization for Test Case Generation**, [paper](https://arxiv.org/abs/2501.01329).
Grounded in the authors' [official replication artifact, Zenodo record 14287744](https://zenodo.org/records/14287744):
`TSE_code.zip/TSE_MAPS-master/code/auto_prompt_ours.py`, especially
`prompt_improvement`, `error_message_analysis`, and `rule_validation`. The archive
README and paper identify it as the replication package. Source was inspected,
not executed.

MAPS selects a beam, generates distinct modification suggestions, and creates one
new instruction per suggestion. It clusters the selected prompts' failures,
chooses a cluster with a size-and-novelty weight, reflects on representative
failures, and proposes prevention rules. Each rule is evaluated separately with
the best old instruction and existing rules. Only the highest scoring strictly
beneficial rule is accepted. All surviving old and new instructions are then
evaluated under the resulting shared rules before the next selection.

```python
from pathlib import Path
from slick import prompts
from maps import MAPS, Evaluation, Cluster

prompts.TEMPLATE_ROOT = Path("/absolute/path/to/maps/prompts")
agent = MAPS(task, provider, evaluate, cluster, domain_context=context)
result = await agent.run(initial_prompts, beam_size=3, mutations=3)
full_prompt = result["best"].prompt
```

`evaluate(full_prompt)` asynchronously returns `Evaluation(score, failures)`, with
a finite higher-is-better metric and generic failure dictionaries. `cluster(rows)`
returns `Cluster(representative, size, examples)` objects. The representative is a
failure-signature string, size is the cluster population, and examples are actual
representative failures. To match the release, use TF-IDF plus DBSCAN (`eps=1`,
`min_samples=10`, retaining noise as a cluster), with up to three examples nearest
each centroid and the nearest example's signature as `representative`. These
domain-sensitive operations are injected rather than requiring scikit-learn.

The native sampling weight is cluster size multiplied by the minimum normalized
Levenshtein distance to a previously tried representative. Normalization divides
by the **sum** of the strings' lengths, matching the released implementation's
similarity calculation. Rejected rules also record their representative, as in the
source. If every cluster has zero novelty, this port skips rule induction instead
of reproducing the source's division by zero. New rules are tested against the
old best prompt, not an unmeasured mutation.

`domain_context` supplies task-specific context extracted by the caller. The
original Java preprocessing, test execution, and coverage aggregation are replaced
by these context/evaluation callbacks. Java instructions and source delimiters are
rewritten into task-neutral prompts and a typed JSON mutation-suggestion list.
Generated suggestion count and distinctness are checked. The rule count defaults
to the release's three candidates for the one selected cluster. Complete rendered
prompts are cached, so provide a fixed evaluator. The returned best belongs to the
final population under accepted shared rules; rejected rule trials cannot win.

The result includes population, accepted rules, attempted representatives, rule
trials and acceptance decisions, and exact generation/evaluation counts. Errors
propagate without the original open-ended retry loops. Dependencies are Slick,
Pydantic, and the standard library; a source-equivalent clustering callback
additionally needs scikit-learn. No generated code is executed by this agent.
Tests check strict rule acceptance, shared-rule rescoring, rejected-cluster novelty,
edit distance, and malformed suggestions with the shared scripted provider.
