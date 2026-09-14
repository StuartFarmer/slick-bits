# PromptAgent

Task-agnostic Slick port of PromptAgent's error-driven Monte Carlo tree search.

Official sources inspected (the original repository now redirects to maitrix-org):

- [Paper](https://arxiv.org/abs/2310.16427)
- [MCTS selection, expansion, simulation and backup](https://github.com/maitrix-org/PromptAgent/blob/main/src/prompt_optim_agent/search_algo/mcts.py)
- [Gradient descent and all-correct branch](https://github.com/maitrix-org/PromptAgent/blob/main/src/prompt_optim_agent/world_model/gradient_descent.py)
- [Official operation prompts](https://github.com/maitrix-org/PromptAgent/blob/main/src/prompt_optim_agent/world_model/prompts/gradient_descent_prompts.py)

`PromptAgent(task, provider, evaluate, observe).run(initial_prompt, examples, ...)`
owns the complete search. `observe(prompt, batch)` executes a sampled training
batch and returns `Feedback(errors=..., correct=...)`; `evaluate(prompt)` measures
the candidate on a fixed validation set with a finite higher-is-better score.
The caller owns both datasets, model execution and required isolation. The
callbacks supply task evidence, not search or revision policies.

Selection uses source UCT: `Q + c * sqrt(log(parent_backups + 1) /
max(1, node_backups))`. Expansion reflects on errors, or on correct examples when
there are no errors, and proposes successors using the ancestor prompt trajectory.
Each expansion samples `expand_width` training batches, each yielding
`proposals_per_batch` successors. Simulation follows the highest immediate reward.
Backup adds **suffix sums of rewards**, and Q is the mean of these returns,
falling back to the node's immediate reward before its first backup.

The source depth limit, root/parent-relative weak-node pruning and improvement
threshold early stop are preserved. Early stop only applies beyond `min_depth`.
The selected result is the best-reward node on the visited path with greatest
mean immediate reward, matching `best_reward_path_selected_node`; `best_global`
also exposes the strongest evaluated node. These can differ. Terminal paths are
still backed up when revisited, as in the source. Zero iterations returns the root.

```python
from pathlib import Path
from slick import prompts
from promptagent import Feedback, PromptAgent

prompts.TEMPLATE_ROOT = Path("promptagent/prompts").resolve()
agent = PromptAgent(task, provider, evaluate, observe)
result = await agent.run(initial_prompt, training_examples)
```

Intentional adaptations: caller-provided string evidence replaces task-specific
question/answer classes and dataloaders. Sampling is seeded and without
replacement within each batch. Four generic operation templates retain error
feedback, successful-example ascent, trajectory-conditioned revision and tagged
successors, but are not byte-identical source prompts. Output count and nonempty
text are checked rather than silently accepting missing successors. No
global prompt deduplication is added: equal text can represent distinct branches.
No held-out test runner, result persistence, model construction or paid calls.

Counts include attempted optimizer calls, validation evaluations and training
observations. Malformed output, nonfinite measurements and dependency failures
propagate without retry. Nodes and paths remain inspectable after failure.
Configure the process-global template root once, not concurrently across agents.

Check: `rtk proxy optimizer/.venv/bin/python -m unittest tests.test_promptagent`.
Deterministic traces establish mechanics, not optimization effectiveness.
