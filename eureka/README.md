# Eureka

`Eureka(task, provider, train, interface=...).run(rounds=5, candidates=16)` evolves reward-program source using actual policy-training feedback. The async `train(code)` must compile, execute and train in caller-owned isolation, returning `TrainingResult(score, components, error)`. Scores are maximized. `components` maps names to nonempty finite time series; the agent calculates sampled trajectories, min, max and mean. Explicit execution failures use `error`; infrastructure exceptions propagate.

Sources: [paper](https://arxiv.org/abs/2310.12931), official [eureka.py](https://github.com/eureka-research/Eureka/blob/9eee42808b52d8abc14845d1547bfde886403ccc/eureka/eureka.py). Each batch reads the same prior context. Its best successfully trained program supplies the next context, even when weaker than the historical best. An entirely failed multi-candidate batch leaves context unchanged; a single failed candidate supplies error feedback. A separate historical best is returned, or `None` if every program failed.

Adaptations: task-neutral interface and explicit JSON code replace IsaacGym-specific insertion and fenced parsing. Calls run sequentially. The adapter supplies the task score (upstream uses peak success) and chooses observable reward components. Context is rendered as one Slick request instead of four chat messages. No generated program is executed here. Final multi-seed held-out evaluation, RL trainers and reward-correlation reporting remain external; deterministic tests are not a performance reproduction.

Set Slick's template root to this folder's `prompts/`. Returns best `Trial`, all trials and training-call count. Check: `rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_eureka`.
