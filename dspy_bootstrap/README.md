# DSPy bootstrapped demonstration compilation

`DSPyBootstrap(modules, teacher, student, metric, threshold=1).run(train, validation, candidates=16, max_bootstrapped=4, max_labeled=16)` implements BootstrapFewShot plus randomized candidate compilation. This complements the separate MIPRO port.

Sources: [original DSPy paper](https://arxiv.org/abs/2310.03714), official [`bootstrap.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/bootstrap.py) and [`random_search.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/random_search.py), inspected 2026-09-14. Candidates include zero-shot, labeled-only, unshuffled bootstrap, and seeded shuffled bootstraps with sampled demonstration counts. Each is evaluated on separate validation examples; highest mean metric wins.

`Example(inputs, outputs)` defines labeled data. Both async executors receive `(program, inputs, rollout_id)` and return `Prediction(outputs, trace)`, where each `Trace(module, example)` records an actual module call. The program maps module names to demonstrations. Executors run fixed programs; they do not perform optimization. Before teacher execution, the current example's label is removed from all teacher demos. Only successful whole-program traces contribute module-specific demonstrations; remaining raw labels fill the configured total. The metric is synchronous and finite. Caller modules must handle the selected input/output fields; training labels never become student validation inputs.

Adaptations: plain records replace DSPy modules and settings. A seeded RNG replaces the source's content-hashed repeated-module trace RNG. Dependency errors propagate instead of being swallowed under a global error budget. There are no optimizer-owned LLM prompts, so executors can use Slick with their own local templates. Returns scored candidate programs and teacher/validation counts. No model compilation framework or checkpoint is bundled.

Check: `rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_dspy_bootstrap`.
