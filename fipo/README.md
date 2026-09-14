# FIPO

`FIPO(task, provider, evaluate).run(response=None, reference=None, max_words=200)`
assembles the paper's modular input and makes one learned-optimizer call. Observed
and golden responses are independently optional. The returned instruction is
scored with the caller's async evaluator (finite, higher-is-better score).

Use a provider serving the authors' **Junrulu/FIPO-IPL-IPO-Tulu2-70B** checkpoint
with its appropriate chat formatting. The provider must return the completion,
not echo the input. Generic chat models do not reproduce learned FIPO. Configure
`slick.prompts.TEMPLATE_ROOT` once to this folder's absolute `prompts/` directory.

Sources inspected: [paper](https://arxiv.org/abs/2402.11811), official
[modular prompts](https://github.com/LuJunru/FIPO_Project/blob/046fc3c86f8545f2a1cc9bfbdd93c845857f071a/data/prompts.json)
and `codes/get_model_infer_batch.py`. The authors' “Sliver” spelling and instruction
wording are retained. Python selects one of four separate prompt operations.
An optional leading `Golden Prompt:` label is removed; blank results reject.

This implements **deployment inference**, not POP construction, SFT, DPO, IPO or
IPL checkpoint training. The requested word limit is a model instruction, not a
proven fidelity guarantee. Generation/evaluation failures propagate; no retries.
Tests check all four module combinations, not checkpoint or benchmark performance.
