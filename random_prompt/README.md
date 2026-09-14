# Random prompt separators

Port of [yaolu/random-prompt/main.py](https://github.com/yaolu/random-prompt/blob/main/main.py), paper [2311.09569](https://arxiv.org/abs/2311.09569). This paper searches separators, not demonstration subsets.

`RandomPrompt(task, provider, evaluate, decode=..., configure_sampling=...)` owns `run(context, mode="vocabulary", draws=20, ...)`. The fixed context contains literal `{separator}` placeholders, including those in demonstrations and queries. Every placeholder is replaced locally. The async evaluator returns a finite higher-is-better score; no held-out evaluation is performed internally.

Vocabulary mode samples a uniform length, then distinct token IDs without replacement and decodes them. Unconditional mode uses the empty local Slick template: configure_sampling(provider, length) must return a provider configured for stochastic unconditional generation with that maximum output length. All draws are generated before scoring; repeated separator strings remain separate draws. Ranking is stable, with the best four retained by default. Exceptions propagate, without retries. Result counts distinguish model calls from evaluator calls.

Configure `slick.prompts.TEMPLATE_ROOT = Path(random_prompt.__file__).parent / "prompts"` once before unconditional generation. Candidate text and whitespace are preserved. The caller owns tokenizer, provider, fixed few-shot construction and task scoring. Benchmark-specific balanced classification contexts and upstream top-four test reporting are excluded. Deterministic tests verify selection and plumbing, not the paper's performance.

