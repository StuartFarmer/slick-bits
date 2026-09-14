# Slick bits

Small, runnable projects built with [Slick](../slick).

Development guidance: [style comparison report](docs/slick-style-report.md),
[Slick style guide](skills/slick-development/references/style-guide.md), and
[reusable development skill](skills/slick-development/SKILL.md).

Each implementation keeps its Jinja templates in its own `prompts/` directory.
Its CLI configures that directory at startup; see the implementation's README for
programmatic use. Python formatting and import checks use [ruff.toml](ruff.toml):
`ruff format .` and `ruff check .` (recorded runs are excluded).

- [ReEvo reflective evolution](reevo/README.md): evolve optimization heuristics with Slick, dual-level reflections, and a runnable TSP ACO example.

- [OpenAlex explorer](scout/README.md): an interactive CLI for selecting papers, crawling their references and citations, and reviewing the next batch.
- [Optimizing the optimizer](optimizer/README.md): CMSA for maximum independent set, five heuristic variants, paired benchmarks, and a Slick code-generation dialogue.
- [QUBE](qube/README.md): quality–uncertainty balanced heuristic evolution with Slick, clustered islands, isolated evaluation, and bin-packing, cap-set, and TSP tasks.
- [Evolution of Heuristics](eoh/README.md): Slick-driven evolution of thoughts and code with five prompt strategies, bin packing, TSP and flow shop evaluators.
- [APEX long-prompt optimization](apex/README.md): sentence-level beam search, history-guided mutations, and LinUCB selection using Slick.
- [Algorithm evolution (AEL)](ael/README.md): evolve TSP construction algorithms with Slick and OpenRouter, isolated evaluation, and exact reference tours.
- [Evolving code with an LLM](llm_gp/README.md): LLM_GP symbolic regression, both paper variants, three baselines, checked Slick operators, and experiment logs.
- [EvoPROMPT](evoprompt/README.md): GA and differential evolution for discrete prompt optimization, with Slick model calls, development scoring, and held-out evaluation.
