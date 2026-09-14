# Slick bits

Problem-agnostic algorithm classes built with [Slick](../slick). Each class
owns its prompt methods and a short async `run()` coordinating named algorithm phases. Callers supply the task,
provider, and an async evaluator where required; each implementation owns its own `prompts/` folder.

The [automatic prompt optimization collection](prompt_optimization/README.md)
covers 86 method entries across the survey and its complete research bibliography,
with individual implementations, official sources, usage contracts and documented adaptations.

| Agent | Flow |
| --- | --- |
| [ProAgent](proagent/README.md) | Cooperative skill planning, precondition verification, persistent control, and observed-behavior belief revision |
| [AlphaEvolve](alphaevolve/README.md) | Evaluated code evolution with protected edits, multi-objective islands, model ensembles, and asynchronous feedback |
| [Alpha-GPT](alpha_gpt/README.md) | Idea polishing, hierarchical retrieval, seeded genetic search, and interactive review |
| [AdaEvolve](adaevolve/README.md) | Adaptive exploration, decayed-reward island scheduling, migration, and solution tactics |
| [ADaPT](adapt/README.md) | Executor-first recursive decomposition with short-circuit AND/OR plans |
| [AEL](ael/README.md) | Population evolution with crossover, mutation, and elitism |
| [Algorithm of Thoughts](aot/README.md) | In-context DFS/BFS or plan exploration in one generation, with optional warm-up |
| [APET](apet/README.md) | One toolbox-guided prompt rewrite, input preservation, and independent answer comparison |
| [Ask Me Anything](ama/README.md) | Reusable QA prompt chains with dependency-aware weak supervision or open-answer majority vote |
| [APEX](apex/README.md) | Sentence-level beam search with history and LinUCB selection |
| [Boosting of Thoughts](bot/README.md) | Parallel weighted trees, chain aggregation, and accumulated trial-and-error feedback |
| [CAMEL](camel/README.md) | Task specification, cooperative role-playing, and optional critic proposal selection |
| [CCMO-LLM](ccmo_llm/README.md) | Constrained multiobjective coevolution with shared GA/LLM offspring and SPEA2 survival |
| [CRITIC](critic/README.md) | Tool-interactive verification and iterative correction of arbitrary outputs |
| [Darwin Gödel Machine](dgm/README.md) | Executable self-modification, performance/child-count selection, and an archive of functional stepping stones |
| [EoH](eoh/README.md) | Five strategies for evolving candidate ideas and content |
| [MEOH](meoh/README.md) | Multi-objective heuristic evolution with dominance-masked AST similarity and a Pareto archive |
| [LLM-GA](llm_ga/README.md) | Four heuristic evolution operators with operator-specific rejection blacklists |
| [LLM4MOEA](llm4moea/README.md) | MOEA/D with LLM or learned linear variation for caller-defined multiobjective problems |
| [LLM genetic improvement](llm_gi/README.md) | LLM block mutations with random sampling and strict best-first local search |
| [LLM genetic improvement (2025)](llm_gi_2025/README.md) | Journal prompts, reversible add/remove patches, and independent searches over hot methods |
| [EoH-S](eoh_s/README.md) | Complementary heuristic sets with farthest-pair search, local refinement, and greedy CPI selection |
| [EvoPrompt](evoprompt/README.md) | Genetic and differential evolution of prompts |
| [EvoPrompting](evoprompting/README.md) | Metric-conditioned candidate evolution with parent retirement and soft prompt-tuning callbacks |
| [EvoX](evox/README.md) | Co-evolve artifacts and executable search strategies using stagnation feedback |
| [ExpeL](expel/README.md) | Experience gathering, cross-task insight voting, successful-task retrieval, and knowledge transfer |
| [GEPA](gepa/README.md) | Reflective prompt mutation, instance-wise Pareto selection, and system-aware merge |
| [HMAW](hmaw/README.md) | CEO → Manager → Worker prompt optimization with the original query at every stage |
| [In-context QD](in_context_qd/README.md) | Generate candidates from archive context to fill diverse niches and improve their measured fitness |
| [Interactive evolution](interactive_evolution/README.md) | Human-feedback-gated steady-state text evolution with tournament selection, crossover, and focused mutation |
| [Forest of Thought](fot/README.md) | Independent ToT/MCTSr trees, dynamic correction, sparse activation, and consensus/expert selection |
| [Graph of Thoughts](got/README.md) | Custom operation graphs with branching, aggregation, refinement, and local or model scoring |
| [LLMGP](llm_gp/README.md) | Model variation, optional model selection and replacement |
| [LLM-PSRO](llm_psro/README.md) | Pairwise tournaments, fictitious-play mixtures, and code-generated best responses |
| [LMCA](lmca/README.md) | Neutral mutation chains with validation repairs, fallback, and convergence analysis |
| [Meta-Prompting Protocol](meta_prompting/README.md) | Best-of-N generation, blind audits, textual prompt updates, and golden regression checks |
| [Meta-Prompting Scaffolding](meta_prompting_scaffolding/README.md) | Task-agnostic conductor, isolated experts, verification feedback, and optional Python execution |
| [Multiagent Debate](multiagent_debate/README.md) | Independent answers followed by synchronous peer revision rounds |
| [Multi-Agent Debate](mad/README.md) | Sequential opposing arguments, adaptive judging, and final answer extraction |
| [Meta-Reasoning Prompting](mrp/README.md) | Score reasoning methods, select the highest score, and execute the chosen method |
| [Minerva](minerva/README.md) | Independent solution sampling, final-answer grouping, and top-n majority voting |
| [LATS](lats/README.md) | UCT search with reversible environment feedback, value-guided rollouts, and reflections |
| [Let's Verify Step by Step](lets_verify_step_by_step/README.md) | Process reward best-of-N selection, first-error supervision, and active learning |
| [Optimizer](optimizer/README.md) | Generate, assess, and revise using explicit feedback |
| [Optimizing the Optimizer](optimizing_the_optimizer/README.md) | Measured heuristic-improvement dialogue and generic CMSA with age-weighted V1/V2 construction |
| [Plan of Thoughts](pot/README.md) | History-based UCT with continue/rollback/think actions and sequential rollouts |
| [Prompt Programming](prompt_programming/README.md) | Task specification, multipart metaprompts, and counterfactual fragment insertion |
| [Promptbreeder](promptbreeder/README.md) | Co-evolve task prompts, mutation prompts, and verified contexts through binary tournaments |
| [PROMST](promst/README.md) | Categorized human feedback, global beam search, and a five-Longformer score ensemble |
| [QUBE](qube/README.md) | Quality–uncertainty search across clustered islands |
| [Rationale-augmented ensembles](rationale_ensembles/README.md) | Fixed, shuffled, or sampled-rationale prompts with plurality voting |
| [RAP](rap/README.md) | World-model MCTS with action priors, confidence rewards, and answer aggregation |
| [ReEvo](reevo/README.md) | Pairwise and accumulated reflection guiding evolution |
| [Reflexion](reflexion/README.md) | Evaluated trials, verbal self-reflection, and bounded episodic memory |
| [Self-consistency](self_consistency/README.md) | Independent reasoning samples and majority voting over task-defined final answers |
| [SELF-REFINE](self_refine/README.md) | Same-model feedback and iterative refinement with full output history |
| [SPELL](spell/README.md) | Semantic prompt reproduction, exponential roulette selection, and elite preservation |
| [STaR](star/README.md) | Generate rationales, rationalize failures, and fine-tune the original model on correct solutions |
| [Strategy finding](strategy_finding/README.md) | Categorized signal discovery, confidence/risk selection, and a ten-ReLU numeric combiner |
| [TAUCHI-GPT](tauchi_gpt/README.md) | Local retrieval, task execution, optional reflection cycles, result memory, and task reprioritization |
| [Tree of Thoughts](tot/README.md) | Thought generation and value/vote guided BFS or DFS |
| [Theory of Mind](tom/README.md) | Editor profiles and geometric trait loss guide judge/editor meta-prompt revisions |
| [Zero-shot parent selection](zero_shot_selection/README.md) | Independent selector synthesis and validation, with reusable Kimi/GPT numerical selection rules |

[DECOMP](decomp/README.md) is a separate question-solving agent implementing
Decomposed Prompting with modular handlers, answer references, and recursion.

Each operation has its own prompt file; Python selects operations and Jinja renders
data without conditional branches. Caller inputs follow the annotated types, with
errors surfacing at use rather than through repetitive type guards.

Each package exports its agent class from `agent.py`. Results and configuration
remain specific to the algorithm. APEX also accepts an embedding callback; QUBE's
evaluator returns a score and behavior signature. See each README for its contract.

```python
from pathlib import Path
from slick import prompts
import llm_gp

async def optimize(task, provider, evaluate):
    prompts.TEMPLATE_ROOT = Path(llm_gp.__file__).resolve().parent / "prompts"
    agent = llm_gp.LLMGP(task, provider, evaluate)
    return await agent.run()
```

`evaluate` scores candidate text asynchronously. The application owns datasets,
execution environments, provider configuration, transport retries, and output
persistence. An optional `session=` passed to `run()` carries conversational
history. Slick's checked template root is process-global: configure it before
running an agent; simultaneous runs requiring different roots need separate
processes. Imports do not change the root.

The former benchmark applications, Docker workers, demo providers, and CLIs are
preserved in a [verified source archive](examples/legacy/README.md). Historical
runs and local environments remain in place. These new APIs and prompts are an
intentional redesign of those applications, not benchmark reproductions.
[Scout](scout/README.md) remains a separate OpenAlex exploration utility.

## Development

Use Python with the local Slick checkout and NumPy installed (APEX uses NumPy).
Existing `optimizer/.venv` provides this environment. The implementation-specific
requirements files install the adjacent Slick checkout from their directories.

```sh
optimizer/.venv/bin/python -B -m unittest discover -s tests
../slick/.venv/bin/ruff check .
../slick/.venv/bin/ruff format . --check
```

Tests share one [`ScriptedProvider`](tests/providers.py); agent implementations
contain no dummy providers. Tests verify decisions and contracts using unrelated
tasks, without paid generation or executing candidate code.

Development guidance: [original style comparison](docs/slick-style-report.md),
[style guide](skills/slick-development/references/style-guide.md),
[development skill](skills/slick-development/SKILL.md), and
[verification record](docs/slick-style-validation.md).
