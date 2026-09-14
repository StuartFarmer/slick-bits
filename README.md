# Slick bits

Problem-agnostic algorithm classes built with [Slick](../slick). Each class
owns its prompt methods and a short async `run()` coordinating named algorithm phases. Callers supply the task,
provider, and async evaluator; each implementation owns its own `prompts/` folder.

| Agent | Flow |
| --- | --- |
| [AEL](ael/README.md) | Population evolution with crossover, mutation, and elitism |
| [APEX](apex/README.md) | Sentence-level beam search with history and LinUCB selection |
| [Boosting of Thoughts](bot/README.md) | Parallel weighted trees, chain aggregation, and accumulated trial-and-error feedback |
| [EoH](eoh/README.md) | Five strategies for evolving candidate ideas and content |
| [EvoPrompt](evoprompt/README.md) | Genetic and differential evolution of prompts |
| [LLMGP](llm_gp/README.md) | Model variation, optional model selection and replacement |
| [Multiagent Debate](multiagent_debate/README.md) | Independent answers followed by synchronous peer revision rounds |
| [Multi-Agent Debate](mad/README.md) | Sequential opposing arguments, adaptive judging, and final answer extraction |
| [LATS](lats/README.md) | UCT search with reversible environment feedback, value-guided rollouts, and reflections |
| [Optimizer](optimizer/README.md) | Generate, assess, and revise using explicit feedback |
| [Plan of Thoughts](pot/README.md) | History-based UCT with continue/rollback/think actions and sequential rollouts |
| [QUBE](qube/README.md) | Quality–uncertainty search across clustered islands |
| [ReEvo](reevo/README.md) | Pairwise and accumulated reflection guiding evolution |
| [Tree of Thoughts](tot/README.md) | Thought generation and value/vote guided BFS or DFS |

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
