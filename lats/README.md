# Language Agent Tree Search (LATS)

A problem-agnostic implementation of [Zhou et al., ICML 2024](https://arxiv.org/abs/2310.04406),
adapted from the authors' [official implementation](https://github.com/lapisrocks/LanguageAgentTreeSearch).
It combines UCT selection, sampled thought/action pairs, external observations,
LM and self-consistency values, greedy rollouts, reward backpropagation, and
reflection memory. No new dependencies beyond the existing Slick installation.

## Use

```python
from pathlib import Path
from slick import prompts
import lats

# Configure once at application startup. Imports leave this global unchanged.
prompts.TEMPLATE_ROOT = Path(lats.__file__).resolve().parent / "prompts"

async def solve(task, provider, evaluate):
    agent = lats.LATS(
        task=task,
        provider=provider,
        evaluate=evaluate,
        actions="Describe your accepted commands, reasoning steps, or answer format here",
        criteria="Describe the task's success conditions here",
        observation="Initial environment context, if any",
    )
    result = await agent.run(
        iterations=50, candidates=5, depth=7,
        exploration_weight=1.0, lm_weight=0.5,
    )
    return result, agent

# result.best is None if no action was evaluated.
# Otherwise result.best.trajectory contains ordered Step(action, feedback) records.
# result.best.trajectory[-1].action.content is the last command/candidate.
# result.stop_reason == "success" means the evaluator declared success.
```

Supply an existing Slick provider and an async callback with this signature:

```python
async def evaluate(history: tuple[lats.Step, ...], action: str) -> lats.Feedback:
    # Restore your environment to history, then apply action. Implement these
    # operations in your own adapter; LATS does not own environment infrastructure.
    await environment.restore(history)
    observation, reward, done, success = await environment.step(action)
    return lats.Feedback(observation, reward, terminal=done, success=success)
```

The callback receives the **parent trajectory**, excluding the proposed action.
Every sibling starts from that same history. Restore a checkpoint, reset and
replay the recorded actions, or use a pure evaluation function. Merely changing
the model's text context does not restore a mutable environment. Stochastic
environments need an adapter that preserves the particular branch state, including
random state where needed. Duplicate action strings share one sampled outcome.

`Feedback.reward` is the finite, higher-is-better outcome score for that state,
not an incremental reward to sum along the path. Normalize it to [0, 1] for
comparability with the LM heuristic. `terminal=True` ends that branch even after
failure; `success=True` stops the entire search regardless of the reward number
or terminal flag. Express invalid actions as feedback if they are recoverable;
raise an exception when the search should abort. The model cannot declare success.

For pure reasoning, accept textual reasoning steps and check final answers in
the callback. For iterative artifact generation, use the no-rollout variant below.
The agent never executes generated content. The caller owns tools, execution
isolation, datasets, environment restoration, model configuration, and persistence.

Slick's template root is process-global; configure the absolute path above once.
Independent applications needing different roots concurrently require separate
processes. Run from the repository root, or put it on `PYTHONPATH` when importing
from another directory. The templates themselves are independent of launch directory.

## Search decisions

1. Select an unexhausted leaf using
   `mean_return + exploration_weight * sqrt(log(parent.visits) / visits)`.
   Unvisited children have infinite UCT; stable ties use sampling order.
2. Independently sample `candidates` actions with identical context. Count exact
   action-string matches before deduplicating siblings, keeping the first thought.
   Different thoughts with the same action contribute to the same count.
3. Apply each distinct action through the callback. For nonterminal children use
   `lm_weight * LM_score + (1 - lm_weight) * action_count / candidates`.
   LM scores are in [0, 1]. Terminal values use measured reward directly.
   `lm_weight=0` skips LM value calls. Action bytes are preserved, including whitespace;
   blank actions are rejected. There is no semantic-equivalence clustering.
4. Simulate by repeating expansion/evaluation and taking the highest heuristic
   child until terminal or depth limit. All rollout siblings remain in the tree.
   Any evaluated success stops immediately, including an off-path sibling; later
   sibling environment calls are skipped. The sampling batch has already completed.
5. Backpropagate one endpoint return along its whole path, incrementing each node's
   own visit count. The first return replaces its initial heuristic; subsequent
   values are empirical mean returns. Do not divide that mean by visits again.
6. Reflect on unsuccessful endpoints, recording the whole trajectory, reward,
   termination reason, and reflection. Later action **and value** prompts receive
   this memory. Terminal siblings not selected for rollout are processed when
   subsequently selected. Each failure is reflected once.

Depth-limited trials use `cutoff_reward=0.0` by default and are explicitly labeled
as interrupted, not successful. Terminal nodes and processed depth-limit leaves
are exhausted. Closed subtrees are retained for inspection and skipped during
selection. When the finite sampled tree is exhausted, search ends; it does not
resample closed parents. In particular, `candidates=1` can exhaust after one
rollout. This bounds work and avoids the official HotPotQA reselection loop's
nontermination when no selectable node remains.

## Complete-candidate variant

For tasks where every action is a complete candidate (a program, plan, draft,
configuration, etc.), section 5.2 omits simulation and uses measured candidate
quality directly:

```python
async def evaluate_candidate(history, candidate):
    # score_candidate is your own async evaluator, isolated when necessary.
    score, feedback, accepted = await score_candidate(candidate)
    return lats.Feedback(feedback, score, success=accepted)

agent = lats.LATS(task, provider, evaluate_candidate,
                 actions="Return a complete revised candidate satisfying the task")
result = await agent.run(iterations=8, candidates=5, depth=8,
                         simulate=False, lm_weight=0.8)
```

Each new candidate's reward is backed up once, without a rollout, and unsuccessful
candidates receive reflections. Leave unsuccessful candidates **nonterminal** so
they can be refined: child actions replace the candidate artifact while the path
provides revision history. LM/consistency values remain recorded; UCT uses the
measured means once candidates have been evaluated. The callback's success signal
must use only search-time evidence; keep held-out tests outside the search.

## Budgets, results, and failures

`iterations` bounds selection rounds, `depth` bounds actions along a path, and
`candidates` bounds samples per expansion. These are not token or transport limits.
An expansion makes `candidates` generation calls, at most that many environment
calls, and one LM value call per distinct nonterminal child. Each unsuccessful
endpoint costs one reflection call. Simulation expands at most `depth` parents
per iteration; the no-rollout variant expands at most one.

`Result` reports `best`, attempted iterations/expansions, and `stop_reason`:
`success`, `budget`, or `exhausted`. With no success, simulation returns the highest
reward terminal node if any, otherwise the highest reward partial node; ties use
the original heuristic and then insertion order. No-rollout mode returns the best
measured candidate across the entire tree. A partial fallback is not a solution.
The result's best node need not have been selected for a rollout or visited yet.

Inspect `agent.root`/`agent.nodes` for the retained tree, `agent.memory` for failure
reflections, `agent.calls` for prompts/raw responses/errors, and `agent.assessments`
for environment attempts/feedback/errors. Raw responses are captured before JSON
parsing and generated-output validation. Records are in memory and reset per run;
an agent supports one run at a time. Full failure history grows with the budget.

Malformed JSON, blank actions/reflections, invalid LM scores, nonfinite measured
rewards, and provider/evaluator exceptions abort without retries or silent repairs.
Attempt records remain available after an exception. Caller configuration is trusted.
There is no shared conversational Session: explicit tree paths and reflection
memory supply context. Provider tool requests are rejected; tools belong in the
reversible environment callback.

## Official source and intentional adaptations

Reviewed official commit **`853d81614607dd27433faf17c7b0a7d660f95d22`**:

- [hotpot/lats.py](https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/853d81614607dd27433faf17c7b0a7d660f95d22/hotpot/lats.py):
  node/trajectory representation, UCT selection, sampling, value prompts,
  backpropagation, and failed-trajectory memory.
- [webshop/lats.py](https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/853d81614607dd27433faf17c7b0a7d660f95d22/webshop/lats.py):
  branch restoration before action execution and environment-reward termination.
- [programming/mcts.py](https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/853d81614607dd27433faf17c7b0a7d660f95d22/programming/mcts.py):
  complete-candidate refinement and direct measured-return backpropagation.

This is an adaptation of those algorithm flows to Slick, with the upstream
[MIT notice](UPSTREAM_LICENSE) retained. Benchmark dependencies, global environment
objects, test generation, and execution infrastructure are not imported.

The paper takes precedence where these implementations differ: the inspected
HotPotQA code divides an already averaged value by visits in UCT; this uses
equation 1's mean directly. Its HotPotQA/WebShop evaluators do not explicitly
combine duplicate frequency with LM scores; this implements equation 2, defining
SC as empirical action frequency. Its rollout routines do not attach all sampled
children to the retained tree; this stores them, as described by the paper.
Unvisited UCT is explicitly infinite, visits start at zero, and terminal feedback
overrides heuristics. Depth cutoffs and finite-tree exhaustion have explicit policies.

Prompts intentionally replace benchmark examples and delimiter parsing with
task-supplied instructions and typed Slick JSON for action/value outputs. Reflections
remain prose. The no-rollout variant uses each candidate's measured score once;
it does not reproduce upstream repeated test execution or held-out-test access.
These are documented algorithm choices, not a claim of byte-identical replication.

## Checks

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_lats
../slick/.venv/bin/ruff check lats tests/test_lats.py
../slick/.venv/bin/ruff format lats tests/test_lats.py --check
```

Tests use the shared `tests.providers.ScriptedProvider` through actual Slick
rendering/parsing. They exercise backtracking, reflection context, self-consistency,
UCT/mean updates, success detection, direct candidate evaluation, budgets,
exhaustion, malformed outputs, error records, and template loading from another
directory. These checks establish algorithm plumbing and decisions; they do not
reproduce the paper's benchmark scores or measure live-model performance.
