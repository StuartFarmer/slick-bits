# Reasoning via Planning (RAP)

Problem-agnostic Slick implementation of **Reasoning with Language Model is
Planning with World Model** ([Hao et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.507/)).
An LLM proposes actions and independently simulates successor states. Monte Carlo
Tree Search explores those imagined trajectories using local rewards and future
returns. Generated actions are never executed by this package.

## Use

Use the existing `optimizer/.venv` environment or install the adjacent Slick
checkout with `python -m pip install -e ../slick`. No additional dependencies.
The checked local Slick checkout declares `slick-ai` 0.3.0.

```python
from pathlib import Path
from slick import prompts
import rap

# Configure once at application startup; importing rap does not change this global.
prompts.TEMPLATE_ROOT = Path(rap.__file__).resolve().parent / "prompts"

async def solve(task, provider):
    agent = rap.RAP(
        task=task,
        provider=provider,
        actions="An incremental step appropriate to this task",
        world="Retain the current facts, intermediate results, and remaining requirements",
        criteria="Correctness and progress toward the requested result",
    )
    result = await agent.run(
        rap.State(content="Initial facts and constraints go here"),
        iterations=10, candidates=4, depth=5, confidence_samples=4,
        aggregate=True,
    )
    return result, agent
```

Supply your own Slick provider, complete task, and domain instructions. Omitting
the initial state uses the task text. `actions` specifies allowed reasoning steps,
subquestions, commands, or candidate revisions. `world` specifies how to represent
their consequences. `criteria` specifies what makes a step useful. Include any
demonstrations in those instructions. The same provider serves all three roles.
Configure sampling temperature (the paper uses 0.8), token limits, and transport
retries in your provider.

`State.content` is the complete textual state, including necessary history. An
action may be a subquestion and its successor state the accumulated intermediate
answers; a plan may use configurations and moves instead. The next proposal sees
the task and current state, not sibling states or implicit conversation history.
Model-produced prose is stripped at the edges; these are not byte-preserving
artifact contracts.

`State.terminal` means this branch has finished, including a dead end. `answer`
holds the final output; terminal states without it are classified as dead ends.
Nonterminal generated states cannot contain an answer.
Both fields are model predictions, not independently established correctness.
For plans and proofs, the output can be the ordered `result.best.actions` and
`result.best.states`; leave `aggregate=False` when a complete trace is required.

## Rewards and confidence

Each distinct action receives an independent self-evaluation score in [0, 1].
Its cheap prior is `helpfulness**reward_alpha * prior_confidence**(1-reward_alpha)`.
Defaults are `reward_alpha=0.5` and `prior_confidence=0.8`.

When that action is explored, independently sample `confidence_samples` successor
states from the same context. Group them by `state_key`, select the largest group,
and retain its first state. Confidence is its count divided by all samples; ties
use first occurrence. The full default reward replaces prior confidence with
this measured agreement. With one sample, confidence is always 1.

The default key compares `(content, terminal, answer)` exactly after whitespace
normalization. Different prose expressing the same result will count separately.
Provide a synchronous, hashable `state_key(state)` to group equivalent states;
it must preserve differences relevant to continuation and termination. For
structured domains, canonicalize the facts or intermediate answer yourself.
`answer_key(text)` separately normalizes final answers for aggregation and
defaults to `str.strip`.

To use a task-specific heuristic, verifier, action likelihood, or another reward
combination, supply an async evaluator:

```python
async def evaluate(transition: rap.Transition) -> float:
    # score_transition is your own evaluation adapter.
    return await score_transition(
        transition.source.content,
        transition.action,
        transition.target.content,
    )

agent = rap.RAP(task, provider, evaluate=evaluate)
result = await agent.run()
```

The callback also receives `confidence` and `helpfulness`. Its finite,
higher-is-better return **replaces** the default full reward. It does not replace
the proposal prior or the LLM world model. Use a compatible reward scale for
useful UCT behavior; adjust `exploration_weight` for your scale. Evaluation runs
once per materialized edge and is cached. The caller owns any tools, execution
isolation, and search-time data; keep held-out evaluation outside the search.

## Algorithm decisions

1. Select through the retained tree using
   `Q + exploration_weight * sqrt(log(parent.visits) / max(1, visits))`.
   Unvisited actions use their finite prior for Q, as in the official code.
2. Sample `candidates` actions independently, deduplicate exact normalized strings
   in order, and score the unique actions. Keep successor states unevaluated until
   selection or simulation follows those edges.
3. Simulate greedily by choosing the largest cheap prior at each expansion.
   Materialize only that successor, compute its full reward, and continue until
   terminal, depth limit, or no actions. Keep rollout siblings for later iterations.
4. Backpropagate each edge's suffix return, including its own reward. Defaults
   implement paper Eq. 2: mean rewards within a suffix, maximum across observed
   suffix returns. `cum_reward=sum, calc_q=statistics.fmean` selects conventional
   sum/mean backup instead. The root's nonexistent incoming reward is excluded.
5. Choose the highest-scoring terminal trace observed in an iteration. Optional
   RAP-Aggregation sums edge rewards once per distinct descendant answer, matching
   the official LLM Reasoners `edge` policy. Repeated visits are not extra votes;
   a shared edge counts once for each answer reachable beneath it.

The full iteration budget runs even after finding a terminal answer. Terminal
paths may be revisited to update visit counts without new provider calls.
Depth counts actions; the final allowed action is simulated, but no further
actions are proposed. A depth cutoff never creates a terminal answer. Duplicate
states in different branches are not globally merged.

## Results, budgets, and failure policy

- `best`: highest-scoring terminal `Trace` with an answer, or `None`. This is a
  model-proposed answer, not an independently verified solution.
- `partial`: highest-scoring unfinished or dead-end trace, or `None`.
- `answer`: aggregated canonical answer when aggregation yields votes; otherwise
  the best trace's answer. An aggregated answer can differ from `best`'s answer.
- `answer_weights`: unnormalized aggregation totals, not probabilities.
- `traces`, `iterations`: immutable trace snapshots and completed iteration count.

Each trace contains ordered states, actions, rewards, its reduced score, and a
`terminal`, `depth_limit`, or `dead_end` reason. Inspect `agent.root` for the tree
and each node's priors, full reward, transition, returns, and visit count.
An initially terminal state returns immediately with zero iterations/calls.
Zero iterations yields no trace. Zero depth/candidates produces partial traces
without provider calls. Caller configuration is trusted rather than prevalidated.

One expansion costs `candidates` proposal calls plus one assessment per unique
action. One new transition costs `confidence_samples` prediction calls and one
reward evaluation. At most `iterations * depth` transitions and expansions are
needed; caching often reduces that. These are algorithm budgets, not token,
transport-retry, or wall-clock limits. Full response and trace records grow with
the search budget.

`agent.calls` records prompts, raw responses, and errors **before** Slick parses
and validates generated output. `agent.evaluations` records full-reward attempts,
values, and failures. Parsing, invalid generated fields, provider tool requests,
nonfinite rewards, and evaluator exceptions abort the run without repair or
retry. Completed records remain inspectable; a later run resets them and the tree.
Use one run at a time per instance. There are no mutable Sessions.

The absolute template root works from the repository root or another directory
with this repository on `PYTHONPATH`. Slick's root is process-global: configure it
once and use separate processes for concurrent applications with different roots.

## Official sources and adaptations

The supplied APET URL implements **Autonomous Prompt Engineering in Large Language
Models**, already represented in `../apet/`; it is not this paper's RAP algorithm.
The paper's `Ber666/llm-reasoners` reference is now maintained at
[maitrix-org/llm-reasoners](https://github.com/maitrix-org/llm-reasoners).
This implementation uses and adapts these inspected official sources:

- [`reasoners/algorithm/mcts.py`](https://github.com/maitrix-org/llm-reasoners/blob/f94e5ac2cb9788c3d7d7dbf2173884ed4088e4b2/reasoners/algorithm/mcts.py):
  lazy expansion, UCT priors, greedy simulation, suffix backup, and edge aggregation.
- [`examples/RAP/gsm8k`](https://github.com/maitrix-org/llm-reasoners/tree/f94e5ac2cb9788c3d7d7dbf2173884ed4088e4b2/examples/RAP/gsm8k):
  sampled state confidence, geometric rewards, and mean/max backup configuration.
- [Original RAP](https://github.com/Ber666/RAP/tree/774817c228b3d5ddfc18de2318f3476128ecf6eb):
  cross-check of the original MCTS and aggregation. Its older aggregation divides
  edge weights by descendant depths; the newer official GSM8K example uses `edge`,
  which is the implemented policy here.

The adapted implementation carries the upstream [Apache-2.0 license](LICENSE).
Changes are deliberate: asynchronous Slick boundaries, generic JSON states and
actions, explicit cutoff/failure records, and injectable evaluation replace the
benchmark-specific prompts and GPU infrastructure. State confidence uses a fixed
sample budget rather than the official optional early stopping. Self-evaluation
uses a generated rating, **not** the paper's next-token `P(Yes)`; Slick's base
provider does not expose token logits. No action likelihood is fabricated.
Best-trace selection follows the paper's best-iteration strategy and excludes
the root's dummy reward when averaging.

## Verification

```sh
optimizer/.venv/bin/python -m unittest tests.test_rap -v
```

The shared scripted provider exercises backtracking, exact suffix returns, cached
transitions, sampled confidence, aggregation, budgets, malformed output, failure
records, and every external template. These checks establish algorithm and API
behavior; they do not reproduce the paper's benchmark accuracy. No paid model
calls or benchmark-specific dependencies are needed.
