# Forest of Thought

A problem-agnostic implementation of [Forest-of-Thought: Scaling Test-Time
Compute for Enhancing LLM Reasoning](https://arxiv.org/abs/2412.09078), grounded in
the authors' [official implementation](https://github.com/iamhankai/Forest-of-Thought).
`ForestOfThought` builds independent ToT or MCTSr trees, repairs uncertain
candidates, excludes inactive trees, and selects an answer by consensus or expert judgment.

## Use

Use the existing local Slick environment, or install the adjacent checkout with
`python -m pip install -e ../slick` from the repository root. No new dependencies.

```python
from pathlib import Path

from slick import prompts
import fot

# Configure once at application startup, before any prompt calls.
prompts.TEMPLATE_ROOT = Path(fot.__file__).resolve().parent / "prompts"

async def solve(task, provider):
    agent = fot.ForestOfThought(task, provider)
    result = await agent.run(
        trees=4,
        search="tot",       # or "mctsr"
        depth=3,            # ToT expansion layers
        breadth=2,          # retained ToT nodes per layer
        candidates=3,       # ToT proposals / MCTSr maturity threshold
        rollouts=4,         # MCTSr refinement iterations after initialization
        max_calls=200,
    )
    return result

# Await solve(your_task, your_provider) in your application.
# Handle result.answer is None, and inspect result.budget_exhausted.
# result.solution.content contains the selected artifact and supporting explanation.
```

The caller supplies a configured Slick provider, including model, temperature,
transport retries, and token limits. Imports do not change Slick's process-global
template root. Configure it with an absolute path for launch-directory independence;
applications using different roots concurrently need separate processes.

Each tree sees the same task and retrieved context, a different perspective, and
independent generations. Default perspectives cycle through construction, working
backward, decomposition, and testing assumptions. Supply `perspectives=(...)` to
tailor them. Sampling diversity depends on the provider configuration.

## Task boundary

`Candidate(content, answer=None)` holds the entire current solution, including
relevant previous steps. Partial ToT candidates use `answer=None`; completed
candidates carry a concise final answer separately from their content. Text is
nonblank and strips surrounding whitespace. This representation suits prose,
plans, expressions, and textual artifacts; use `answer_format` to specify the
output contract and `criteria` to specify what makes a candidate useful.

All domain-specific behavior is optional and injected into the constructor:

| Argument | Contract |
| --- | --- |
| `evaluate(candidate)` | Async; returns `Evaluation(score, confidence, valid=True, feedback="")`. Both numbers must be finite and in [0, 1]; higher scores are better. Replaces LM assessment. It receives both partial and complete candidates. |
| `retrieve(task)` | Async; returns relevant prior knowledge as text. Called once per run; concatenate your best matching question/answer example here (paper equations 3–4). |
| `correct(candidate, feedback)` | Async rule-based repair, replacing the LM correction call. Returns a replacement `Candidate`, or `None` to reject it. |
| `verify(candidate)` | Async correctness check for complete candidates. Only a true result establishes verified success and stops immediately. False permits further search; use `evaluate.valid=False` to reject unusable candidates. |
| `answer_key(answer)` | Synchronous canonicalization for voting, returning a hashable key. `None` rejects an unextractable answer. Default: stripped, case-sensitive text equality. |

For example, wrap your evaluator without changing the algorithm:

```python
async def solve_with_checks(task, provider, evaluate_candidate, check_solution):
    agent = fot.ForestOfThought(
        task, provider,
        evaluate=evaluate_candidate,  # async Candidate -> fot.Evaluation
        verify=check_solution,        # async Candidate -> bool
        answer_key=str.casefold,      # use only when case does not affect meaning
    )
    return await agent.run(search="mctsr", trees=4, rollouts=8)
```

Without a custom evaluator, quality and confidence are **LM estimates**. Slick's
text provider contract does not expose token logits. To use model probabilities,
provide confidence through the evaluator on [0, 1]; for mean log probability `l`,
`exp(l)` supplies the corresponding geometric-mean probability. Set the correction
threshold on that scale. A high score or consensus is not a correctness proof.

Callbacks own retrieval, datasets, execution isolation, evaluation costs, and
external resources. The agent never executes generated content. Caller inputs
are trusted; malformed generated outputs and invalid measured scores fail visibly.

## Search and decisions

- **ToT:** propose siblings, assess and conditionally repair each candidate, then
  retain the globally highest-scoring `breadth` nodes. Stable ties preserve order;
  identical corrected candidates are deduplicated within the expanded layer.
  A layer with no valid candidates deactivates its tree. Complete candidates are
  retained without expansion. At `depth`, finalize and assess the best leaf.
  `depth=0` generates one complete solution directly.
- **MCTSr:** initialize a complete solution, then repeatedly select an eligible
  node by UCB, resample its reward, and refine using its assessment feedback.
  Reward is half the sum of the minimum and mean sampled scores. Internal-node
  UCB uses the average of its reward and its best child's reward. Sample counts
  serve as visits. Nodes with `candidates` children remain expandable only when
  all children have lower conservative reward. An invalid resampled parent or
  failed refinement deactivates the tree. `rollouts=0` returns the assessed initial solution.
- **MCTSr final choice:** use the official 0.5 × minimum reward + 0.3 × sample
  count + 0.2 × UCB formula. Quality is converted to the official [0, 100] scale
  before UCB and final selection; `exploration=1.4` is on that scale. Distinct
  node IDs preserve ancestry even when two generations produce identical text.
- **Dynamic correction:** confidence strictly below `correction_threshold`
  triggers one rule or LM repair followed by reassessment. Accept a valid repair
  when confidence improves, or when it rescues an invalid original. Otherwise
  retain the original valid candidate. A repair cannot drop a completed answer.
  This is conditional single-step correction, not a fixed self-refinement loop.
- **CGED:** each active tree gets one vote. Default `consensus="majority"`
  requires more than half of the completed active trees. Otherwise an expert
  chooses an existing active solution by its checked index. `"plurality"` matches
  the official repository's unique-most-frequent rule; tied leaders use the expert.
  Voting returns the first representative of an equivalent-answer group.
- **Early stopping:** a caller-verified answer ends search immediately, including
  remaining siblings. Agreement stops further trees only after exceeding half of
  the *planned* tree count, so remaining trees cannot overturn that majority.
  Model confidence alone never triggers success.

Only completed, usable tree outputs enter CGED. There is no fallback to an
inactive tree or an invented expert answer. An entirely inactive forest returns
`solution=None`, `answer=None`, and `decision="no_solution"`.

## Budgets and records

`max_calls` caps attempted Slick prompt operations across the entire forest,
including scoring, repair, finalization, and expert judgment. There are no hidden
retries. A provider's internal transport retries and token costs, and all callback
costs, remain the caller's responsibility. Trees, depth, breadth, candidate count,
and rollouts bound the search separately.

Budget exhaustion deactivates the unfinished tree and returns the completed tree
records. Available completed solutions can still reach consensus; this is marked
`budget_exhausted=True`, not reported as a completed forest. If an expert call is
needed but cannot fit, the result has no selected solution and
`decision="budget_exhausted"`. The call cap is never exceeded.

`Result` includes the selected solution, tree records, decision, attempted call
count, and budget flag. The agent also retains `calls` (operation, tree, rendered
prompt, raw response, error), `assessments`, `corrections`, `callbacks`, beam
`history`, and MCTSr `nodes`. Parsing and postprocessing failures retain their raw
responses. Provider, schema, evaluator, and callback errors propagate immediately;
inspect the records after catching them. Records reset on each run. One instance
supports one run at a time; all calls are sequential and use no Session history.

## Source mapping and deliberate adaptations

Inspected official revision: `f8d6e9215f1ac7a2d43683e76591c612b2c7c288`.
This implementation adapts the algorithms into Slick; it does not import the
official GPU stack or benchmark scripts.

| Reference | Used here |
| --- | --- |
| [methods/bfs.py](https://github.com/iamhankai/Forest-of-Thought/blob/f8d6e9215f1ac7a2d43683e76591c612b2c7c288/methods/bfs.py) | Independent trees, proposal/correction, beam selection, sparse activation, verified early termination |
| [run_with_mcf_stop_noearly.py](https://github.com/iamhankai/Forest-of-Thought/blob/f8d6e9215f1ac7a2d43683e76591c612b2c7c288/run_with_mcf_stop_noearly.py) | `filter_mature_node`, `compute_ucb`, `update_ucb`, `get_tree_ans`, MCTSr orchestration and forest stopping |
| [models/load_local_model.py](https://github.com/iamhankai/Forest-of-Thought/blob/f8d6e9215f1ac7a2d43683e76591c612b2c7c288/models/load_local_model.py) | Confidence-triggered single repair and confidence-improvement acceptance |
| [utils/early_stop.py](https://github.com/iamhankai/Forest-of-Thought/blob/f8d6e9215f1ac7a2d43683e76591c612b2c7c288/utils/early_stop.py) | Unique-plurality aggregation, available explicitly alongside the paper's majority rule |

The official README credits [ToT](https://github.com/princeton-nlp/tree-of-thought-llm)
and [MCTSr](https://github.com/naivoder/MCTSr). The algorithm details above follow
the FoT authors' integrated implementation. The existing sibling `tot/` supplies
the local search/style reference; FoT needs its own loop to assess, repair, and
deactivate candidates during expansion without changing that agent's contract.

The paper's Algorithms 1–2 and sections 3.1–3.3 define the forest boundary.
Domain prompts, answer extraction, arithmetic correction, and retrieval are
intentionally generalized. Complete candidate snapshots replace arithmetic
strings. Structured JSON replaces benchmark delimiters. Assessment feedback
supplies the critique for MCTSr refinement without a separate hint-generation
call. The injected-rule rejection and invalid-original rescue policies are explicit
extensions. MCTSr omits the official synthetic refusal root; it never treats
"I don't know" as a fabricated competing solution. It also omits the script's
high-reward success shortcut and retains only caller-verified early success.
Knowledge is retrieved once without recursively prepending earlier questions.

These are algorithm and interface checks, **not a reproduction of benchmark
accuracy or proof that FoT improves your chosen model**. No paid calls or model
downloads are needed for verification:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_fot
../slick/.venv/bin/ruff check fot tests/test_fot.py
../slick/.venv/bin/ruff format fot tests/test_fot.py --check
```
