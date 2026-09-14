# ExpeL

Problem-agnostic Slick implementation of **ExpeL: LLM Agents Are Experiential
Learners** (Zhao et al., AAAI 2024). It gathers experiences through ReAct and
Reflexion, extracts cross-task insights, and retrieves successful demonstrations
for single-attempt evaluation. No model weights are updated.

## Use with your own problem

Configure the local prompt root once at application startup. Supply your own
provider, async environment callbacks, and async text embedder:

```python
from pathlib import Path

import expel
from slick import prompts
from expel import ExpeL

prompts.TEMPLATE_ROOT = Path(expel.__file__).resolve().parent / "prompts"

async def learn(task_interface, provider, reset, step, embed, training, held_out):
    agent = ExpeL(task_interface, provider, reset, step, embed)
    result = await agent.run(
        training, held_out,
        max_retries=3, max_steps=20, chunk_size=8, k=3, seed=0,
    )
    return agent, result
```

The callbacks define the problem:

| Input | Contract |
| --- | --- |
| `task_interface: str` | Describe the action syntax, goal, and constraints for this task family. |
| `reset(task: str) -> str` | Async; start a fresh episode for this task and return its initial observation. |
| `step(action: str) -> Outcome` | Async; execute the action and return observation, termination, success, reward, and feedback. |
| `embed(text: str) -> Sequence[float]` | Async; return a consistent embedding of the task description. |
| `training`, `held_out` | Sequences of task descriptions. Identical strings identify the same task for pairing and retrieval. |

For example, an adapter for an existing environment can return:

```python
from expel import Outcome

async def step(action):
    observation, reward, terminated, truncated, info = await environment.step(action)
    return Outcome(
        observation=str(observation),
        done=terminated or truncated,
        succeeded=info["success"],
        reward=reward,
        feedback=info.get("feedback", ""),
    )
```

Map `success` from your evaluator's actual result; termination alone does not mean
success. A failed terminal episode still gets training retries. A successful
episode stops immediately even if `done` is false. Partial rewards never imply
success. `max_steps` bounds environment actions, including the final action.

Your environment owns reset semantics, tools, action validation, async evaluation,
and any execution isolation. The agent neither executes generated Python nor
constructs benchmark environments. For a non-interactive problem, make each episode
a single candidate action and have `step` evaluate it with `done=True`.

## Learning and memory

The phases are also available independently:

```python
await agent.gather(training, max_retries=3, max_steps=20)
await agent.extract_insights(chunk_size=8, seed=0)
result = await agent.evaluate(held_out, max_steps=20, k=3)
```

- `gather` uses fixed `manual_examples`, with no learned cross-task insights.
  It records every completed attempt, succeeds early, and generates a reflection
  only when another retry is available. Reflections accumulate within one task
  and reset for the next task; failed trajectories are not replayed to the actor.
- `extract_insights` rebuilds from constructor-supplied seed insights and the
  current pool. Every same-task success/failure pair is compared first. Then the
  first success per distinct task is shuffled without replacement into chunks,
  including the final smaller chunk. Failed-only tasks remain in memory but do
  not supply extraction evidence. Reflections are excluded from extraction.
- `ADD` starts at importance 2; `EDIT`/`UPVOTE` add 1; `DOWNVOTE` subtracts 1.
  Rules reaching zero disappear. At most four operations can change a snapshot,
  and each existing rule can be targeted once. Valid updates are committed as
  one batch and sorted by importance. `NONE` is an explicit no-op.
- `retrieve` uses exact inner product on **task descriptions**, returning the
  first success from each of the top-k distinct tasks. Failed trajectories are
  ineligible. Equal scores preserve pool order. Embeddings are cached by task
  text; use a new agent or clear `agent.embeddings` when changing the embedder.
- `evaluate` fixes the retrieved examples for the whole episode and supplies all
  insights. Each evaluation task gets exactly one attempt and no reflection.
  Evaluation trajectories never enter the training pool. `k=0` uses fixed manual
  examples instead of retrieval; leave those empty for insights-only evaluation.

`manual_examples` is a tuple of caller-verified successful `Experience` objects;
it seeds the pool as in Algorithm 1. You can inspect or persist `agent.pool` and
`agent.insights`, or supply existing experiences by extending the pool before
extraction. All internal experience/insight records are frozen dataclasses.

`run` appends training experience, rebuilds insights, and evaluates. Repeated
gathering accumulates experience; repeated extraction rebuilds counts instead of
silently counting the previous extraction twice. Use a new agent for independent
experiments. Callers select disjoint training/evaluation data and compatible task
families; there is no automatic fold splitting or environment-type filtering.

The returned `Evaluation` contains immutable `experiences` and `success_rate`
(0 for an empty evaluation set). Its experiences retain the full trajectory,
success, final reward, feedback, and action count. Zero step budget creates an
empty failed episode after reset; zero retries still permits the initial attempt.

## Embeddings

The paper uses `sentence-transformers/all-mpnet-base-v2` with FAISS. This version
keeps the embedder injectable and computes exact top-k inner products using the
standard library, with no vector database dependency. An application using the
paper's embedder can supply:

```python
import asyncio
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

async def embed(text):
    vector = await asyncio.to_thread(encoder.encode, text, normalize_embeddings=True)
    return vector.tolist()
```

`sentence-transformers` is optional application infrastructure, not an added
dependency. Normalized vectors make inner product equivalent to cosine similarity;
normalization is controlled by the caller. The agent uses the vectors as supplied,
rejects non-finite similarities, and reports dimension mismatches. The linear
scan is intended for modest experience pools; use a vector index when scale
justifies it. Insights and logs are not token-truncated or retrieved individually.

## Transfer to a related problem

```python
from expel import ExpeL, Insight

paragraph = await agent.adapt_insights(target_interface, target_examples)
target = ExpeL(
    target_interface, provider, target_reset, target_step, embed,
    manual_examples=target_examples,
    insights=(Insight(paragraph),),
)
transferred = await target.evaluate(target_tasks, k=0)
```

`target_examples` is a tuple of successful `Experience` records. Adaptation uses
the source insights and these target demonstrations, returns a concise paragraph,
and leaves source state intact. The new agent uses the target callbacks and fixed
target examples, with no retrieval from the source experience pool. Passing no
target examples supports the paper's transfer-without-demonstrations variant.

## Slick boundaries and failures

Checked against the adjacent `slick-ai` 0.3.0 checkout. Each operation has a local
Jinja template: `act`, `reflect`, `compare`, `summarize`, and `transfer`. The shared
local `rules.j2` renders the voting contract. Python owns all control flow.

Actions use explicit `output_type=Action` with an embedded JSON schema and
keyword-only `generated` validation. Action bytes are preserved after checking
for nonblank text. Reflection and transfer consume plain text; insight extraction
keeps the paper's tagged operation format. Invalid JSON, blank actions/prose,
invalid rule indices, duplicate changes, inconsistent vote text, duplicate rules,
and provider tool requests raise instead of being silently repaired or retried.

Pass optional `reflection_provider` and `insight_provider` for separate models;
otherwise all operations use `provider`. The insight provider also handles transfer.
Every prompt call receives exactly one explicit `provider=`. No mutable Slick
Session is shared; full episode context is passed explicitly. Providers should
not retain implicit conversation history. One operation at a time per agent.

`agent.calls` records operation, rendered prompt, raw response when available,
and generation/validation error before rejection. `len(agent.calls)` counts
attempted model operations, excluding transport retries hidden inside providers
and work inside callbacks. Callbacks and providers propagate errors immediately;
only ordinary environment-declared failure triggers a training retry. The latest
partial `agent.trajectory`, completed pool entries, successful insight batches,
and completed `agent.evaluations` remain inspectable after an exception. A failed
environment callback does not invent an evaluated experience or model failure.
Persistence and transport retry policy belong to the caller.

The template root is process-global. Configure it once before rendering and use
separate processes for concurrent implementations needing different roots.
Importing `expel` does not modify it. Explicit rendering requires owner binding,
for example `await ExpeL.transfer.render(agent, target_interface, target_examples)`.

## Official implementation and deliberate adaptations

The paper names [LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL) as its
official implementation. Source inspected and used at commit
[`e41ec9a24823e7b560c561ab191441b56d9bcefc`](https://github.com/LeapLabTHU/ExpeL/commit/e41ec9a24823e7b560c561ab191441b56d9bcefc).

| Official source | Algorithm used here |
| --- | --- |
| [`agent/reflect.py`](https://github.com/LeapLabTHU/ExpeL/blob/e41ec9a24823e7b560c561ab191441b56d9bcefc/agent/reflect.py) | Failure reflection, bounded retries, fresh episodes, task-local reflection memory. |
| [`agent/react.py`](https://github.com/LeapLabTHU/ExpeL/blob/e41ec9a24823e7b560c561ab191441b56d9bcefc/agent/react.py) | Generated actions interleaved with environment observations and independent environment success. |
| [`agent/expel.py`](https://github.com/LeapLabTHU/ExpeL/blob/e41ec9a24823e7b560c561ab191441b56d9bcefc/agent/expel.py) | Experience storage, `create_rules` comparison/chunk order, `update_rules` counts and sorting, successful task retrieval. |
| [`prompts/templates/human.py`](https://github.com/LeapLabTHU/ExpeL/blob/e41ec9a24823e7b560c561ab191441b56d9bcefc/prompts/templates/human.py) | Separate comparison/success prompts and numbered rule operations. |

This is a deliberate generic adaptation, not a drop-in wrapper or benchmark
reproduction. The benchmark-specific LangChain prompts and parsers are replaced
by local generic Slick templates and a structured action boundary. Transfer is
implemented from the supplied paper's Section 4.4/Figure 4; the inspected repository
contains the FEVER transfer guidance insertion but no general adaptation routine.

The official code calls UPVOTE/DOWNVOTE `AGREE`/`REMOVE`, uses substring matching
and forgiving parsing, and applies a -3 decrement when its rule list is full.
Here the paper's operator names and constant -1 decrement apply; generated IDs
and rule text must match, and invalid batches fail atomically. There is no rule
cap, pruning heuristic, automatic context trimming, or long-context fallback.
The official FAISS construction does not specify its distance strategy; this
implementation explicitly follows the paper's maximum-inner-product objective.
The paper's Algorithm 1 also conflates `done` with success in its retry condition;
the implementation follows the surrounding prose and official success predicate.
Budgets mean exactly H actions and Z additional retries, avoiding inclusive-loop
off-by-one ambiguity.

Upstream's Apache-2.0 license is retained in [LICENSE.upstream](LICENSE.upstream).
The source algorithms and prompt structure were rewritten for the generic Slick
interface described above; no benchmark datasets or environment code are bundled.

## Verification

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_expel tests.test_reflexion
../slick/.venv/bin/ruff check expel tests/test_expel.py
../slick/.venv/bin/ruff format --check expel tests/test_expel.py
```

The shared scripted provider exercises real Slick parsing and templates. Checks
cover retries, terminal failures, budgets, task-local reflections, extraction
order, rule counts and atomic rejection, task retrieval, evaluation isolation,
provider routing, raw failure records, transfer, and rendering from another
working directory. These are offline algorithm/interface checks, not paid model
evaluation or reproduced benchmark improvements.
