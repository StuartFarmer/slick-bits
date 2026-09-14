# ERM

Task-agnostic paper reconstruction of **Efficient and Accurate Prompt Optimization:
the Benefit of Memory in Exemplar-Guided Reflection** (Yan et al., 2025),
[paper](https://aclanthology.org/2025.acl-long.37/), §§3.2–3.4 and equations 2–9.
No author-maintained implementation was located. Public third-party replication
repositories were not treated as official sources.

ERM generates detailed correct exemplars from failed training cases, verifies them,
and uses their solutions to generate distinct feedback. Each feedback produces a
candidate instruction. Only feedback whose candidate improves validation enters
semantic-deduplicated memory. Periodically, a priority-weighted sample of historical
feedback produces an additional group-guided revision. Its realized gain updates
the selected priorities by exponential moving average of a binary reward, and low
priority entries are removed.

The exemplar factory stores verified solutions, randomly replaces duplicate
questions, and retrieves by priority multiplied by semantic similarity to each
query. Retrieval samples without replacement during optimization and selects the
highest products deterministically at inference. A measured comparison of the
selected instruction with and without its exemplars rewards or forgets the used
exemplars. Beam search and the best observed instruction-plus-exemplar snapshot
remain inside the agent.

```python
from pathlib import Path
from slick import prompts
from erm import ERM

prompts.TEMPLATE_ROOT = Path("/absolute/path/to/erm/prompts")
agent = ERM(task, provider, evaluate, failures, verify, similarity, validation_queries)
result = await agent.run(initial_prompt, beam_size=2)
instruction = result["best"].prompt
measured_exemplars_by_query = result["best"].exemplars
inference_examples = agent.retrieve(new_question, count=5)
```

Required callbacks and data:

- `failures(instruction)` asynchronously returns training dictionaries with string
  `question` and ground-truth `answer`, plus observed response context.
- `verify(exemplar, original_failure)` asynchronously checks the generated solution
  process. Before calling it, the agent checks that the question was in the sampled
  failures and the answer exactly matches ground truth. Matching answer text alone
  does not establish a valid solution.
- `similarity(left, right)` returns a comparable semantic similarity. The paper uses
  BGE-M3; the model and its dependencies belong to the caller.
- `evaluate(instruction, exemplars_by_query)` asynchronously returns a finite
  higher-is-better validation score. `queries` are the fixed validation questions
  or stable serialized query keys. Each maps to a list of `Exemplar` objects with
  `question`, `answer`, and `rationale`; the evaluator renders them for that query.
  An empty mapping means no exemplars.

Explicit adaptations and choices: meta-prompts are task-neutral rewrites with
typed JSON feedback/exemplar generation. Memory priorities start at 1; duplicate
exemplars are identified by exact question; replacement probability, reward EMA,
forgetting threshold, similarity cutoff, retrieval temperature and counts are
configurable. The paper's `softmax({exp(score/temperature)})` notation is interpreted
as the usual normalized exponential weights, avoiding double exponentiation. Both
retrieval samplers select without replacement. A fixed retrieval sample is shared
across all sibling evaluations in a round so their score differences reflect
instruction edits. Aggregate validation gain supplies the binary exemplar reward;
this makes the paper's benefit test concrete for arbitrary tasks. No score cache
is used because the exemplar context changes between rounds.

`best.exemplars` retains the context actually measured even if factory forgetting
later changes inference retrieval. It is not a promise that the current factory
has the same score. The result also exposes final memories, factory verification
and storage decisions, feedback gains/storage decisions, and call/evaluation
counts. Blank generated fields and nonfinite measured scores raise; provider,
verification, similarity, and evaluator exceptions propagate without retries.
Dependencies are Slick, Pydantic, and the standard library, plus caller-selected
semantic verification/embedding implementations. Tests check wrong-answer and
wrong-reasoning rejection, probabilistic replacement, feedback gain/forgetting,
query-dependent retrieval and exemplar pruning. They do not reproduce paper scores.
