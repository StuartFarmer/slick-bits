# EvoPrompting

Problem-agnostic implementation of Algorithms 1–3 in
[EvoPrompting: Language Models for Code-Level Neural Architecture Search](https://arxiv.org/abs/2302.14838)
(Chen, Dohan and So, NeurIPS 2023). Candidate content is arbitrary text; your task,
evaluator, metrics and model backend define the problem.

## Usage

Set the process-global Slick template root once at application startup. The
installed local Slick 0.3.0 provider contract is `acall(context) -> (text, tools)`;
it has no per-call temperature argument. Supply a factory that configures the
actual sampling temperature on your backend, instead of putting it in a prompt.

```python
from pathlib import Path

from slick import prompts
import evoprompting
from evoprompting import Evaluation, EvoPrompting

prompts.TEMPLATE_ROOT = Path(evoprompting.__file__).resolve().parent / "prompts"


async def search(task, seeds, provider_at_temperature, evaluate, tune):
    agent = EvoPrompting(
        task,
        provider_at_temperature,
        evaluate,
        tune=tune,
        rounds=10,
        prompts_per_round=10,
        samples_per_prompt=16,
    )
    return await agent.run(seeds)
```

| Dependency | Contract |
| --- | --- |
| `task` | Instructions, candidate interface and constraints as text |
| `seeds` | Known candidate strings, evaluated before round zero |
| `provider(temperature)` | Synchronous factory returning a Slick-compatible provider at that temperature and the current model checkpoint |
| `evaluate(content)` | Async callback returning `Evaluation(error, cost, metrics={}, fitness=None)` |
| `tune(current_factory, children, settings)` | Async callback returning a factory bound to the updated soft prompt |
| `targets(parents)` | Optional synchronous function returning the target metric dictionary |

`error` is a nonnegative measured loss; children must satisfy `error < alpha`.
`cost` is positive (parameters, runtime, tokens, or another resource). By default,
fitness is `-error * cost`, and higher fitness is better. Supply a finite
`Evaluation.fitness` for another objective. Measured values and metric dictionary
values must be finite. The evaluator receives the exact generated text, including
leading indentation and trailing newlines. It owns datasets, execution isolation,
training schedules, and validation/test separation. Raise `CandidateRejected`
for expected artifact failures; unexpected errors and cancellation propagate.

For example, a query evaluator could return
`Evaluation(error=0.02, cost=12, metrics={"latency_ms": 12, "error": 0.02})`.
Pair it with `targets=lambda parents: {"latency_ms": 0.9 * min(p.evaluation.cost
for p in parents), "error": 0.9 * min(p.evaluation.error for p in parents)}`.
Setting cost to 1 makes the default objective minimize error alone. The default
NAS metrics and targets should be replaced together for unrelated problems.

## Search and tuning

1. Evaluate unique seeds with the same evaluator, retaining them only as the
   initial parents. Seeds are not subject to the child error threshold.
2. Build `m` prompts by uniformly drawing `k` examples **with replacement** from
   the current parents. Generate `n` independent completions per prompt. Each
   completion draws a temperature uniformly from `(0.2, 0.6, 0.8, 1.0)`.
3. Evaluate unique nonblank children, reject invalid measurements, and filter
   `error >= alpha`. Accepted children enter the historical archive and the pool
   eligible for future parent selection. Exact-string duplicates, including
   repeated rejected strings and seed copies, are never evaluated again.
4. Except after the last round, select the highest-fitness `p` eligible children
   across all previous rounds. Remove them permanently from the eligible pool;
   they serve as parents for just the next generation. Stable ties favor earlier
   children. Fewer than `p` available children are all selected.
5. Tune on the **current round's accepted children minus the selected parents**.
   Previously unselected children may become parents in a later round, even if
   they were used for tuning earlier. Feed the returned factory into the next round.

Defaults are the paper's `T=10, m=10, n=16, k=2, p=10, alpha=0.5`. Without early
termination, generation makes exactly 1,600 provider calls. Rejected and duplicate
samples count toward this budget; there are no refill attempts or algorithm-owned
retries. Seeds add evaluator calls but consume no generation budget. Evaluation
begins after each complete generation batch. Backend transport retries remain
the caller's responsibility.

The default `paper_targets` requests 90% of the smallest sampled parent's cost,
rounded to the nearest 100, and 102% of the highest sampled parent's accuracy,
rounded to 0.001. Accuracy means `1 - error`. As in the paper, these aspirational
targets are **not clamped**: small costs can round to zero and target accuracy can
exceed one. Python's nearest-even rule resolves exact rounding ties.

**Soft prompt optimization is a required external backend for the full method.**
The callback receives measured `Individual` records, including `.content`,
`.metrics`, `.evaluation` and `.fitness`, plus `TuningSettings` with five epochs,
16 learned prompt tokens, batch size 16 and learning rate 0.1. It must freeze the
base LM, train its continuous prompt embeddings on these records, continue from
the current prompt between rounds, and return a factory that uses those updated
embeddings at every sampling temperature. Metric-conditioning serialization,
tokenization, loss masking, optimizer choice, checkpointing and compute belong to
that backend; the paper does not fully specify these training details.

Passing `tune=None` explicitly runs **EvoPrompting without prompt-tuning**. The
optimizer does not simulate tuning with rewritten natural-language instructions.
Empty tuning sets skip training; an empty eligible parent pool stops early rather
than reusing retired parents. No tuning occurs after the final round.

## Results and failure records

`result.top` is Algorithm 1's final top set from the **remaining eligible pool**.
`result.archive` retains every accepted child for inspection, including retired
parents; `result.best` is the best historical child, or `None`. Keeping these
separate resolves the paper's pseudocode/prose ambiguity about historical `G`
and permanent removal. An earlier winner can therefore be absent from `top`.

`history` records each round's parent pool, accepted children, next parents and
training subset. `attempts` records each sample's actual few-shot parents, targets,
temperature, raw text, status and rejection reason. `evaluations` counts actual
evaluator invocations, including seeds and rejected measurements. `provider` is
the final factory. `stop_reason` is `rounds` or `no_parents`.

Errors in seed evaluation, providers, evaluator infrastructure and tuning abort
the run; partial state remains on the agent. Generation failures have a failed
attempt record. Candidate rejection records retain the raw output. Calls use
direct providers with no shared conversation history or automatic JSON parsing.
Each `run` resets population, counters and RNG but retains the current provider;
use a fresh factory to restart from the original model. Do not run one instance
concurrently. Configure the absolute template root once before use; import does
not mutate it. Launch from the repository root, or put the repository on
`PYTHONPATH` when launching elsewhere.

## Sources and adaptations

Checked on 2026-09-14: the [paper's arXiv record](https://arxiv.org/abs/2302.14838),
[NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html),
and the [first author's publication page](https://angie-chen55.github.io/).
No public official EvoPrompting search implementation could be verified from
these sources. The author's EvoPrompting entry links the paper and OpenReview,
not code; OpenReview itself presented a browser verification challenge.
`microsoft/EvoPrompt` implements a different paper. The public
`algopapi/EvoPrompting_Reinforcement_learning` describes itself as an independent
replication, so neither was used as an official implementation.

The supplied paper is the algorithm source. Its Flax/Haiku links are neural
network libraries, not an official search implementation. No third-party code
was copied. MNIST-1D and CLRS training infrastructure and the discovered GNNs are
outside this task-agnostic optimizer.

The local template preserves the paper's triple-quoted metrics/example sequence
and final target metrics, but adds generic task instructions, uses JSON numeric
metrics, and requests a whole artifact instead of continuing `class Model`.
Crossover and mutation remain one decorated operation because the paper combines
them. Empty-pool behavior, exact-text deduplication, custom objectives, and the
separate historical archive are explicit implementation choices.

```sh
rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_evoprompting -v
```

These are deterministic tests of search decisions and Slick integration. They
do not train an LM, execute generated code, or reproduce the paper's NAS results.
