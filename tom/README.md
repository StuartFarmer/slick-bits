# Theory of Mind meta prompting

Task-agnostic implementation of the judge → meta-prompt editor → regeneration
loop in **Automated Meta Prompt Engineering for Alignment with the Theory of
Mind**, Baughman et al. (2025): [paper](https://arxiv.org/abs/2505.09024),
[PDF](https://arxiv.org/pdf/2505.09024).

The supplied paper and its arXiv record do not identify an official code
repository. Searches by exact title, arXiv ID, and authors on 2026-09-14 did not
locate one. This is an independent implementation, with explicit interpretations
of underspecified mathematics; no third-party implementation is represented as
official or copied into this folder.

## Use

Use the existing repository environment, or install the adjacent Slick checkout
with `pip install -r requirements.txt` from this directory. No additional numerical
dependencies are needed. Python 3.10+ is required; checked against the local
Slick 0.3.0 checkout, Pydantic 2, and Python 3.14.

Configure the process-global template root once at application startup. Run from
the repository root, or put that root on `PYTHONPATH` when launching elsewhere.

```python
from pathlib import Path

from slick import prompts
import tom
from tom import Dimension, Profile, TheoryOfMind

prompts.TEMPLATE_ROOT = Path(tom.__file__).resolve().parent / "prompts"

dimensions = (
    Dimension("accuracy", "Agreement with the supplied source facts; 100 means all agree."),
    Dimension("detail", "Amount of useful detail; 0 is terse, 100 is exhaustive."),
    Dimension("repetition", "Repeated information; 0 means each point appears only once."),
)
profile = Profile({"accuracy": 100, "detail": 60, "repetition": 0})

async def align(task, context, provider):
    agent = TheoryOfMind(
        task, provider, dimensions, profile,
        context=context,
        max_iterations=21,
        timeout=120,
        threshold=0.05,
    )
    result = await agent.run()
    # No evaluated draft is available if the deadline expired during the first one.
    if result.best is None:
        return None
    return result.best.content
```

Point `task` and `context` at any text-producing problem: prose, structured text,
code, plans, etc. Give each dimension a stable, measurable 0–100 rubric. High
scores mean more of a trait, so targets can be high, low, or intermediate. The
nonempty dimension names and profile target keys must match exactly. Caller
configuration is trusted; generated scores are checked for exact keys, numeric
0–100 values, and finiteness. Generated artifact whitespace is preserved.

`judge_provider=` and `editor_provider=` may select different models; both default
to `provider`. The generator and all prompts retain the original task and source
context throughout revision. `run(instruction="...")` sets the initial generation
instruction separately from that task. Each operation has its own local Jinja
template. The judge's structured boundary explicitly requests JSON; no Session
history is used. Do not change Slick's global template root during a run.

For measured traits, pass an async `evaluate(content) -> Judgement` callback:

```python
from tom import Judgement

async def evaluate(content):
    scores, explanation = await measure_in_your_environment(content)
    return Judgement(scores=scores, feedback=explanation)

# Add evaluate=evaluate to TheoryOfMind(...).
```

This replaces the judge for both generated candidates and human edits. The
callback closes over task data and owns any execution isolation. The algorithm
never executes generated code. Feedback is a brief assessment of observed traits.

## Learning an editor profile

Call `await agent.learn(human_edited_text)` on the instance with that text's
original task and context. It judges the edit and mutates the supplied profile.
Only valid judgements are added. Each target becomes the arithmetic mean of that
editor's observed scores; cold-start targets do not count as observations.
`profile.observe(scores)` also accepts already-measured human scores directly.
This direct method trusts the caller's measurements.

Keep separate profiles for separate editors, reusing each across task instances
with the same dimension definitions. Preserve both `targets` and `samples` when
saving/restoring a profile. Persistence belongs to the application. Generated
drafts never update the profile; learning is explicit and outside `run()`'s budget.
The run snapshots target means and covariance weights for comparable losses.

## Algorithm and numerical interpretation

1. Generate a draft, then measure its dimensions with the judge or callback.
2. Compare those scores with the editor's target using volume and vertex distance.
3. Stop if `loss < threshold`; the default is the paper's `0.05`.
4. Otherwise give the editor the draft, previous instruction, judgement, geometry,
   and signed trait differences. It writes a new generation instruction.
5. Regenerate using that instruction and the latest draft; judge again until
   convergence, the iteration limit, or the deadline.

The following choices make the paper's equations executable and reproducible:

| Paper component | Implementation |
| --- | --- |
| Editor expectations (§5.4) | Mean scores of judged human edits, retaining raw samples |
| Covariance (Eq. 6–7) | Sample covariance on scores divided by 100; weight of each axis is the maximum off-diagonal `1 - abs(cov)` in its column |
| Graph transform `f(G)` (Eq. 11) | A diagonal matrix whose entries are normalized scores multiplied by those axis weights |
| Area/volume (Eq. 11) | Product of the diagonal entries, equal to its determinant |
| `tmd` (Eq. 12) | Mean absolute difference of corresponding scaled axis entries; this equals mean Euclidean distance between diagonal row vertices |
| Loss (Eq. 17) | `(r**2 + abs(r)) / 4 + tmd`, with `r = abs(expected_area - area) / expected_area` |
| Trait feedback (§5.3) | Score minus target, in percentage points; zero targets require no division |

With fewer than two observations, or only one dimension, weights are one.
Weights derived from human samples are applied to both expected and generated
vertices. Sample covariance, score normalization, excluding self-edges, and this
specific diagonal transform are **implementation choices**: the paper does not
fully define the graph-to-matrix transform or how to construct a generated
covariance matrix during a single-draft iteration. No correlation substitution
or model gradients are used.

**Zero target volume:** the paper's ideal repetition score of zero makes the
diagonal determinant zero and Eq. 17 undefined. Here, exactly zero expected area
uses a denominator of one (the normalized unit hypercube), leaving `tmd` active.
This finite fallback is an extension, not an author-specified formula. With a
shared zero axis, the volume term cannot distinguish the remaining axes; the
distance term still does. In many dimensions, products can also underflow.
Convergence at a nonzero threshold does not require every individual trait to
match exactly and is not a guarantee of factual correctness.

The literal Eq. 17 has a factor of **one quarter** on each area-error term,
despite the prose describing an equal average. This implementation follows the
displayed equation. Rectangular Hausdorff/subspace approximations (Eq. 13–14)
are not needed by the chosen square representation and are not implemented.

The paper alternates between an editor directly rewriting content (§5.3) and
editing prompts before regeneration (§5.4). This implementation uses the latter.
The prompt text is a deliberate domain-agnostic adaptation, not a verbatim
production prompt. Domain-specific fact extraction, tennis few-shot examples,
Kafka, databases, and publication workflows stay outside the algorithm.
Temperature/top-p/top-k remain provider settings: the paper does not supply an
executable gradient estimator or parameter update rule for Eq. 19–21. The
implemented optimization is in-context prompt revision, without weight training.

## Budgets, results, and failures

`max_iterations` counts fully attempted generate/judge cycles, including the
initial draft. A successful run with `n` evaluated drafts makes `n` generation,
`n` judgement, and `n-1` editor calls: `3*n-1` provider calls (at most 62 by default).
An injected evaluator replaces the `n` judge calls. No final unused editor call
is made. `timeout` is a total wall-clock budget for the loop, including pending
generation, evaluation, and editing. Cancellation is cooperative; a blocking
provider or one that suppresses cancellation can exceed the deadline. A provider
or evaluator raising `asyncio.TimeoutError` also ends the run as `timeout`.

`Result.history` contains all completed evaluated drafts, their generation
instructions, judgements, and loss components. `history[-1]` is the paper-style
final draft. As a convenience, `Result.best` also retains the lowest-loss draft
(earlier wins ties); it does not change which draft feeds the next iteration.
`stop_reason` is `converged`, `iterations`, or `timeout`. `best` is `None` if no
draft was evaluated. Callers choose whether to use unconverged content.

Malformed output, invalid measurements, tool requests, and non-timeout provider
errors propagate immediately without retry. External cancellation propagates.
`agent.calls` retains operation names, rendered prompts/raw responses where
available, and errors, including failures before structured parsing succeeds.
Injected evaluator calls are labeled `evaluate`. `agent.history` retains complete
iterations on failure. `run()` resets both lists; `learn()` appends call records.
Only one operation may run at a time per instance; callers own transport retries.

## Checks

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_tom
../slick/.venv/bin/ruff check tom tests/test_tom.py
../slick/.venv/bin/ruff format --check tom tests/test_tom.py
```

The shared scripted provider exercises two unrelated task types, role routing,
profile learning, covariance, hand-calculated loss, zero targets, strict stopping,
budget exhaustion, deadline cancellation, invalid output, failure logging, and
template rendering from another directory. These are deterministic algorithm and
interface checks, not a reproduction of the paper's production results or an
empirical evaluation of model quality.
