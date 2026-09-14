# Meta-Prompting Protocol

Problem-agnostic Slick implementation of the supplied paper,
[The Meta-Prompting Protocol: Orchestrating LLMs via Adversarial Feedback Loops](https://arxiv.org/abs/2512.15053)
by Fanzhe Fu, especially Algorithm 1. It optimizes a **reusable instruction set**,
using a Generator, a blind Auditor, and an Optimizer that groups textual critiques.

## Use

Use the repository's `optimizer/.venv` environment, or install the adjacent Slick
checkout with `python -m pip install -e ../slick` from the repository root.
There are no additional dependencies beyond Slick's existing Pydantic/Jinja stack.

Configure the template root once at application startup:

```python
from pathlib import Path

from slick import prompts
import meta_prompting
from meta_prompting import Case, MetaPrompting

prompts.TEMPLATE_ROOT = Path(meta_prompting.__file__).resolve().parent / "prompts"


async def optimize(task, rules, initial_prompt, train, gold, generator, auditor, optimizer):
    agent = MetaPrompting(task, generator, optimizer, rules=rules, auditor=auditor)
    result = await agent.run(
        initial_prompt,
        train,  # Nonempty list[Case], used for feedback and successful examples.
        gold,   # Separate nonempty list[Case], used only for regression validation.
        best_of_n=3,
        max_iterations=5,
        threshold=0.95,
    )
    return agent, result


async def apply(agent, result, input_text, context=""):
    return await agent.predict(result.instruction, Case(input_text, context=context))
```

`Case(input, context="", reference="")` accepts arbitrary text: questions, documents,
specifications, messages, or serialized domain data. A reference is visible only to
the auditor/evaluator. Task and rules remain outside the mutable instruction set.
`Instruction(text, examples)` is the complete optimized value: preserve both fields
when saving or reusing it, together with the original task and rules.

Pass configured, stateless Slick providers. The paper suggests generator temperature
around 0.7 and auditor temperature 0; configure these on the respective providers.
The optimizer has its own provider. The algorithm makes independent, sequential
calls with explicit history, without a Session. A provider must not introduce
hidden conversation history or collapse repeated generator calls through caching.

For executable checks, a rubric, or an existing evaluator, replace the model auditor
with an async callback returning `Audit`:

```python
from meta_prompting import Audit


async def evaluate(case: Case, artifact: str) -> Audit:
    # Small exact-match example; replace this body with your domain's evaluator.
    correct = artifact.strip() == case.reference.strip()
    return Audit(
        score=1.0 if correct else 0.0,
        critique="Matches the reference." if correct else "Does not match the reference.",
    )


def build_agent(task, rules, generator, optimizer):
    return MetaPrompting(task, generator, optimizer, rules=rules, evaluate=evaluate)
```

The callback takes precedence when both it and `auditor` are supplied. Scores must
be finite numbers in `[0, 1]`; critiques must be nonblank. Reserve 1 for fully
satisfying the requirements. The callback closes over task-specific rules and owns
any execution isolation, metrics, RAGAS/DeepEval integration, or external resources.
The agent never executes generated artifacts or installs evaluation infrastructure.

## Loop and explicit implementation choices

1. Measure the initial instructions on the full training and golden batches.
   Generate and audit N artifacts per case, keep the first maximum on ties, and
   score each batch by the mean of its per-case winners.
2. Group **all** training candidates scoring below 1, including losing candidates,
   by their semantic failure. Each report carries `(1 - score, critique)` and its
   input/context/artifact. A typed model response assigns every report exactly once;
   Python checks coverage and computes group frequencies from membership counts.
3. Revise the current instructions using those groups, raw failures, and the history
   of proposed instructions and acceptance decisions. The revision prompt requests
   constraint hardening and strategy refactoring when earlier changes fail.
4. Attach successful training winners (score exactly 1) as demonstrations. The
   optimizer cannot fabricate demonstration contents. New successes take priority,
   then older examples fill the remaining `max_examples` slots (default 4).
   Exact duplicates are removed. Matching input/context examples are excluded from
   generation for that case, preventing direct answer copying during training.
5. Optionally review the complete proposal, then measure it on both batches. Reject
   it if **any golden case** scores below its incumbent winner, or if the training
   mean decreases. Exact ties are accepted, allowing exploration without measured
   regression. Retain the previous instructions **and examples** on rejection.

Both batch means must reach `threshold` to stop successfully. `max_iterations`
counts proposed updates after the baseline, including rejected updates; zero runs
only the baseline. A final accepted proposal is evaluated before it can be returned.
If all training candidates score 1 but gold remains below threshold, stop with
`no_gradients`: golden failures do not become optimization feedback.

The paper does not specify its clustering algorithm, aggregation/acceptance policy,
example cap, or exact stopping score. The above are concrete implementation choices.
It also does not specify a separate repair call for examples: successes produced
under revised instructions become eligible on the following update. No untested
example is appended after the last evaluation.

## Ground truth, review, and observations

Supply separate human-verified examples with `run(..., anchors=[Example(...)])`.
These persist outside the synthetic example cap. One anchor plus four synthetic
examples gives the paper's illustrative 20% human-data mix at full capacity; callers
choose the actual mix. Anchors are optional and are not claimed to be verified by
the library. Keep them separate from `gold` and from any final test set.

Gold inputs, references, artifacts, scores, and critiques never enter optimizer
prompts or demonstrations. Gold does influence acceptance, so it is a validation
set, not an untouched final test set. The generator sees golden inputs only when
solving those cases; the auditor sees their references. Callers own split integrity.

For human meta-auditing, pass `review=async_review`, where
`async_review(current: Instruction, proposed: Instruction) -> bool` approves or
rejects each change before evaluation. It receives the complete example sets as
well as the instruction text. The caller may use an Inbox or another review UI.
Without the callback, updates run automatically. A rejected review consumes an
update attempt but no candidate generation or evaluation for that proposal.

`Result` includes `instruction`, accepted `train`/`gold` batches, all proposed
`history` trials, `stop_reason`, and call/evaluation counts. Each batch retains all
observations and its per-case `best` tuple. A trial records its instruction,
gradients, optional batches, acceptance, and reason. Inspect `agent.calls` for
rendered prompts, raw responses, and failures; `agent.evaluations` records split,
case, output, audit, and evaluator errors. These records remain in memory for caller
persistence and contain the supplied data.

Malformed output, invalid scores, tool requests, provider errors, review errors,
and evaluator errors propagate immediately without retries. Raw model responses
are logged before parsing/validation; partial trials stay available on failure.
Review exceptions leave the proposal pending and the incumbent unchanged. Runs
reset state; do not overlap operations on the same instance. `predict` makes one
unaudited call and appends to the call log; result counts describe the completed run.

For T training cases, G golden cases, N candidates, and K evaluated updates, there
are `(K + 1) * N * (T + G)` generation calls and the same number of audits/evaluator
invocations. Each proposed update adds two optimizer calls (group and revise).
A model auditor adds one provider call per audit; an evaluator callback adds none.
Provider-internal transport retries are outside these counts. No SDK retries are
added here. Full history and batches are retained; context and memory grow with the
run budget.

Slick's template root is process-global. Configure it once before running, and use
separate processes for simultaneous algorithms with different template roots.
Imports do not modify it. The four local templates contain no conditional logic.

## Official implementation references

No official implementation link is given in the supplied paper; a search of its
arXiv record and title did not identify a paper-specific repository. The cited
TextGrad and DSPy repositories are official implementations of those frameworks,
not of this protocol. Their source informed these concrete behaviors, re-expressed
using Slick rather than introducing a second model runtime:

| Source inspected | Behavior used here |
| --- | --- |
| TextGrad [`TextualGradientDescent._update_prompt` and `step`](https://github.com/zou-group/textgrad/blob/75e912e210864b61999781778cdf756d4468120f/textgrad/optimizer/optimizer.py) | Feed the current text, contextual critiques, constraints, and examples into a distinct update operation; retain raw update output |
| TextGrad [`run_validation_revert`](https://github.com/zou-group/textgrad/blob/75e912e210864b61999781778cdf756d4468120f/evaluation/prompt_optimization.py) | Evaluate a proposed instruction on separate validation data and revert regressions; this implementation strengthens mean rollback to per-case golden rollback |
| DSPy [`BootstrapFewShot`](https://github.com/stanfordnlp/dspy/blob/ecba33763316d2a4c6c756046a1118ecbff033e7/dspy/teleprompt/bootstrap.py) | Promote metric-approved outputs into capped demonstrations, combine them with labeled examples, and remove a case's own demonstration when generating for it |

Source references checked on 2026-09-14. Both upstreams are MIT licensed:
[TextGrad license](https://github.com/zou-group/textgrad/blob/75e912e210864b61999781778cdf756d4468120f/LICENSE),
[DSPy license](https://github.com/stanfordnlp/dspy/blob/ecba33763316d2a4c6c756046a1118ecbff033e7/LICENSE).
This is an independent algorithm implementation; it does not vendor their code,
recreate TextGrad's autograd graph, or run DSPy's MIPRO/Bayesian search. The paper
presents those as enabling frameworks rather than required steps of Algorithm 1.
Generic templates are new, not benchmark prompt reproductions.

## Verification and limits

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_meta_prompting
../slick/.venv/bin/ruff check meta_prompting tests/test_meta_prompting.py
../slick/.venv/bin/ruff format --check meta_prompting tests/test_meta_prompting.py
```

Tests use the shared `tests/providers.py` scripted provider through real Slick
decorators. They cover best-of-N, blind audits, golden separation, rollback,
demonstrations, grouping coverage, thresholds/budgets, review, failures, state
reset, inference, and template rendering from another working directory.
These are offline algorithm/interface checks, not a reproduction or a measured
quality improvement. Temperature zero does not guarantee deterministic or correct
audits; score quality depends on the evaluator. Finite regression tests do not
establish generalization, prevent model collapse, or prove convergence.
Reported scores describe audited best-of-N selection; a single unaudited `predict`
call is a different inference budget and need not achieve the same quality.
