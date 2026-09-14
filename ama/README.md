# Ask Me Anything (AMA)

Problem-agnostic implementation of Arora et al., *Ask Me Anything: A Simple
Strategy for Prompting Language Models*. Each chain reformulates an input,
answers using its context, and maps the answer back to the original task.
Multiple chains supply noisy votes. Fixed-label tasks use dependency-aware
weak supervision; open-ended tasks use majority vote, as in the paper.

```python
from pathlib import Path

from slick import prompts
import ama

# Configure once at application startup; supply your configured Slick provider.
prompts.TEMPLATE_ROOT = Path(ama.__file__).resolve().parent / "prompts"

async def answer_batch(task, inputs, provider):
    agent = ama.AMA(task, provider)
    result = await agent.run(inputs)
    return result.predictions

# inputs = [ama.Example(text="your question or document", context="optional evidence")]
```

For classification, supply a shared label set and a batch of unlabeled examples:

```python
async def classify(task, labels, inputs, provider, extra_unlabeled=()):
    agent = ama.AMA(task, provider, labels=labels)
    result = await agent.run(inputs, unlabeled=extra_unlabeled)
    return result
```

`task` describes how to interpret the input and labels. No benchmark names,
datasets, models, credentials, retrieval systems, or evaluators are embedded.
An `Example` holds arbitrary text plus optional evidence; serialize structured
inputs to text at the application boundary. This is a prediction algorithm,
so there is no fitness evaluator or generated-code execution.

Install from this directory (the relative Slick path follows this repository's
existing requirements convention):

```sh
cd ama
python -m pip install -r requirements.txt       # prompt chains and majority vote
python -m pip install -r requirements-ws.txt    # official numerical backend
```

The full numerical stack was exercised with Python 3.12, NumPy 1.26.4,
CVXPY 1.7.5, SCS 3.3.1, Torch 2.14.0, NetworkX 2.8.8, and the pinned MeTaL
revision. Use Python 3.12 for these legacy numerical dependencies. Basic
prompting also passes with the repository's Python 3.14 / Slick 0.3.0 environment.
NetworkX is pinned because MeTaL expects clique members to be sets. Its missing
`tensorboardX` dependency is supplied explicitly. Upstream emits deprecation
warnings with modern Torch; its source is not patched or monkey-patched here.

## Customize chains and mapping

The default collection contains yes/no, wh, and cloze question styles. Each
`Chain` accepts additional `question_examples` and `answer_examples` transcripts.
Combine styles and demonstrations to obtain different, reusable chains. All
operations own separate external Jinja templates in `ama/prompts/`.
`Chain("identity")` skips reformatting when an input is already the desired
question; vary its answer demonstrations across chains.

Question and answer generation remain plain text. A separate mapping call
produces a typed JSON label or free answer, or `null` for abstention. Mapping sees
the original input, context, generated question, and intermediate answer so a
"no" to a negated reformulation is not blindly treated as a negative task label.
Open-answer mapping makes the different question styles comparable before voting.
Both mapping prompts explicitly forbid independently solving the task again.
These are instructions, not a guarantee of model correctness.

For deterministic normalization, domain-specific parsing, or naturally aligned
answers, supply an async mapper. It replaces the mapping LLM call:

```python
async def map_answer(example, trace):
    # Use only when the QA answer already has the original task's meaning.
    value = trace.answer.strip().casefold()
    return None if value == "unknown" else value

agent = ama.AMA(task, provider, map_answer=map_answer)
```

With fixed labels the callback must return an exact label or `None`. Labels such
as "unknown" or "neither" are real classes when included in your label set;
`None` alone denotes abstention. Invalid generated labels and malformed JSON
raise. An open-ended vote can be any nonblank answer text. Majority voting uses
exact string equality after mapping, discards abstentions, and breaks ties by
first chain order; all-abstaining rows return `None`. It does not perform semantic
clustering. Custom mappers own their normalization and any internal costs.

## Weak supervision

`aggregation="auto"` selects WS when labels are supplied and majority otherwise.
Use `aggregation="majority"` to explicitly request the baseline, including for a
single example. WS needs at least two examples, three chains, and two labels;
these are minimum shape requirements, not a statistical sufficiency guarantee.
Use a representative unlabeled batch with varied predictions. Constant or
entirely abstaining columns raise with guidance to collect more varied votes.

WS fits on `extra_unlabeled + target_votes`, following the paper's transductive
setup. No gold labels enter fitting, dependency selection, or prediction. Classes
come from the caller. The prior is uniform unless supplied as
`WSConfig(class_balance=(...))` in label order. Returned probabilities follow
that order; prediction is their argmax, with ties going to the first label.
All-abstaining prediction rows receive the fitted prior.

The covariance and sparse/low-rank objective, binary/multiclass structure
encoding, regularizers, SCS solver, and strongest-edge/dense-noise policy follow
the official runner. As upstream does, structure learning merges abstentions
with class zero; MeTaL itself receives distinct abstention code 0 and label codes
1 through K. Automatic recovery retains at most one dependency. An all-zero
structure produces no edge rather than an arbitrary self-edge.

`WSConfig` exposes the original optimization budgets: 10,000 initial epochs at
1e-5, then 80,000 dependency epochs at 1e-6 if an edge is selected. **MeTaL also
hard-codes a further 20,000 mu-estimation epochs** in the dependency path.
`dependencies=None` learns the graph; `dependencies=()` runs the no-dependency
ablation. Explicit dependencies can be supplied to the official backend, whose
graph restrictions still apply; the paper's automatic path uses at most one.
Numerical failures propagate, without switching to majority vote.
Starting a refit invalidates the previous model; a failed fit cannot be used
for prediction.

To fit once and predict later without LLM calls:

```python
model = ama.WeakSupervision(labels)
model.fit(unlabeled_vote_rows)  # strings or None; columns have stable chain order
probabilities = model.predict_proba(new_vote_rows)
```

`agent.collect(inputs)` obtains vote rows without fitting and appends records.
Keep task, labels, chains, model, and mapping fixed between fit and prediction.

## Results and execution policy

`Result` contains target `predictions`, target `votes`, `traces`, attempted LLM
`calls`, and WS `probabilities`/`dependencies` when applicable. Traces include
both target and additional unlabeled QA pairs, in execution order. The fitted
backend remains available as `agent.label_model`, including its recovered
`structure` matrix and official `model.get_conditional_probs()`.

Three default chains cost nine provider calls per example: question, answer,
mapping. A custom mapper removes three; identity chains omit question calls.
No hidden retries occur. Providers own transport retries and raw-response
logging, including rejected mapping JSON. Errors propagate; partial
`agent.traces`, raw question/answer `agent.generations`, and `agent.calls` survive.
Each `run()` resets records. Additional unlabeled examples are ignored in the
majority-vote path. Empty targets return an empty result without generation.

Calls are sequential and use explicit providers, without shared conversation
history. Configure Slick's process-global template root once; imports never
change it. Launch from the repository root, or put that root on `PYTHONPATH`
when launching elsewhere. MeTaL fitting runs synchronously on CPU and sets
global Python, NumPy, and Torch RNG state. Use separate processes for concurrent
fits or applications requiring independent template roots.

## Provenance and differences

- [Official AMA repository](https://github.com/HazyResearch/ama_prompting/tree/460843d93a9e4bf2115eb35e4a02e82b6b67feac):
  `aggregation.py` adapts the numerical structure-learning code in
  [`boosting/run_ws.py`](https://github.com/HazyResearch/ama_prompting/blob/460843d93a9e4bf2115eb35e4a02e82b6b67feac/boosting/run_ws.py).
- Its official submodule,
  [metal-ama at 18769e7](https://github.com/mayeechen/metal-ama/tree/18769e7653d4f6dc16a9858cd6fe4ed5ad9f5779),
  is installed and called directly for parameter estimation and posterior
  inference. This is not an independent-source substitute for the paper's WS.
- Short question/answer demonstrations are adapted from
  [`tasks/RTE_final.py`](https://github.com/HazyResearch/ama_prompting/blob/460843d93a9e4bf2115eb35e4a02e82b6b67feac/tasks/RTE_final.py).
  The accompanying [Apache-2.0 license](LICENSE) covers reused source material.

This deliberately generalizes the benchmark-specific prompts and mapping code.
It adds task instructions, generated-output checks, explicit semantic mapping
for open answers, and corrected negation guidance. The original runner passes
`Y_dev` to estimate class balance; this implementation uses no gold labels.
It preserves the upstream estimator, including its covariance convention and
loss behavior, rather than claiming to repair or rederive MeTaL. The initial
no-dependency fit is followed by a dependency fit when needed, as upstream.
Benchmarks, historical models, published accuracies, and latency results are
not reproduced.

Offline verification, from the repository root:

```sh
python -B -m unittest tests.test_ama
../slick/.venv/bin/ruff check ama tests/test_ama.py
```

The numerical integration checks run when WS dependencies are installed;
otherwise unittest reports them as skipped. They test synthetic binary and
multiclass unlabeled votes, not paper benchmark performance.
