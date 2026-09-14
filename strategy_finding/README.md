# Strategy finding

Problem-agnostic implementation of **Automate Strategy Finding with LLM in Quant
Investment** (Kou et al.), following the supplied paper's Appendix A.6. It discovers
numeric signals for a caller-defined prediction problem. Finance-specific data,
operators, backtesting, and trading are outside the agent.

The three stages are:

1. Filter each document, categorize its relevant evidence, then generate candidate
   signals per category. Merge with an optional existing factory, deduplicating
   exact `(category, content)` pairs. An irrelevant document can yield no candidates.
2. Evaluate every candidate through your async callback. Independent Slick calls
   assess confidence and risk suitability using that evidence and current context.
   Compute `0.6 * confidence + 0.4 * risk`; retain the highest scoring candidate
   **strictly above** the threshold in each category. Ties keep the first candidate.
3. Prepare aligned candidate values and outcome targets through your callback.
   Fit an input → 10 ReLU units → scalar linear-output network with minibatch SGD,
   MSE and L2 regularization. Return the checkpoint with lowest validation MSE,
   including the untrained checkpoint when no update improves it.

## Use

Install the repository's adjacent Slick checkout and the numerical dependency:

```sh
python -m pip install -e ../slick 'numpy>=1.24,<3'
```

Configure Slick's process-global template root once at application startup. This
module does not change it on import. The absolute path also works from other
launch directories when this repository is on `PYTHONPATH`.

```python
from pathlib import Path

from slick import prompts
import strategy_finding
from strategy_finding import StrategyFinder

prompts.TEMPLATE_ROOT = Path(strategy_finding.__file__).resolve().parent / "prompts"

async def discover(task, provider, documents, evaluate, prepare, context):
    agent = StrategyFinder(
        task=task,
        provider=provider,
        evaluate=evaluate,
        prepare=prepare,
        candidate_interface="Describe the exact signal format, available inputs and operators here.",
        risk_preference="Prefer stable performance across operating conditions.",
    )
    return await agent.run(documents, context=context)
```

Supply two callbacks:

- `async evaluate(candidate: Candidate, context: str) -> Evaluation`: interpret
  `candidate.content` using your own evaluator and return `Evaluation(score,
  evidence)`. The score is finite and higher-is-better; evidence is text containing
  measured quality, risk statistics, uncertainty and relevant historical conditions.
  An IC, rank IC or Sharpe ratio can be included for finance; other tasks use their
  own metrics. The LLM scores are judgments, not calibrated probabilities or direct
  copies of the measured score. Risk score means suitability, so larger is better.
- `async prepare(selected: tuple[Candidate, ...]) -> TrainingData`: compute numeric
  matrices with columns in exactly the supplied candidate order. Return
  `TrainingData(x_train, y_train, x_validation, y_validation)`, where feature arrays
  have shape `(observations, len(selected))` and targets have shape `(observations,)`.
  Supply nonempty, finite arrays. The agent never executes candidate content.

The caller owns any execution isolation, available operators, missing-data handling,
outcome alignment and data splits. For sequential problems, prepare chronological
training/validation splits and purge overlapping target horizons. **Neither judge
nor candidate selection should receive validation or test outcomes.** The prepare
callback is invoked after selection; test arrays are never supplied to training.
The algorithm cannot detect leakage introduced within a caller callback.

`result.selected` fixes column order for inference:

```python
if result.model is not None:
    predictions = result.model.predict(new_signal_values)
    weights, intercepts = result.model.local_weights(new_signal_values)
    # predictions == (weights * new_signal_values).sum(axis=1) + intercepts
```

This is a numeric signal ensemble. Arbitrary candidate text is supported at the
generation boundary, but your task must map each candidate to one numeric feature
per observation and provide scalar outcome targets for the combiner.

For new documents or conditions, run again with `factory=result.factory`. Every
candidate is re-evaluated and a fresh model is fitted; rejected or unselected
candidates remain in the factory and may become useful in another context.
Empty selection returns `selected=()` and `model=None` without preparing data.
One `StrategyFinder` instance runs sequentially; do not share it across concurrent runs.

Defaults are `threshold=0.5`, `confidence_weight=0.6`, `risk_weight=0.4`,
`epochs=1000`, `learning_rate=0.001`, `batch_size=32`, `regularization=0.001`,
`seed=0`. Scores are combined directly without normalizing the supplied weights.
L2 contributes `regularization * weight` to each weight gradient; biases are
unregularized. Scaling uses training minima/ranges only, maps constant column ranges
to one, and does not clip inference values. Validation selects a checkpoint after
each full epoch; it never contributes gradients. No automatic retraining schedule
or separate context-gating network is introduced.

## Official implementation and fidelity

Inspected the authors' [official repository](https://github.com/kouzhizhuo/Automate-Strategy-Finding-with-LLM-in-Quant-investment)
at commit **ebcb5ed44a71664316c99b021026358a44aef38d**. The release was used as an
implementation reference, not executed or vendored. This is a fresh task-neutral
implementation, not an exact port of its finance scripts.

| Official source / paper | Use in this implementation |
| --- | --- |
| [`main.py`](https://github.com/kouzhizhuo/Automate-Strategy-Finding-with-LLM-in-Quant-investment/blob/ebcb5ed44a71664316c99b021026358a44aef38d/main.py) | Preserve evaluation before selection; move formula execution, neutralization, rank IC, returns and downside analysis into `evaluate`. Avoid the source's direct `eval(formula)`. |
| [`modules/alpha_formula_reader.py`](https://github.com/kouzhizhuo/Automate-Strategy-Finding-with-LLM-in-Quant-investment/blob/ebcb5ed44a71664316c99b021026358a44aef38d/modules/alpha_formula_reader.py) and spreadsheets | Replace fixed workbook inputs with a caller-supplied or generated categorized factory. |
| [`AutoGPT/main.py`](https://github.com/kouzhizhuo/Automate-Strategy-Finding-with-LLM-in-Quant-investment/blob/ebcb5ed44a71664316c99b021026358a44aef38d/AutoGPT/main.py) and local prompts | Retain the separation of historical evidence and current-condition reasoning. Its implementation is a pairwise incumbent/challenger contest using uploaded spreadsheets, not the paper's dual-agent score calculation. Here selection follows A.6. |
| [`train_dnn.m`](https://github.com/kouzhizhuo/Automate-Strategy-Finding-with-LLM-in-Quant-investment/blob/ebcb5ed44a71664316c99b021026358a44aef38d/train_dnn.m) | Reuse range normalization, but fit it only to training data. The release uses `patternnet(1)`, random splits, and 10,000 epochs; follow the paper's 10-ReLU regression architecture and separate validation instead. |
| Paper §§3.2–3.4, A.6 | Separate filter/categorize/generate operations; two independent judges; strict threshold; best eligible signal per category; neural combination. |

The paper is internally ambiguous: A.5 can retain multiple candidates per category,
while A.6 retains one; this implementation chooses A.6. It does not add the A.7
IC/IR fallback or the release's pairwise contest. The paper does not specify a
general-domain score calibration, generation count, epoch budget, initialization,
optimizer variant or complete loss definition; the choices above make those
implementation details explicit. Categories encourage coverage but do not prove
independence, and a category with no qualifying candidate is omitted.

The paper also switches between nonlinear MLP predictions and a global weighted
sum without explaining extraction of those weights. `local_weights` resolves this
by exposing the MLP's exact piecewise affine representation, including input
normalization and biases. Weights change with active ReLU units. Discarding the
intercepts or treating these as globally constant would change the predictions.

Documents/context are text: callers supply extracted reports, numerical summaries,
image descriptions, audio transcripts or video summaries. Raw multimodal ingestion
is not implemented. Prompt wording and typed JSON contracts deliberately generalize
the finance instructions. Top-k/drop-n trading (A.8) and paper datasets are domain
infrastructure, not part of the generic discovery algorithm.

## Failures and checks

There are no hidden retries or mutable chat sessions. The provider owns transport
retry policy. Malformed JSON, invalid generated scores, duplicate category names,
blank generated candidate content and unexpected evaluation errors propagate.
`CandidateRejected` from evaluation is the one recoverable rejection: it is recorded
and selection continues. Nonfinite measured scores or validation losses fail the run.

`result` includes the factory, selections, both assessments and combined scores,
rejections, model, evaluation-attempt count and generation-call count. Raw responses
are captured before parsing in `agent.generations`, including the response causing a
failed run; transport errors retain an error record. Successful results include
these records too. A provider retry internal to one `acall` is not separately counted.

Run the offline checks from the repository root:

```sh
python -m unittest tests.test_strategy_finding -v
```

The shared scripted provider exercises all prompt operations without model calls.
Synthetic regression checks verify learning, repeatability and exact local-weight
predictions. These establish algorithm behavior, not reproduction of the paper's
reported market returns or evidence of effectiveness on a new task.
