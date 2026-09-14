# Self-consistency

Slick implementation of Wang et al., [Self-Consistency Improves Chain of Thought
Reasoning in Language Models](https://arxiv.org/abs/2203.11171), Section 2:
sample independent responses to the same prompt, extract each final answer, then
return the answer with the most votes. Duplicate responses each contribute a vote.
No verifier, training, prompt search, or benchmark-specific code is required.

## Use

Use the existing `optimizer/.venv` environment, or install the adjacent Slick
checkout with `python -m pip install -e ../slick` from the repository root.
No additional dependencies are needed. Checked against local Slick 0.3.0.

Configure the local template root once at application startup:

```python
from pathlib import Path

from slick import prompts
import self_consistency

prompts.TEMPLATE_ROOT = Path(self_consistency.__file__).resolve().parent / "prompts"


async def solve(task, input, provider):
    agent = self_consistency.SelfConsistency(task, provider)
    result = await agent.run(input, samples=10)
    return result.answer, result.consistency
```

`task` supplies instructions, constraints, and the answer representation; `input`
is arbitrary textual task data. Optional `examples` supplies caller-written
few-shot Q:/A: exemplars with worked solutions and final answers. Without examples,
the local template is a generic zero-shot adaptation. Edit that template for
another output format; it is not the paper's benchmark-specific prompt collection.
Generated text is never executed. Evaluation, if needed, happens outside the agent.

Supply a **stateless provider configured for stochastic decoding**, using the same
model for every sample. The paper uses temperature 0.7 for GPT-3; top-k/top-p and
temperature are provider settings, not template parameters. This agent does not
change them or guarantee diversity from a greedy or caching provider. The checked
Slick decorator passes only context to `acall`; inject a provider that exposes the
decoding controls your backend supports. Do not share conversational history or
reset a random seed to the same value for each call.

For example, Slick's `LiteLLMAPI` forwards sampling options to the selected backend
(requires Slick's optional `litellm` extra and a model supporting temperature):

```python
from slick.providers import LiteLLMAPI


def sampling_provider(model):
    return LiteLLMAPI(model, options={"temperature": 0.7, "max_tokens": 2048})
```

The default budget is 40 samples, matching the paper's main experiments. A budget
of 5 or 10 is useful for cheaper exploration. Calls run sequentially and use no
Session; each receives an identical prompt and never sees prior generated paths.
Slick's template root is process-global. Configure it before rendering, and use
separate processes for simultaneous algorithms requiring different roots.

## Answer identity and results

The default extractor reads the last exact `The answer is ` marker before the
first `Q:`, strips surrounding whitespace and one final period, and preserves the
remaining text. Missing/empty answers are rejected. It does not merge case, units,
numeric spellings, synonyms, or paraphrases. Multiple markers use the last answer
as an explicit convention. Supply a synchronous `extract_answer(response) -> str`
to parse and canonicalize answers for your task:

```python
from decimal import Decimal


def amount(response):
    answer = self_consistency.extract_answer(response)
    return str(Decimal(answer.removeprefix("$")).normalize())


agent = self_consistency.SelfConsistency(task, provider, extract_answer=amount)
```

A custom extractor can parse JSON, validate a label set, or normalize equivalent
artifacts. Raise `ValueError` to reject an output; other exceptions propagate.
For example, catch and translate `decimal.InvalidOperation` in `amount` if malformed
numbers should be rejected instead of aborting. Custom canonical strings are kept
exactly, but must contain non-whitespace text. The task/template should request the
format your extractor expects.

`Result` contains `answer`, `counts` per canonical answer, `consistency` (winning
votes divided by **all** sampled responses), and ordered `samples`. Each `Sample`
has raw `response`, canonical `answer` or `None`, and a rejection `error` or `None`.
Voting selects a plurality; a strict majority is not required. Ties go to the
first encountered valid answer. Agreement is not a calibrated correctness probability.
Arbitrary prose needs a meaningful equivalence rule before vote counts are useful.

Malformed answers consume the budget and contribute no votes. There are no repair
calls or replacement samples. If every answer is rejected, aggregation raises
`ValueError`. This rejection policy and deterministic tie rule are explicit
implementation choices where the paper does not specify behavior.

Provider errors, unexpected tool requests, and extractor programming errors abort
the run. Partial `agent.samples` and `agent.calls` remain available. Call records
retain rendered prompts, raw responses before extraction, and errors/rejections.
`run()` resets these records; use one run at a time per instance. Caller-owned
transport retries are outside the sample count. No error is turned into a vote.

## Sources and scope

The supplied [daankepel/APET](https://github.com/daankepel/APET) repository's
[`experiment.py`](https://github.com/daankepel/APET/blob/775ab29323d794c6f445156a0c0d344841e7196a/experiment.py)
implements autonomous prompt engineering, a different algorithm already represented
in this repository's `apet/` folder. It is not an official self-consistency implementation.

Checked on 2026-09-14: the supplied paper, its
[arXiv record](https://arxiv.org/abs/2203.11171), and the
[Google Research publication page](https://research.google/pubs/self-consistency-improves-chain-of-thought-reasoning-in-language-models/)
did not identify an author-published self-consistency code repository. The paper's
official [UL2 repository](https://github.com/google-research/google-research/tree/master/ul2)
and [BIG-bench StrategyQA task](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/strategyqa)
are model/data resources, not implementations of this decoding algorithm. No
third-party implementation is presented here as official; this is a direct
implementation of the supplied paper's unweighted answer marginalization.

Weighted probability aggregation in Table 1, beam-search/sample-and-rank ablations,
and benchmark runners are omitted. Slick's portable text response contract does
not expose token log probabilities; the paper's primary majority-vote method needs
none. Defaults and general prompts do not reproduce its historical model results.

## Verification

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_self_consistency
../slick/.venv/bin/ruff check self_consistency tests/test_self_consistency.py
../slick/.venv/bin/ruff format --check self_consistency tests/test_self_consistency.py
```

Offline tests use the shared scripted provider through the real Slick decorator.
They check fixed-budget independent calls, plurality and ties, task-specific answer
equivalence, rejected outputs, partial failure records, and template rendering from
the repository and algorithm directories. They verify algorithm/interface behavior,
not improved model accuracy or reproduction of the paper's benchmark scores.
