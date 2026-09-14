# Rationale-augmented ensembles

Problem-agnostic Slick implementation of Wang et al.,
[Rationale-Augmented Ensembles in Language Models](https://arxiv.org/abs/2207.00747),
Sections 2.1–2.2. Each independent model call generates a rationale and an answer;
the most frequent answer wins. No training, verifier, benchmark data, or execution
of generated content is involved.

| `method` | Input variation | Output decoding |
| --- | --- | --- |
| `self_consistency` (default) | Fixed exemplars | Sampled |
| `prompt_order` | Independently shuffled exemplar order | Sampled or greedy |
| `input_rationale` | Replace one exemplar rationale with a sampled alternative | Sampled or greedy |

For the paper's additional input-rationale variant, set `replace_all=True` to
replace every exemplar rationale independently. Questions, answers, and exemplar
order stay fixed in both input-rationale variants. Sampling is with replacement;
duplicate prompts and outputs remain legitimate draws, each contributing a vote.

## Use

Use `optimizer/.venv` from the repository root, or install the adjacent Slick
checkout with `python -m pip install -e ../slick`. Checked against local Slick
0.3.0; no new dependencies are required by the algorithm.

Configure the template root once at application startup:

```python
from pathlib import Path
from random import Random

from slick import prompts
import rationale_ensembles
from rationale_ensembles import Example, RationaleEnsemble

prompts.TEMPLATE_ROOT = Path(rationale_ensembles.__file__).resolve().parent / "prompts"


async def solve(task, input, provider, examples):
    agent = RationaleEnsemble(
        task,
        provider,
        examples=examples,
        rng=Random(42),
    )
    return await agent.run(input, samples=40)


# Supply your own task, provider, input, and few-shot examples:
# examples = [Example(input="...", rationale="...", answer="...")]
# result = await solve(task, input, provider, examples)
# print(result.answer, result.counts)
```

`task` specifies the instructions, constraints, and answer representation. `input`
is arbitrary text: a question, document, structured data serialized by the caller,
or another problem. Examples belong to the caller. Omitting examples permits a
zero-shot adaptation; the paper's primary experiments use human rationale seeds.
Evaluation belongs outside this inference algorithm, so no evaluator is required.

The provider must expose **independent stateless calls** and use the desired
decoding settings. The paper uses temperature **0.7**, **40** output samples,
and **128** decoded tokens (**256** for GSM8K). Only the sample count is an agent
parameter; temperature and token limits belong to the provider. For example,
Slick's optional LiteLLM integration forwards backend options:

```python
from slick.providers import LiteLLMAPI

provider = LiteLLMAPI(model, options={"temperature": 0.7, "max_tokens": 128})
```

That adapter requires Slick's optional `litellm` dependency and a backend/model
supporting these settings. A caller may supply any compatible provider instead.
Use a greedy provider for the two input-varying greedy ablations; fixed-prompt
self-consistency requires stochastic decoding to produce useful diversity.
Provider caching or resetting its RNG on every call can defeat sampling. The
injected Python RNG controls prompt variation only, not model randomness.

## Reusable input-rationale pools

Prepare once for a task and exemplar set, then reuse across queries:

```python
agent = RationaleEnsemble(task, provider, examples=examples, rng=Random(42))
pools = await agent.prepare_rationales(samples_per_example=1024)
result = await agent.run(
    input,
    method="input_rationale",
    rationale_pools=pools,
    samples=40,
)
```

Preparation costs **K × samples_per_example** model calls; it is never performed
implicitly by `run()`. Lower this explicit budget for inexpensive experiments.
For each exemplar, generation sees only the other K−1 original exemplars and the
held-out question. Its known answer is used only after generation to filter
outputs. Correctly answered draws contribute their rationale to that exemplar's
pool. Accepted duplicates retain their multiplicity. There is no gold-answer
hint, iterative seed replacement, or regeneration until success.

An empty pool raises after the fixed preparation budget; increase the budget or
improve the seeds before retrying. No human-rationale fallback silently changes
the algorithm. Returned pools are tuples of rationale strings in exemplar order;
reuse them only with the same task, exemplars, and answer-equivalence rule.

For greedy output with sampled input rationales, construct the agent with a greedy
provider and pass a stochastic provider to
`prepare_rationales(provider=sampling_provider, samples_per_example=1024)`.

## Answers, failures, and ownership

The local templates preserve the paper's textual `The answer is X.` delimiter.
They are generic task templates, deliberately replacing the task-specific appendix
prompts. Both prompt operations return text; no JSON structure is imposed.
The default parser uses the last exact delimiter before a subsequent newline
`Q:`, requires nonblank rationale and answer, and removes one terminal period.
It keeps answer case, units, internal punctuation, and spelling unchanged.

Pass `normalize_answer: Callable[[str], str]` to define equivalent answers
(for example `str.casefold` for case-insensitive labels). It applies to generated
answers and exemplar ground truth. For another response format, supply
`parse_response: Callable[[str], tuple[str, str]]`, returning rationale and answer,
and edit both local templates accordingly. Raise `ValueError` to reject malformed
generated content. Other callback errors propagate. Prose answers need a useful
equivalence rule for voting; this implementation does not infer semantic similarity.

`Result` contains the canonical winning `answer`, `counts`, ordered `samples`,
and `consistency`: winning votes divided by all attempted output samples. Each
sample retains its raw response, rationale, canonical answer, and rejection error.
Malformed samples consume budget without replacement and contribute no votes.
All-invalid output raises `ValueError`. Ties choose the first valid answer seen;
a strict majority is unnecessary. Agreement is not a calibrated probability.

`agent.calls` accumulates operation names, rendered prompts, raw responses,
canonical answers, and rejection/error records across preparation and inference.
Provider failures and tool requests abort, retaining partial records. Raw responses
are logged before parsing, including rejected responses. `agent.samples` resets at
each `run()`; prior results keep their sample tuples. Transport retries belong to
the provider/caller and are outside the algorithm's draw count.

Calls run sequentially without Sessions. Use one phase at a time per instance.
Slick's template root is process-global: set it once before rendering; simultaneous
algorithms needing different roots should run in separate processes. Importing this
package does not mutate the template root.

## Source provenance and limits

The supplied [APET repository](https://github.com/daankepel/APET) was inspected,
including its [`experiment.py`](https://github.com/daankepel/APET/blob/main/experiment.py).
It rewrites prompts with an autonomous prompt-engineering toolbox, then evaluates
answers; it is a different algorithm, already implemented under `apet/` here.
It is not used as an official implementation of rationale-augmented ensembles.

On 2026-09-14, the supplied paper, its [arXiv record](https://arxiv.org/abs/2207.00747),
and a repository search did not identify an author-published implementation of
this ensemble algorithm. The paper references
[PromptSource](https://github.com/bigscience-workshop/promptsource) for alternative
task templates, not as an ensemble implementation. This implementation follows the
supplied algorithm directly; no unrelated repository code is vendored or executed.

Uniformly choosing the single replaced exemplar, deterministic first-seen ties,
and rejection without replacement are explicit implementation choices where the
paper does not fully specify operational details. Generalized prompts and current
providers are adaptations, not reproductions of historical PaLM/GPT-3 results.
Benchmark suites and no-rationale/zero-shot-CoT baselines are outside this folder.

## Verification

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_rationale_ensembles
../slick/.venv/bin/ruff check rationale_ensembles tests/test_rationale_ensembles.py
../slick/.venv/bin/ruff format --check rationale_ensembles tests/test_rationale_ensembles.py
```

Offline tests use the shared scripted provider through the real Slick decorator.
They check voting, both input variations, held-out filtering, pool reuse, answer
equivalence, failures, fixed budgets, seeded variation, and template rendering from
the repository and algorithm directories. These establish algorithm/interface
behavior, not improved accuracy or reproduced benchmark scores.
