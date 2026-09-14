# Minerva inference

Problem-agnostic Slick implementation of the inference procedure in
[Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858)
(Lewkowycz et al., 2022), sections 2.3–2.5 and Appendix D:
sample independent solutions, extract their final answers, group matching answers,
and return the most frequent answer (`maj1@k`) or the top `n` groups (`majn@k`).
“Majority” means the most frequent answer; it need not receive over half the votes.

This implements inference with a caller-provided model. It does not reproduce
Minerva's PaLM continued pretraining, technical corpus, checkpoints, or reported
benchmark performance.

## Use

Use the existing `optimizer/.venv` environment or install the adjacent Slick
checkout with `python -m pip install -e ../slick` from the repository root.
No new dependencies are required by the algorithm.

```python
from pathlib import Path

from slick import prompts
import minerva

# Configure once at application startup.
prompts.TEMPLATE_ROOT = Path(minerva.__file__).resolve().parent / "prompts"


async def solve(task: str, provider):
    agent = minerva.Minerva(task, provider)
    result = await agent.run(k=16, n=1)
    return result.answer, result.output  # final answer and representative solution
```

Put your instructions, constraints, and source material in `task`. Supply optional
`examples=[minerva.Example(problem, solution, answer), ...]` for few-shot prompting.
Examples are fixed across samples. There are no built-in mathematics examples or
benchmark assumptions. Results and generated code are never executed.

The caller owns credentials, model selection, tokenization, decoding, and transport
retries. The paper uses temperature `0.6`, nucleus probability `0.95`, and up to
512 generated tokens for multiple samples; single-sample evaluation uses greedy
decoding. Configure the provider accordingly. The checked Slick `LiteLLMAPI`
supports these settings via `options` (requires its optional LiteLLM dependency):

```python
from slick.providers import LiteLLMAPI


def make_provider(model: str, k: int):
    return LiteLLMAPI(
        model,
        max_retries=0,
        options={
            "temperature": 0.0 if k == 1 else 0.6,
            "top_p": 1.0 if k == 1 else 0.95,
            "max_tokens": 512,
        },
    )
```

Use a backend that supports these settings. `run(k=1)` does not reconfigure an
already-created provider. Repeated calls to a deterministic provider do not supply
the diversity that this method depends on. The generic agent preserves the entire
prompt; the paper's left truncation to 1024 input tokens belongs in a caller's
model/tokenizer adapter when reproducing that experiment.

## Answers, votes, and evaluation

The local template uses textual solutions and the paper's final-answer delimiter:

```text
Final Answer: The final answer is ANSWER. I hope it is correct.
```

The default extractor reads the last such line, requires the closing marker, and
preserves the answer's notation, punctuation, units, and internal whitespace.
Missing, truncated, or empty answers consume a sample but do not vote. The default
normalizer only strips surrounding whitespace. Case, units, and equations are
not discarded. Inject `extract(response) -> str | None` and
`normalize(answer) -> str` to support your own answer format or equivalence rules.
Return `None` from extraction or an empty normalization key to reject a sample;
callback exceptions propagate.

`Result.samples` retains every raw response, extracted answer, normalized key, and
rejection reason. `votes` contains every valid group ranked by count; `selected`
contains up to `n` groups. Ties go to the first sampled group. Each `Vote` retains
its member indices and first answer as representative. `answer` and `output` are
`None` when no group is selected. `calls` includes invalid samples. Records reset
on each run; do not overlap runs on one instance.

For optional reference evaluation, pass an async `evaluate(answer) -> bool`:

```python
async def check_label(answer):
    return answer.casefold() == "urgent"


async def classify(task, provider):
    agent = minerva.Minerva(
        task, provider, check_label, normalize=str.casefold,
    )
    return await agent.run(k=16, n=5)
```

Evaluation runs after sampling and selection. References never enter generation
or influence voting. `correctness` records each sample's correctness (invalid
samples are false). `pass_at_k` reports whether any sample is correct; `maj_at_n`
reports whether any selected representative is correct. Both are per-problem
booleans, not dataset accuracy estimates, and are `None` without an evaluator.
Ensure your normalization groups only answers your evaluator considers equivalent.
These checks concern final answers, not the validity of the preceding explanation.

The paper's SymPy comparison is a reference-grading procedure, separate from
frequency selection. Supply a domain-specific evaluator if needed; the agent
does not evaluate model text as Python or impose mathematical parsing on other
tasks. Symbolic evaluation timeouts and execution isolation belong to that caller.

## Sources and adaptations

Sources inspected on 2026-09-14:

- The supplied paper and [Google's project write-up](https://research.google/blog/minerva-solving-quantitative-reasoning-problems-with-language-models/)
  establish the sampling and voting procedure. This implementation uses that
  procedure and delimiter, with caller-provided examples and general instructions.
- The supplied [APET repository](https://github.com/daankepel/APET) implements
  *Autonomous Prompt Engineering in Large Language Models*, a different paper.
  Its implementation already lives in [apet/](../apet/README.md); its rewrite
  algorithm is not part of Minerva.
- The paper names [official T5X](https://github.com/google-research/t5x) as its
  training framework. Its [decoding source](https://github.com/google-research/t5x/blob/main/t5x/decoding.py)
  provides `temperature_sample`, including temperature and nucleus sampling.
  It is a model-level reference, not a standalone Minerva voting implementation.
  Here the injected provider owns decoding, so T5X/JAX training infrastructure is
  not imported or copied into the generic agent.
- No official standalone Minerva inference implementation was identified in the
  inspected sources. The paper's [supplementary archive](https://storage.googleapis.com/minerva-paper/minerva_supplementary_data.zip)
  could not be inspected because the download returned HTTP 403.

Stable first-occurrence ties, abstention on malformed samples, and strict closing
delimiter checks are explicit implementation choices. The generic prompt is an
intentional adaptation, not the paper's exact four-shot mathematics prompt.
There is no log-likelihood reranking, reference-guided selection, or verifier call
inside generation.

## Failures and checks

`agent.calls` retains prompts, raw responses, and errors, including the failing
call. Provider failures, unexpected tool requests, and callback errors stop the
run without repair calls. Completed samples remain available. Evaluator failures
also propagate. Transport retries inside a provider are outside `calls` accounting.
Use a stateless provider without autonomous tool execution; no Session is created.

Imports do not alter Slick's process-global template root. Configure it before
rendering; algorithms needing different roots concurrently require separate
processes.

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_minerva
../slick/.venv/bin/ruff check minerva tests/test_minerva.py
../slick/.venv/bin/ruff format --check minerva tests/test_minerva.py
```

Tests exercise real Slick decorators/templates with the repository's shared
scripted provider: independent samples, plurality/ties, top-n selection, rejected
answers, custom normalization, evaluation isolation, failure records, reset, and
template loading from both launch directories. These are offline correctness
checks; no paid model experiment or benchmark reproduction has been run.
