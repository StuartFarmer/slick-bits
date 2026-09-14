# SELF-REFINE

Problem-agnostic implementation of [Self-Refine: Iterative Refinement with
Self-Feedback](https://arxiv.org/abs/2303.17651), Algorithm 1 and Equations 1–4.
One model generates an output, critiques it, and refines it using the full history.
The algorithm returns the **last output**, including when the budget expires.

## Use

Use the repository's existing Slick environment (`optimizer/.venv`), or install
the adjacent checkout with `python -m pip install -e ../slick` from the repository
root. No additional dependencies are needed.

Configure the template root once at application startup, before making calls:

```python
from pathlib import Path

from slick import prompts
import self_refine

prompts.TEMPLATE_ROOT = Path(self_refine.__file__).resolve().parent / "prompts"


async def solve(task, input, provider):
    agent = self_refine.SelfRefine(task, provider)
    result = await agent.run(input, max_refinements=4)
    return result.output
```

`task` describes the goal, constraints, quality criteria, and output format.
`input` is the material to work on: a question, document, specification, code, or
any other text. Supply the same configured Slick provider for the entire run;
the caller chooses its model, decoding settings, credentials, and transport policy.
The agent never executes generated content or requires a benchmark or evaluator.

To improve something you already have, pass
`await agent.run(input, initial_output=draft)`. This skips initial generation,
as in the paper's code-readability setup. Empty supplied drafts are passed through
unchanged; generated responses must contain non-whitespace text. Output bytes,
including leading and trailing whitespace, are otherwise preserved as strings.

## Prompts and stopping

The three separate local Jinja templates provide general instructions. Customize
them using the constructor's `generation_prompt`, `feedback_prompt`, and
`refinement_prompt` text. These are appended instructions/examples for their
respective stages, not Jinja code. Put shared requirements in `task`.

Few-shot examples can follow the paper's formats: input/output for generation,
input/output/feedback for feedback, and input/output/feedback/revision for refinement.
Defaults are zero-shot and deliberately contain no benchmark-specific examples.
The bundled prompts are a generic redesign, not exact reproductions of the paper's
experimental prompts. For complete control, edit the three local templates.

By default, feedback must be exactly `NO_FEEDBACK` (ignoring surrounding whitespace)
to stop. A mention of that marker inside a longer critique does not stop the loop.
Override both the feedback instructions and the stopping function for another format:

```python
agent = self_refine.SelfRefine(
    task,
    provider,
    feedback_prompt="When all requirements are satisfied, reply exactly ACCEPT.",
    stop=lambda feedback, iteration: feedback.strip() == "ACCEPT",
)
result = await agent.run(input)
```

`stop(feedback: str, iteration: int) -> bool` is synchronous and replaces the default
check. Its iteration is zero for feedback on the initial draft. Returning true
stops **before** refinement. For a fixed number of revisions, use
`stop=lambda feedback, iteration: False`; the hard cap still applies.

`max_refinements=N` permits N revisions and N+1 feedback calls, plus one initial
generation unless a draft is supplied. Every final output receives feedback, even
at the cap, matching Algorithm 1's feedback-before-stop ordering. Thus N=4 costs at
most 10 calls (9 with a supplied draft); N=0 gives feedback but no revision.
Feedback stopping takes precedence if both conditions hold at the cap.

## Results and failures

`Result` contains `output`, an immutable `history` tuple of `Step(output, feedback)`
pairs, `stop_reason` (`"feedback"` or `"budget"`), and the number of `calls`.
Each critique sees the input and current output only. Each refinement sees the
input and **all** previous output/feedback pairs in chronological order.

`agent.output`, `agent.history`, and `agent.calls` retain partial progress on failure.
Call records contain the operation, rendered prompt, raw response when available,
and any exception. Raw responses are recorded before generated-output validation.
Blank generated text, unexpected tool requests, provider errors, and callback
errors abort the run; there are no implicit retries or fallback drafts. A callback
failure occurs after its successful feedback call and leaves that history intact.
The caller owns persistence and any retries, including transport attempts hidden
inside a provider. Records reset on each run; use one run at a time per instance.

History is assembled explicitly without a shared Session, so earlier requests do
not leak into feedback prompts. Use a provider that does not add implicit chat
history. Slick's template root is process-global: configure it once, and use
separate processes for simultaneous implementations needing different roots.
Full history is not truncated; its context cost grows with the number of revisions.

## Official implementation reference

The supplied [gepa-ai/gepa](https://github.com/gepa-ai/gepa) link implements
Genetic-Pareto reflective optimization, a separate algorithm. This folder follows
the supplied SELF-REFINE paper and its official implementation below.

The [project site](https://selfrefine.info/) identifies
[madaan/self-refine](https://github.com/madaan/self-refine) as the official code.
This implementation uses its CommonGen flow and history construction as references,
re-expressed with generic text and Slick instead of the task-specific backend.
Sources checked on 2026-09-14 at commit
[`9a206d4`](https://github.com/madaan/self-refine/commit/9a206d41e5d2d0c241bb441f41eeadb945afaa55):

| Official source | Behavior retained here |
| --- | --- |
| [`src/commongen/run.py`](https://github.com/madaan/self-refine/blob/9a206d41e5d2d0c241bb441f41eeadb945afaa55/src/commongen/run.py) | One model for all stages, bounded iteration, feedback-based stopping, ordered history |
| [`src/commongen/task_init.py`](https://github.com/madaan/self-refine/blob/9a206d41e5d2d0c241bb441f41eeadb945afaa55/src/commongen/task_init.py) | A separate initial generation prompt with optional examples |
| [`src/commongen/feedback.py`](https://github.com/madaan/self-refine/blob/9a206d41e5d2d0c241bb441f41eeadb945afaa55/src/commongen/feedback.py) | Specific feedback on the current output against the original requirements |
| [`src/commongen/task_iterate.py`](https://github.com/madaan/self-refine/blob/9a206d41e5d2d0c241bb441f41eeadb945afaa55/src/commongen/task_iterate.py) | Refinement using every previous output and feedback pair |

The upstream code is [Apache-2.0 licensed](https://github.com/madaan/self-refine/blob/9a206d41e5d2d0c241bb441f41eeadb945afaa55/LICENSE).
Its `max_attempts` counts candidate/feedback rounds, including the initial candidate;
our `max_refinements` counts revisions after that candidate. There is no dependency
on upstream `prompt-lib`, NLP tooling, datasets, task-specific feedback corrections,
or experiment scripts. The paper's task-specific best-score selection and oracle
feedback variants are outside this implementation of Algorithm 1.

## Verification

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_self_refine
../slick/.venv/bin/ruff check self_refine tests/test_self_refine.py
../slick/.venv/bin/ruff format --check self_refine tests/test_self_refine.py
```

Tests use the shared scripted provider through the real Slick decorator. They check
call ordering, complete history, early and budget stopping, custom prompts and
stopping, supplied drafts, state reset, failure records, and template rendering
from another working directory. These are offline algorithm/interface checks;
they do not establish quality improvements or reproduce the paper's benchmark results.
