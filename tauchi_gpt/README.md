# TAUCHI-GPT

A problem-agnostic implementation of the task-processing architecture in
[Farooq et al., *Securing local LLMs for academic research: a human-system integration analysis and evolution of TAUCHI-GPT*](https://doi.org/10.1007/s42454-025-00085-9).
It combines V2's task queue and local retrieval with optional V1 reflection cycles.
This is an algorithm adaptation, not a reproduction of the authors' application,
fine-tuned model, user studies, or security guarantees.

## Use

From this directory, install into your chosen environment:

```sh
python -m pip install -r requirements.txt
```

Run your application from the repository root, or add that root to `PYTHONPATH`.
The existing `optimizer/.venv` already has the required Slick and NumPy dependencies.

```python
from pathlib import Path

from slick import prompts
import tauchi_gpt
from tauchi_gpt import Document, Evaluation, TauchiGPT

# Configure once at application startup; Slick's template root is process-global.
prompts.TEMPLATE_ROOT = Path(tauchi_gpt.__file__).resolve().parent / "prompts"

async def solve(goal, source_paths, provider, embed, evaluate):
    documents = [
        Document(str(path), Path(path).read_text(encoding="utf-8"))
        for path in source_paths
    ]
    agent = TauchiGPT(goal, provider, embed, evaluate)
    return await agent.run(
        documents,
        max_steps=10,
        reflection_cycles=4,
        top_k=5,
    )
```

Supply these dependencies:

- `provider`: a Slick-compatible provider. The same provider handles all six
  prompt operations. Configure the model, context/output limits, and any transport
  retries outside the agent.
- `async embed(texts)`: return one nonzero, finite vector per text, with a consistent
  dimension. Use the same embedding model for documents, task results, and queries.
- `async evaluate(goal, steps) -> Evaluation`: check the overall objective using
  the complete step history. Return `Evaluation(True, "verification feedback")`
  only when your own criterion is met; otherwise return `Evaluation(False, feedback)`.
  Each `Step` exposes `task`, `output.content`, `output.citations`, retrieved `context`,
  and the intermediate `reflections`. Evaluation runs once per finished task after
  all its reflection cycles. It owns any needed sandboxed execution or domain tools.

The goal can describe any domain. Pass `documents=()` for tasks without a source
corpus, or supply text extracted from any format. File selection, PDF/Office parsing,
permissions, model serving, and persistence belong to the application. The library
does not scan directories or interpret generated output as executable commands.

## Algorithm

1. Index the supplied documents. Initialize tasks with a model call, or pass an
   explicit `initial_tasks=[...]` to skip that call.
2. Select the first pending task. Retrieve chunks using the goal and task together.
   Generate its artifact with retrieved evidence and the three most recent artifacts.
3. Run exactly `reflection_cycles` critique/revision pairs on the artifact. There is
   no model-selected early exit or numerical best-candidate selection.
4. Record the task, obtain caller evaluation, and embed the result into run-local memory.
5. Stop if the evaluator confirms overall completion. Otherwise retrieve context
   again, create follow-up tasks using the result and evaluation feedback, and
   reprioritize the queue. Repeat until exhausted or `max_steps` is reached.

`reflection_cycles=0` gives direct execution. Values 4 and 5 reproduce the cycle
counts discussed for V1 System B, composed here with V2's retrieval architecture.
The combined mode is an explicit adaptation; it is not a claim that the paper
specifies exactly this combined implementation.

New tasks receive stable IDs. Normalized exact duplicates of pending or executed
tasks are ignored; this does not detect semantic duplicates. Reprioritization must
return an exact permutation of pending IDs, so a malformed response cannot silently
drop or replace tasks. Zero or one pending task needs no prioritization call.

The local store splits text into overlapping character chunks (defaults: 2,000
characters, 200 overlap), normalizes embeddings, and performs exact cosine ranking.
Ties preserve insertion order. These chunking settings are implementation choices;
the paper does not supply exact values. The store uses existing NumPy rather than
adding Chroma or FAISS. It is rebuilt for each run and retains both document and
generated-result chunks, with distinct origin labels.

## Results, budgets, and failures

`Result.steps` contains every task artifact; `Result.output` is the latest artifact,
not a new synthesis of the whole run. `pending` records unfinished work, and
`evaluations` records caller feedback. Stop reasons are:

- `completed`: the evaluator confirmed the objective.
- `exhausted`: the task queue is empty, without confirmed completion.
- `budget`: the step limit was reached.

At most one initialization call and `max_steps * (3 + 2 * reflection_cycles)`
generation calls occur. Each ordinary step uses execution, creation, optional
prioritization, and two calls per reflection cycle. Confirmed completion skips
creation/prioritization. Planning still runs after the final allowed execution so
the returned pending queue is useful. `max_steps=0` performs no generation,
embedding, or evaluation. The limits count agent calls, not provider-internal
transport attempts or tokens.

Model, parsing, provenance, embedding, and evaluator failures propagate immediately
without automatic retry or fallback. `agent.calls` retains prompts, raw responses,
and prompt-operation errors, including rejected generations. Transport failures
may have no response. Partial steps, evaluations, and pending tasks remain on the
agent. A failed execution/revision leaves its task pending; evaluator failures occur
after its generated step has been recorded. Retrying `run()` starts afresh, not from
a checkpoint. Calls and memory can contain private input: persist them only under
your application's data policy. Use one active run per instance.

## Provenance and security scope

Each retrieved chunk includes its source, exact character offsets in supplied text,
and the SHA-256 of that full text. These hashes identify the ingested snapshot; they
do not authenticate its author or establish that the document is trustworthy.
Generated citations must name a retrieved chunk and contain an exact nonblank quote
from it. An empty citation list is permitted for unsourced work. Presence checking
does not prove that the quote supports the answer or that every claim is cited.
Your evaluator can apply stricter coverage and faithfulness criteria.

Prompts serialize source material as JSON and label it untrusted. These markers
assist instruction/data separation but are not a prompt-injection security boundary.
Provider tool requests are rejected. The agent makes no implicit network calls,
downloads, or cloud fallbacks; **offline operation requires local implementations of
the provider, embedder, and evaluator**. No certification, encryption, adversarial
training, model-weight inspection, UI, or multimodal API integration is implemented.
Those are application infrastructure or proposed mitigations, not the task-loop algorithm.

## Official source references and adaptations

Sources inspected on 2026-09-14:

- The [published article](https://link.springer.com/article/10.1007/s42454-025-00085-9),
  especially §3.1.2, §3.2.2, Fig. 5, and §5. Its data-availability statement says
  “TAUCHI-GPT is available at GitHub here” but supplies no clickable repository target.
  Public GitHub repository search for `TAUCHI-GPT` returned no matches. An official
  TAUCHI-specific source tree could not be verified; none is claimed or silently
  substituted here.
- The cited [official BabyAGI repository](https://github.com/yoheinakajima/babyagi)
  points to its original task-planning implementation in
  [babyagi_archive, commit e8726e3](https://github.com/yoheinakajima/babyagi_archive/blob/e8726e3f83b81e51bdbfebf6df6ab034d52e58ec/babyagi.py).
  Its `execution_agent`, `context_agent`, `task_creation_agent`,
  `prioritization_agent`, and `main` informed this implementation's ordering.
  The archive implements execute → store result → create tasks → prioritize.
  This implementation uses result passages instead of returning only task names,
  adds caller completion evaluation and bounded execution, and uses checked JSON
  IDs instead of parsing/renumbering free-form lists. It does not import the
  upstream application, dependency stack, or cloud-fallback behavior.
- The cited [official AutoGPT classic prompt strategy](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/original_autogpt/autogpt/agents/prompt_strategies/one_shot.py)
  was inspected for its planning, observations, self-criticism, and structured action
  output. TAUCHI's separate critique/revision operations express the paper's fixed
  reflection cycles using Slick; AutoGPT's command execution is outside this core.
  The current AutoGPT file is a reference, not the unavailable historical TAUCHI fork.

Prompts are newly written for arbitrary objectives, not copies of the study's
academic prompts. Slick replaces LangChain orchestration; provider injection
replaces the fixed LLaMA2/Gradient setup. These choices and the completion, ID, and
citation contracts are deliberately exposed rather than presented as unspecified
details recovered from the original implementation.

## Checks

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_tauchi_gpt
../slick/.venv/bin/ruff check tauchi_gpt tests/test_tauchi_gpt.py
../slick/.venv/bin/ruff format --check tauchi_gpt tests/test_tauchi_gpt.py
```

Tests use the repository's shared scripted provider and small deterministic vectors.
They verify control flow, schemas, citation provenance, budgets, state isolation,
and template rendering, including a different working directory. They do not
measure LLM quality, embedding retrieval quality, attack resistance, or reproduce
the paper's user-study results. Checked against the adjacent Slick 0.3.0 source.

Validation on 2026-09-14: eight TAUCHI tests and the shared prompt-layout check passed;
scoped Ruff lint and formatting passed. Repository-wide discovery ran 662 tests with
seven unrelated errors (missing `prompt_optimization` modules/catalogue files and
`sklearn`) and four skips.
