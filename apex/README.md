# APEX

`APEX` improves individual sentences in a caller-supplied prompt using immediate
beam updates, LinUCB sentence selection, and examples retrieved from edit history.
The caller supplies the task description, Slick provider, async evaluator, and
synchronous batch encoder. No task dataset, model construction, or runner is built in.

```python
from pathlib import Path

from slick import prompts

import apex
from apex import APEX, Config, Document

# Configure Slick's process-global template root once before rendering.
prompts.TEMPLATE_ROOT = Path(apex.__file__).parent / "prompts"

agent = APEX(task, provider, evaluate, encode)
document = Document(("Instructions:\n", initial_sentence, "\nOutput: JSON"), (1,))
result = await agent.run(document, config=Config(iterations=20, beam_size=4))
best_instruction = result["best"]["prompt"]
```

`evaluate(text)` must asynchronously return a finite score; higher is better.
Scores are cached for each run, so the evaluator should be repeatable.
`encode(sentences)` returns a finite two-dimensional matrix with one nonzero
vector per sentence and a stable vector dimension. Embeddings are normalized
and cached by sentence text; no embedding model is selected by the agent.

Use `Document(parts, mutable_indices)` to preserve exact immutable boundaries.
For a text file, use `Document.split(Path("prompt.txt").read_text())` and pass
that document to `run`. Splitting preserves whitespace and common question/answer
labels, but can misinterpret abbreviations or complex markup; explicit fragments
avoid that ambiguity. A document with no mutable fragments returns its initial
evaluation without generation. `run` now accepts only `Document`; callers that
previously passed strings must convert them explicitly with `Document.split`.

`Config` exposes the iteration budget, beam size, random-versus-LinUCB selection
probability, bandit exploration and regularization, history limit and distance,
guided mutation toggle, and random seed. Each valid mutation immediately updates
the beam and bandit. Negative history entries are reversed into improvement
examples. `run` composes sentence selection, mutation, and beam update phases.
The decorated `mutate` and `mutate_guided` methods each own a template and return
a checked `Mutation`, retaining raw output for rejected edits.
Empty, multiline, tagged, label-prefixed, and fenced
responses consume an iteration without evaluation. Provider, evaluator, and
encoder exceptions propagate without retries.

Defaults follow Section 4.1: beam size 4, exploration alpha 0.05, random sentence
probability 0.5, and at most four nearest nonzero-reward history entries within
strict normalized Euclidean distance 0.5 (equivalent to cosine similarity > 0.875).
Similarity uses the sentence **before** each historical edit; negative edits are
then reversed for display. LinUCB uses all valid edits, including zero rewards,
with reward measured relative to the selected parent. Its dual ridge calculation
is algebraically equivalent to Equations (1)–(2). Ridge regularization defaults
to 1.0 because the paper does not specify it. Ties retain older beam candidates;
parents are sampled uniformly from the current beam.

`iterations=50` permits 50 mutation attempts plus the initial evaluation, hence
at most 51 distinct prompt evaluations. Invalid and duplicate mutations can
reduce evaluation count. Set `iterations=49` for a total ceiling of 50.

Results contain the best prompt, final beam, initial score, per-edit history
(including raw responses and rejections), and distinct evaluation count.
Pass `session=your_session` to `run` for an intentional sequential conversation;
otherwise generation calls independently use the injected provider. The global
template root means different applications should use separate processes if they
need to render concurrently. Each run resets the agent's search state; use one run
at a time per instance. Caller types are trusted; generated mutations, scores,
embeddings, and numerical search constraints are still checked.

Use training data only in `evaluate`; evaluate the returned best prompt on held-out
data separately. Supply a semantic sentence encoder (the paper uses T5); the agent
normalizes its outputs. Configure mutation temperature on your provider (the paper
uses 0.5), and target-model evaluation temperature in your evaluator (the paper
uses 0). Plain text is the generated contract; `Mutation` is the postprocessed
record, not a generated JSON schema. The local templates adapt Figures 2 and 4
with task context and output-format instructions. Semantic preservation is requested,
not proven; inspect edits before use. Passing a Session intentionally adds
conversation history beyond the paper's retrieved examples.

Source: Hsieh et al., [Automatic Engineering of Long Prompts, ACL 2024](https://aclanthology.org/2024.findings-acl.634/).
No official APEX implementation was identified when checking the paper and its
linked repositories on 2026-09-14. This is a local implementation of Sections
3.1–3.3. The paper links [BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)
for `cot-prompts/` and [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub)
for GSM8K prompts and runners; those are benchmark resources, not APEX search code.
They can supply initial prompt text and evaluator data without changing this agent.

Run offline checks from the repository root with a Python environment containing
Slick and NumPy: `python -m unittest tests.test_apex`.
These scripted checks validate algorithm behavior and Slick integration; they do
not reproduce the paper's accuracy gains or establish performance on a real task.
The former dataset CLI and model adapters are preserved in
`examples/legacy/2026-09-14-problem-specific.tar.gz`; historical `runs/` remain in place.
