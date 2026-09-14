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
A plain string is also accepted, with heuristic sentence splitting that preserves
whitespace and common question/answer labels. It can misinterpret abbreviations
or complex markup; explicit fragments avoid that ambiguity.

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

Results contain the best prompt, final beam, initial score, per-edit history
(including raw responses and rejections), and distinct evaluation count.
Pass `session=your_session` to `run` for an intentional sequential conversation;
otherwise generation calls independently use the injected provider. The global
template root means different applications should use separate processes if they
need to render concurrently. Each run resets the agent's search state; use one run
at a time per instance. Caller types are trusted; generated mutations, scores,
embeddings, and numerical search constraints are still checked.

Run offline checks from the repository root with a Python environment containing
Slick and NumPy: `python -m unittest tests.test_apex`.
The former dataset CLI and model adapters are preserved in
`examples/legacy/2026-09-14-problem-specific.tar.gz`; historical `runs/` remain in place.
