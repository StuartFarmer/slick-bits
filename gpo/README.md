# GPO

`GPO(task, provider, evaluate, embed=embed).run(initial_prompt, examples=...)`
defaults to the authors' highlighted **GPO recipe**: retrieve the current and
semantically relevant past prompts, supply their measured scores to generation,
and reduce the allowed word edits using a cosine schedule. `embed(texts)` returns
sentence embeddings; local code computes cosine similarity and selects history.
Both `embed(texts)` and `evaluate(prompt)` are async. Evaluation is higher-is-better
and uses a fixed validation set. This port requires NumPy in addition to Slick.
Task examples can show the instruction's position in any input/output format.

The best previously unselected proposal becomes the next trajectory point, even
when its score decreases; `best` separately retains the global maximum. Exact
text duplicates reuse measurements. The word allowance is a model instruction,
not a post-hoc edit-distance filter, matching the release. Exceptions propagate.
Configure Slick's template root to `gpo/prompts` before execution.

Optional feedback ablations use `recipe="feedback"` plus a `failures(prompt)`
callback, `momentum="feedback"` or `"parameter"`, and `selection="importance"`
or `"recency"`. Gradient utility is the measured change in the selected prompt's
score. Fixed and linear schedules are also available. The returned history
contains retrieved parameters, feedback, word budgets and chosen candidates.

Inspected [official RUCAIBox/GPO at 8629ceec](https://github.com/RUCAIBox/GPO/tree/8629ceecfd9aa8a2e4c0298d7d0607a4faed9901):
`GPO.sh`, `src/optimization/optimize.py`, `selection_methods/relavance_selection.py`,
recency/importance selection, `utilize_gradient/generate_without_gradient.py`,
edit-based generation, `learning_rate_scheduler/consine.py`, and `learning_rate/w_lr.py`.
The [paper](https://arxiv.org/abs/2402.17564) also evaluates many alternative
configurations; this port does not include real-time momentum summarization or
all feedback/generation combinations. Feedback importance uses a total bounded
memory size, whereas the release adds the latest item beyond its selected count.

Task-neutral templates replace hardcoded QA formatting; caller embeddings replace
constructing a Llama-2 sentence transformer on every retrieval. Similarity ties
are stable. Scores are unbucketed. Deterministic tests check history retrieval,
novelty preference, schedules and the no-feedback recommended path; no benchmark
results, model weights or paid calls are included.

Two release bookkeeping defects are deliberately corrected: the visited set
starts with the initial prompt, rather than its numeric score, and the incumbent
keeps its measured score instead of a temporary zero during candidate ranking.
