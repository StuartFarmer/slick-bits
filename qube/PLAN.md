# QUBE implementation plan

Goal: implement the supplied QUBE paper as a runnable Slick bit.
Architecture: ordinary Python owns clustered islands and async evolution;
Slick renders two-parent prompts and calls providers; a Docker worker evaluates
generated Python. NumPy supplies the three deterministic problem evaluators.
Spec: supplied paper, equations 1–2, sections 4.1–4.3, tables 4 and 8–10.

- [x] Implement `search.py`: behavior signatures, offspring means, UIQ, length
  sampling, reset, and rolling metrics. Check equations, attribution, and resets
  with `python -m unittest qube.test_qube`.
- [x] Implement `problems.py`: bin packing with L2 bounds and OR input, cap-set
  greedy construction, and TSP guided 2-opt. Check known tiny optima and invariants.
- [x] Implement `run.py` and `worker.py`: Slick prompts, bounded async workers,
  reset barriers, resource-limited Docker evaluation, offline demonstrations,
  JSONL history, best program and summary. Exercise all three offline CLIs.
- [x] Document paper defaults, cold-start/failed-offspring conventions, usage,
  benchmark limitations, and verified commands in `README.md`.

Constraints: use the existing sibling Slick checkout; no Slick changes, new
frameworks, paid experiment runs, or claims of reproducing published scores.
No repository metadata exists here, so no worktree or commits are needed.

Validation: nine unittest checks passed with Docker enabled; Ruff passed for all
five source/test modules. Offline bin-packing (12 children, two island replacements),
cap-set n=8 (four children), and TSP20 (four children) runs completed and are saved
under `qube/runs/*-smoke`. Built `qube-evaluator:local`. Code review identified and
verified a fix for unbounded native output; both streams now reject above 1 MiB
while the worker is running, with Docker logging disabled. No real LLM calls or
paper-scale experiments were run.
