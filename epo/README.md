# EPO / TRIPLE

Fixed-budget best-arm identification for a supplied pool of instructions.
`EPO(task, evaluate).run(prompts, budget=100, algorithm="sequential_halving")`
returns the best measured prompt, per-arm pulls and means, surviving arms, and
elimination history. `continuous_rejects` implements the official CR-A/CR-C union.
Each async `evaluate(prompt)` call is a fresh reward observation. Higher is better;
use [0, 1] rewards for the continuous-rejection threshold's assumptions.

Sources inspected: [paper](https://arxiv.org/abs/2402.09723), official
[TRIPLE](https://github.com/ShenGroup/TRIPLE/tree/be15d626c755a55039a6ee075a089ac45a936cff),
`src/bandit/stochastic/{sequential_halving,continuous_rejects}.py`.
This is the survey's EPO entry; its official implementation calls the framework
TRIPLE. The sequential-halving allocation and continuous-rejection threshold are
ported; contextual clustering/GSE variants are not included.

Adaptations: enforce the requested pull budget even when it cannot cover every
arm, exclude unobserved arms from final selection, and choose the measured best
survivor instead of upstream SH's last array entry. Stable ties keep earlier
prompts. Evaluator errors and nonfinite observations propagate, without retries.
The caller supplies the candidate pool, so no Slick generation boundary or prompt
templates are needed. Tests exercise allocation/selection, not benchmark accuracy.
