# MEOH

Problem-agnostic [Multi-objective Evolution of Heuristic Using Large Language
Model](https://arxiv.org/abs/2409.16867), Yao et al., AAAI 2025. One `MEOH` class
evolves descriptions and artifact text against any number of caller-measured
objectives. It returns a non-dominated archive of evaluated alternatives.

```python
from pathlib import Path

from slick import prompts
import meoh


async def optimize(task, provider, evaluate, maximize, similarity=meoh.python_similarity):
    # Configure once at application startup, before rendering any prompts.
    prompts.TEMPLATE_ROOT = Path(meoh.__file__).resolve().parent / "prompts"
    agent = meoh.MEOH(
        task, provider, evaluate, maximize=maximize, similarity=similarity
    )
    pareto = await agent.run(population_size=20, generations=20, parents=5, seed=0)
    return pareto, agent
```

Supply the artifact's interface, constraints, and objective meanings in `task`.
`evaluate(content: str)` is async and returns a sequence of measured floats in a
fixed order. For example, `(quality, runtime, size)` uses
`maximize=(True, False, False)`. The evaluator owns data, syntax/interface checks,
execution isolation, and timeouts. The agent never executes generated content.
Raise `CandidateRejected` for an expected infeasible artifact; other evaluator
errors propagate. Each returned `Individual` has `id`, `description`, `content`,
and the original, unnegated `objectives`.

For Python heuristics, install `python -m pip install -r meoh/requirements.txt`
in the application environment alongside Slick. The default calls precisely
`codebleu.syntax_match.calc_syntax_match([reference], candidate, "python")`, as
LLM4AD does; it does not substitute a token or edit-distance approximation.
CodeBLEU imports lazily. Its PyPI 0.7.0 dependency bounds conflict with LLM4AD's
Python grammar on this platform, so the requirements pin an upstream revision
whose syntax matcher is unchanged and whose dependency bounds permit Tree-sitter
0.23.2. This combination was tested with Python 3.14 on macOS ARM64.

For another language or arbitrary artifacts, pass a deterministic
`similarity(reference, candidate) -> float` in `[0, 1]`, where 1 means maximally
similar. For example, a prose experiment can use
`lambda a, b: SequenceMatcher(None, a, b, autojunk=False).ratio()` from `difflib`.
That changes the paper's AST metric deliberately and requires no CodeBLEU install.

The search computes `v[j] = -sum(similarity(i, j) for i dominating j)`, samples
parents with softmax probabilities, and retains the largest scores. Similarity
is directional; there is no `1 - similarity` substitution or objective scalarization.
Dominance is strict, with equality allowed on individual axes but improvement
required on at least one. Equal selection scores retain insertion order.
Parent draws are independent and may repeat an individual.

Initialization allows at most `init_attempts` calls (default `3 * population_size`)
to obtain a full population; failure raises with records retained. Each subsequent
generation makes exactly `population_size` generation attempts, cycling through
E1, E2, M1, M2, M3 across generation boundaries. Each accepted offspring immediately
joins the available parent pool; truncation occurs at generation end. This follows
the N-offspring loop in paper Algorithm 1. The paper's experimental totals count
five operator passes: use 100 such generations with N=20 for 2,000 offspring
attempts, plus initialization. This implementation's `generations=20` means 400
offspring attempts, not the experimental 2,000.

Malformed generated JSON, blank artifacts, provider errors, evaluation timeouts,
explicit rejections, and nonfinite or dimensionally wrong objective measurements
consume attempts. They do not trigger hidden repair calls. Transport retries
belong to the supplied provider. `agent.attempts` retains raw responses before
parsing, proposals, measured objectives, parent IDs, statuses, and failures;
`evaluations` counts evaluator invocations separately. `population` is the bounded
working population; `history` contains its initial and end-of-generation snapshots.
The archive includes every accepted candidate's contribution, even when working
population truncation discards it. Identical content/objective pairs are collapsed
in the archive, while distinct artifacts with equal scores remain available.

Prompt methods are independent Slick calls with explicit `output_type=Proposal`
and no Session history. Each operation owns a separate Jinja template; shared
partials only render supplied data. Set Slick's process-global template root once;
different roots running concurrently require separate processes. Imports do not
change that root. Tests use the repository's shared `tests.providers.ScriptedProvider`.

Official source used: [Optima-CityU/LLM4AD at
ffb6acf64497be93932c98d25369352efd3865cf](https://github.com/Optima-CityU/LLM4AD/tree/ffb6acf64497be93932c98d25369352efd3865cf/llm4ad/method/meoh),
specifically `population.py` (masked AST penalties, softmax, truncation, archive),
`meoh.py` (operator cycling and repeated parent sampling), and `prompt.py`
(initialization/E1/E2/M1/M2 instructions). Acknowledgement: **LLM4AD**, Liu et al.,
[A Platform for Algorithm Design with Large Language Model](https://arxiv.org/abs/2412.17287).
Upstream notices are retained in [LICENSE](LICENSE).

Intentional adaptations: ordinary async Slick orchestration replaces LLM4AD's
threaded benchmark infrastructure; all five paper operators are included (current
LLM4AD omits M3); paper-strict dominance replaces upstream's weak comparison on
exact ties; finite attempt budgets replace retry-until-valid loops; mixed objective
directions replace all-maximization; full initialization and the growing parent
pool follow Algorithm 1 rather than upstream's quarter-population initialization
and pending-offspring batches. JSON replaces the paper's brace-delimited prose and
code response, and prompts expose objective measurements. These are explicit
redesigns, not claims of identical prompts, schedules, or reproduced results.

Run offline checks from the repository with
`python -m unittest tests.test_meoh -v` after installing the optional requirements.
They cover numerical selection, all operators, archives, failures, budgets,
template binding, and the real CodeBLEU integration. No paid model calls or
BPP/TSP benchmark reproduction were performed.
