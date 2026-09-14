# Persuasive debate

A problem-agnostic Slick implementation of [Debating with More Persuasive LLMs
Leads to More Truthful Answers](https://arxiv.org/abs/2402.06782) (Khan et al.,
ICML 2024), adapted from the authors' [official
implementation](https://github.com/ucl-dark/llm_debate).

Supply a question, **two candidate answers**, and private evidence as text.
These can be designs, explanations, proposed solutions, or any other pair of
claims. There is no correct-answer label in the algorithm. It selects between
the supplied answers; generating that pair belongs to the caller.

## Use

Install from this directory with `python -m pip install -r requirements.txt`.
The relative Slick path follows this repository's adjacent-checkout convention.
Slick 0.3.0's actual local source was inspected for this implementation.

```python
from pathlib import Path

from slick import prompts
import persuasive_debate
from persuasive_debate import Debater, PersuasiveDebate, Problem

# Configure once at application startup, before concurrent work.
prompts.TEMPLATE_ROOT = Path(persuasive_debate.__file__).resolve().parent / "prompts"

async def solve(question, answers, evidence, expert_provider, judge_provider, logprobs):
    problem = Problem(question, answers, evidence)
    expert = Debater(expert_provider, best_of=16)
    debate = PersuasiveDebate(
        problem, (expert, expert), judge_provider, preference=logprobs,
    )
    return await debate.run(rounds=3)
```

The returned `Result` contains `answer`, its original zero-based `choice`,
canonical `votes`, both answers' `approval` shares, public `turns`, and total
generation/scoring `calls`. Equal vote shares return `choice=None` and
`answer=None`; disagreement remains visible rather than selecting an arbitrary
answer. Vote shares are not calibrated probabilities of correctness.

`Debater(provider)` defaults to best-of-1 without critique. It needs no scoring
callback. Providers are caller-configured Slick providers supporting
`acall(context, *, tools=None, tool_results=None)`; they must support concurrent
requests if shared between debaters. No Session or tools are used. The caller
owns provider temperature, transport retries, context limits, and persistence.
The root is process-global, so different template roots require separate
processes. Imports do not change it, and no current directory is assumed.

### Preference scoring

Slick's portable provider contract exposes text, not token log probabilities.
To preserve the paper's selection rule, supply an async callback:

```python
async def logprobs(context: str) -> dict[str, float]:
    # Implement using your model backend's first generated token distribution.
    # Return actual token log probabilities, not model-written confidence scores.
    return await backend.first_token_logprobs(context)
```

`backend` here is your application's model client. Ask it for one generated
token with top log probabilities (upstream uses the top five). Return exact
token keys `A`/`B` for argument preference and `Y`/`N` for critique helpfulness.
The algorithm selects the highest **raw target-token log probability**, with
`-100` for a missing target, matching upstream `convert_to_prob`. It checks
observed scores for finiteness and nonpositivity and keeps their raw tables.
There is no softmax or generated-confidence substitute. This callback is
required when `best_of > 1` or `critiques > 1`; `critique_preference=` optionally
uses a different callback for critiques, otherwise it reuses `preference`.

Use independently configured judge/preference models to evaluate generalization.
Model choices and sampling temperatures belong to provider setup, not prompts.
The paper uses expert temperature 0.4 for best-of-1, 0.8 for best-of-2 through
16, and 1.0 above 16 for debate; its final judge and preference temperature is 0.

### Protocols and evidence

- `protocol="debate"` is the default. Each side sees the same completed-round
  snapshot, with its own prior arguments first. Both next arguments are
  completed before either is published. Rounds open, challenge, then rebut.
- `protocol="interactive_debate"` adds judge questions between rounds.
- `protocol="consultancy", consultant=0` uses only the first expert; use `1`
  for the second answer. Questions occur between rounds and word budgets
  double. To evaluate consultancy fairly, run both assignments and average.
- `Debater(provider, best_of=4, critiques=8, critic=critic_provider)` runs
  Algorithm 1: select an initial argument, select a helpful critique, sample
  refinements, and select a refinement. The critic defaults to that expert's
  provider and sees the private evidence; neither preference model does.
- `run(votes=3)` collects three final judgments **per answer order**. Every
  run judges both orders and maps votes back to the original answer indices.

Experts and critics receive `Problem.evidence`; judges receive only the
question, answer pair, criteria, and public transcript. Prompts ask participants
to refer to answer text, avoiding A/B references within the argument or judge
questions so the final transcript can be reordered. This is a prompted
convention, not a semantic guarantee for arbitrary model-generated references.

`<thinking>` content is removed and a single nonempty `<argument>` block is
required. All claimed `<quote>`, `<v_quote>`, and `<u_quote>` tags (including
dotted spellings) are reverified. Verification follows upstream normalization:
curly quotes, ASCII punctuation, case, and whitespace. Empty normalized quotes
cannot pass. Other markup is escaped. A verified quote establishes provenance,
not the truth of its interpretation or of the supplied source itself.

For a different evidence mechanism, pass `verify_quote=async_verify`, where
`async_verify(quoted_text) -> bool`. The caller can check trusted documents,
proof certificates, or measurement records. Describe that mechanism in the
problem's evidence/criteria as appropriate. The algorithm never executes
generated code; any such evaluator must provide its own isolation.

The default verifier is designed for prose. Stripping punctuation can collapse
meaningful operators: `x != y` and `x == y` normalize identically. For code,
mathematics, or numeric evidence, inject a verifier that preserves those symbols,
for example an async function returning `quote in evidence` for exact matching.

### Budgets and failures

Defaults are three rounds, word target 100, preferred minimum 70, and maximum
150 per debater argument; consultancy doubles those limits. Each sampling
phase generates exactly `candidates_per_sample * best_of` outputs (default
multiplier three). Length-compliant arguments containing quotes are preferred.
As upstream, an insufficient valid pool is padded with extractable invalid
arguments, truncating long arguments and closing partial quote tags. The word
minimum and quote-presence requirement are therefore preferences on fallback;
the maximum is enforced. Fallbacks are recorded in `debate.failures`.

Malformed public blocks never become transcript content. If fewer than
`best_of` extractable outputs remain, initial generation fails. If all usable
refinements cannot be obtained, the original argument is retained. Refinements
mentioning the critique process are rejected. Malformed critique/judge output,
transport errors, verifier errors, and invalid measured scores propagate.
Siblings finish before a round failure is raised; incomplete rounds are not
published. Raw call records remain in `debate.calls`, including rejected output.
Those records contain private evidence and scratchpads; only `Result.turns`
is the public transcript. Each new run resets records and transcript state.

For best-of-16, no critiques, three rounds, and one final vote per order, the
full budget is 288 expert generations + 96 preference calls + 2 judgments.
Best-of-1 avoids preference scoring. Quote verifier invocations and any internal
backend retries are not included in `Result.calls`.

## Cross-play and Elo

`evaluation.balanced_match(first, second, play)` calls your async
`play(first, second)` twice with reversed assignments and averages the first
player's vote share. Use the same questions in both calls. `play` constructs
the relevant `Debater` pair and aggregates `Result.approval[0]` over your data.
Do not pass correctness accuracy as the match result.

`await evaluation.swiss_tournament(seed_order, play)` uses
`ceil(log2(n))` rounds of score-sorted nearest-neighbor pairing without
rematches. `rounds=` can cap this explicitly. It returns rankings, point scores,
assignment-balanced `Match` records, and byes. The number of model match runs
is O(n log n); the simple upstream-style greedy pairing can produce extra
byes. Draws receive half a point each and byes receive one.

`evaluation.fit_elo(matches, reference="baseline")` uses SciPy BFGS to minimize
the paper's unweighted squared error between observed win rates and
`1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))`. The reference is fixed at
zero. Disconnected match graphs are rejected because relative ratings would
be unidentified. Extreme 0/1 rates can imply unbounded ideal rating gaps;
returned finite ratings are numerical fits, not uncertainty estimates.

## Provenance and differences

Official source inspected at commit
[`f9c71d16e08bac1d30757511c5c7d231f34b4afc`](https://github.com/ucl-dark/llm_debate/tree/f9c71d16e08bac1d30757511c5c7d231f34b4afc):

- `core/agents/debater_quality.py`: sampling, egocentric transcript order,
  rejection, truncation, critique/refinement, and fallback.
- `core/agents/judge_quality.py` and `core/llm_api/base_llm.py`: dummy opponent,
  argument/critique scoring, and missing-token handling.
- `core/rollouts/quality_sim.py`: simultaneous round completion.
- `web/backend/services/parser.py`: quote normalization and strict verification.
- `core/config/experiment/`: expert, judge, and critique prompts and defaults.
- `core/swiss_tournament.py`: assignment reversal and greedy Swiss pairing.

The paper supplies the squared-error Elo objective (Appendix D.5). This is an
intentional generic redesign, not a byte-identical port: prompt wording is
adapted to arbitrary evidence, operations use local Jinja files, no labels or
QuALITY schema are required, and providers/scoring are injected. It retains the
paper's tagged text outputs instead of replacing them with JSON. Empty/forged
quote handling and malformed-output rejection are stricter. Interactive
questions use stable answer text; upstream's human protocol uses names and
reruns swapped debates. Swiss draws are handled symmetrically and the paper's
round count is used instead of upstream's hard-coded early break.

The MIT upstream license is included in [LICENSE](LICENSE). Dataset filtering,
human UI, training/few-shot experiments, bootstrapped intervals, and plots are
outside this inference algorithm. No paper accuracy or truthfulness improvement
has been reproduced or claimed for new domains.

## Checks

From the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_persuasive_debate
../slick/.venv/bin/ruff check persuasive_debate tests/test_persuasive_debate.py
```

Tests use the shared scripted provider for generation. They exercise snapshot
and information boundaries, both answer orders, raw-logprob selection, critique
selection and refinement, fallback and failure records, quote verification,
all templates, and synthetic Swiss/Elo examples. They are offline algorithm
checks, not model-performance measurements.
