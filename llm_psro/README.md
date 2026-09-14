# LLM-PSRO

Implements Algorithm 1 of [Combining Code Generating Large Language Models and
Self-Play to Iteratively Refine Strategies in Games](https://www.ijcai.org/proceedings/2025/1249)
(Bachrach et al., IJCAI 2025). Strategies are source strings; you supply the task,
Slick provider, and async match evaluator. There is no game engine or assumed
strategy language in the optimizer.

```python
from pathlib import Path

import llm_psro
from slick import prompts
from llm_psro import LLMPSRO

# Configure once at application startup; independent of the working directory.
prompts.TEMPLATE_ROOT = Path(llm_psro.__file__).resolve().parent / "prompts"

agent = LLMPSRO(
    task=task_description,  # Rules, source language, entrypoint, API, constraints.
    provider=provider,
    play=play_one_game,
)
population = await agent.run(
    initial_population=initial_sources,
    rounds=5,
    candidates_per_round=5,
    games_per_pair=1000,
    fp_iterations=10000,
    seed=42,
)
last_round = agent.history[-1]
opponent_sources = last_round.mixture.population
opponent_probabilities = last_round.mixture.weights
```

Define `async play_one_game(left: str, right: str, seed: int) -> float` to load
both strategies and run one complete game. Return `+1` for a left-player win,
`0` for a draw, and `-1` for a left-player loss. Create fresh bot/game state on
every call and use the supplied seed for game randomness. The task must specify
everything generation needs, including any engine API or helper source. The
evaluator owns syntax/interface checks, execution isolation, resource limits,
and cleanup. The optimizer never executes generated code.

This is problem-agnostic within **two-player zero-sum competition** with a common
strategy space and interchangeable player roles. Other problems can use a
well-defined pairwise comparison returning win/draw/loss. It is not a general
multiplayer or cooperative Nash solver; the paper leaves those extensions open.

## Algorithm and explicit implementation choices

1. Copy supplied initial sources. With none supplied, generate `population_size`
   sources (default 3) using `initialize.j2`, with at most `initial_attempts`
   attempts (default three per requested source). Initialization checks generated
   structure and nonblank content; actual playability is checked by the evaluator.
2. Each round runs `games_per_pair` games for **every ordered pair**, including
   self-play, and averages the signed outcomes. No previous payoff estimates are
   cached. Convert the raw matrix `S` to `(S - S.T) / 2`, averaging both player
   positions and ensuring a skew-symmetric metagame despite sampling noise.
3. Approximate its equilibrium with simultaneous fictitious play. Both players
   start with one observation per strategy, best respond to the other's empirical
   distribution, and break ties using a seeded RNG. Average the two empirical
   distributions to obtain one population mixture. The numerical helper uses
   the standard library and costs O(iterations × population size).
4. Construct `Mixture.source`: Python selection code containing **all full source
   strings** and their probabilities. `select_strategy(rng)` returns the source
   for the engine to load once before the game. `Mixture.sample(rng)` implements
   that same selection without executing source. There is no action-level mixing.
5. Make exactly `candidates_per_round` independent `respond.j2` generation
   attempts, each containing the full mixed opponent source. Evaluate each valid
   response over `games_per_pair` matches. Reuse the same sampled opponents and
   game seeds across candidates, alternating which player position the candidate
   occupies (an odd game count gives it one extra first-position match).
6. Append the candidate with the highest **win rate**, not mean signed score.
   Draws count as zero wins. Equal win rates keep the first successful candidate;
   duplicate source is allowed. Add the winner even when it has no measured
   advantage over the mixture, as in Algorithm 1.

The defaults `rounds=5`, `population_size=3`, and `games_per_pair=1000` follow the
paper's experiment. The paper does not specify T, FP iterations/initialization,
tie handling, failure policy, or exact prompt text. Those choices, seat averaging,
and common evaluation schedules are explicit implementation decisions here.
The Jinja prompts are new task-agnostic prompts with typed JSON output, rather
than a claim to recover the original Checkers prompts.

`run()` returns the final population in insertion order. `history` retains each
completed round's population snapshot, payoff matrix, mixture, and selected
source. The final recorded mixture precedes the last addition: it does **not**
assign a weight to the final new bot. Algorithm 1 returns the population without
an extra final tournament, and this implementation preserves that budget.

## Failures, budgets, and ownership

`attempts` records generation prompts, raw responses (even rejected JSON), parsed
content, candidate win rate, mean signed score, and errors. Raw source bytes are
preserved; blank generated content is rejected. Invalid JSON/schema, blank
content, `CandidateRejected`, provider errors, and timeouts consume one attempt.
The evaluator should raise `CandidateRejected` for invalid generated strategies;
unexpected exceptions propagate after recording the error. Invalid measured
outcomes also reject a candidate. Any failure while building the incumbent
payoff table aborts the run instead of inventing a score. If initialization or
an entire response round exhausts its budget, `RuntimeError` is raised with the
current population and attempt records retained.

With a valid population of size N, a completed round makes `k * (N*N + T)` game
calls and T provider calls. Failed generation/evaluation can reduce game calls;
`games_played` counts actual calls, including ones that raise. Transport retries
inside a supplied provider are that provider's responsibility. The optimizer has
no hidden generation retries, repair loop, or mutable conversation session.

Each run resets state. Calls are sequential; use one active run per instance.
Slick's template root is process-global, so concurrent algorithms needing
different roots must use separate processes. The seed controls FP tie breaking,
opponent samples, and game seeds, not model sampling or an evaluator that ignores
its seed. Full opponent source and raw logs grow with the population; there is
no silent truncation or summarization.

## Sources and verification

The [official proceedings page](https://www.ijcai.org/proceedings/2025/1249),
[published PDF](https://www.ijcai.org/proceedings/2025/1249.pdf), and
[author publication listing](https://mahnerak.com/) were checked on 2026-09-14.
No official implementation repository was linked there or located by title and
LLM-PSRO searches. This is an independent implementation of the supplied
algorithm; no unrelated PSRO/Checkers repository is presented as official code.

From this directory, install the adjacent Slick checkout with
`python -m pip install -r requirements.txt`. From the repository root, run:

```sh
python -m unittest tests.test_llm_psro
```

The offline tests use the repository's shared `tests.providers.ScriptedProvider`.
They check fictitious-play exploitability on known games, exact budgets,
population growth, prompt contents, role balancing, win-rate selection,
failure handling, and raw-response retention. These are algorithm/interface
checks, not a reproduction of the paper's CodeLlama/Checkers results. No paid
model call or real-game performance claim is included.
