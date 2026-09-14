# Interactive evolution with LLM operators

Problem-agnostic Slick implementation of **ChatGPT and Other Large Language
Models as Evolutionary Engines for Online Interactive Collaborative Game Design**
by Pier Luca Lanzi and Daniele Loiacono (GECCO 2023,
[arXiv:2303.02155v2](https://arxiv.org/html/2303.02155v2),
[DOI](https://doi.org/10.1145/3583131.3590351)). Candidates are arbitrary text.
The brief supplies the domain, constraints, response structure, and length limits;
the caller supplies mutation topics, a Slick provider, and async feedback collection.

## Use

Use the existing Slick environment. Configure the local template root once at
application startup; importing this package does not change global configuration.
This example provides a simple single-person terminal evaluator outside the engine:

```python
import asyncio
from pathlib import Path

import interactive_evolution
from interactive_evolution import Evaluation, InteractiveEvolution
from slick import prompts

prompts.TEMPLATE_ROOT = Path(interactive_evolution.__file__).resolve().parent / "prompts"


async def collect_votes(population):
    votes = []
    for candidate in population:
        if not candidate.ratings:
            print(f"\nCandidate {candidate.id}\n{candidate.content}")
            rating = int(await asyncio.to_thread(input, "Negative -1 / neutral 0 / positive 1: "))
            votes.append(Evaluation(candidate.id, rating))
    return votes


async def evolve(provider, brief_path, mutation_topics, seeds=()):
    engine = InteractiveEvolution(
        task=Path(brief_path).read_text(),
        provider=provider,
        evaluate=collect_votes,
        mutation_topics=mutation_topics,
    )
    population = await engine.run(
        initial=seeds,
        population_size=10,
        iterations=30,
        new_evaluations=1,  # One reviewer; use 25 for the paper's collaborative setting.
        seed=42,
    )
    return population, engine
```

Call `await evolve(your_provider, "brief.md", ["resources", "structure", "interaction"])`
with topics meaningful for your problem. A brief could describe a product concept,
lesson plan, protocol, story, or algorithm. Initial seeds may be supplied as text;
the model fills remaining slots. Supplying all seeds makes initialization model-free.
The terminal example assumes valid integer input; an application's collector owns
its input handling. The engine rejects ratings outside the three-value scale.

For collaborative use, replace `collect_votes` with
`async evaluate(population: tuple[Individual, ...]) -> Sequence[Evaluation]`.
It receives immutable snapshots of the **current active** population. Publish new
IDs once and await new votes, returning only votes not previously delivered. It may
return votes for older published IDs even after their removal from the population.
Participant deduplication, changed-vote reconciliation, anonymity, hiding others'
votes before submission, publication timing, and persistence belong to this adapter.
An automated evaluator can also return these ratings, but replaces the paper's
human feedback with a different evaluation procedure. Generated code is never
executed by the engine; any execution and its isolation belong to the caller.

## Algorithm and explicit choices

1. Build a population from supplied seeds and independently generated candidates.
2. Wait for `new_evaluations` new votes **and** at least `min_evaluations` votes
   on every active individual. Defaults are 25 and 1, respectively. Initial votes
   count toward the first trigger. Late votes on retired individuals count toward
   the trigger and update their archive records, but cannot restore them to the population.
3. Select two parents through independent tournaments of size two. Each tournament
   samples distinct contestants without replacement; winners can coincide across
   tournaments. Maximize mean vote, with negative/neutral/positive mapped to -1/0/+1.
   Tournament ties go to the first randomly sampled contestant.
4. With probability 0.7, ask the model to recombine the parents into **one** child.
   Otherwise copy the first parent's text. Always mutate the resulting text using
   one uniformly sampled, caller-supplied high-level topic. The intermediate
   crossover text is not published or evaluated.
5. Insert the mutated child and evict the worst **incumbent**, preserving population
   size. Equal-fitness incumbents are evicted oldest first. The unevaluated child
   receives feedback before it can participate in another evolutionary iteration.
6. Repeat for the requested number of iterations, then collect the final child's
   minimum feedback and return active individuals in descending fitness order.
   Final collection does not require another activation quota. Zero iterations
   returns an initialized, evaluated population without evolutionary calls.

The paper specifies the three-value scale, steady-state loop, size-two tournaments,
0.7 crossover probability, mandatory mutation, and experimental settings N=10,
30 iterations, and 25 new evaluations per activation (5 for the game jam).
It does **not** specify the fitness aggregation formula, tournament tie/sampling
details, a numeric per-individual feedback minimum, or how an unscored child is
ranked during deletion. Mean vote, the coverage gate, and protected insertion above
are explicit implementation choices. In particular, this is not elitist survival
over an already-evaluated parent-plus-child pool: a poorly rated child still gets
its initial chance and can replace a better incumbent before its score is known.

Votes received in a batch are applied together before selection. The activation
counter resets after each evolutionary step; surplus votes in an oversized batch
are not carried into future activations. This defines the behavior at the injected
batch boundary. There are no background polling tasks or automatic extra iterations.

The three local Jinja templates intentionally generalize Section 4's game-specific
prompts. They use the same brief for initialization, crossover, and mutation, with
separate methods and no template control branches. Responses remain free-form text;
nonblank checks preserve the exact returned whitespace. Semantic coherence,
constraint compliance, novelty, and changing only the requested mutation topic
remain model instructions, not guarantees proved by those checks.

## State, budgets, and failure policy

- `population` exposes active individuals. `archive` retains all inserted individuals,
  including retired ones, their ratings, parent IDs, and mutation topics.
  `published` records IDs offered to the feedback collector; actual successful
  publication remains the collector's responsibility.
- `evaluations` stores accepted vote events; `history` stores immutable population
  snapshots before each evolutionary step and after final feedback.
  `completed_iterations` counts inserted offspring, including one awaiting final feedback.
- `attempts` records every model operation, raw response when available, status,
  and error. Invalid blank output and provider failures propagate without retries.
  Mutation failure after crossover preserves all incumbents and both call records.
  Initialization failure retains its partial population. Feedback errors propagate;
  an invalid vote batch is rejected atomically. Empty feedback raises `RuntimeError`
  to prevent an unproductive busy loop; wait inside the collector or cancel the run.
- With N individuals, S seeds, and K successful iterations, model calls total
  `N - S + K + C`, where C is the number of crossover activations, between 0 and K.
  The engine owns no transport retries. Feedback collection can wait indefinitely;
  the caller owns deadlines/cancellation. For an unbounded online session, choose
  the iteration budget and enforce its lifetime at the application boundary.
- Each run resets state and RNG; use one active run per engine. Seeded randomness
  controls tournaments, crossover decisions, and mutation topics, not model output
  or human votes. Calls are sequential with exactly one explicit `provider=` and
  no conversation history. Slick's template root is process-global: configure it
  once and use separate processes for applications needing different roots.

Caller configuration is trusted: use N >= 2, at most N seeds, positive feedback
thresholds, nonempty mutation topics, nonnegative iterations, and a crossover
probability between zero and one. No framework, CLI dependency, database, Telegram
client, benchmark, or provider implementation is bundled.

## Source provenance and verification

The supplied paper and the authors' [arXiv v2 text](https://arxiv.org/html/2303.02155v2)
were inspected on 2026-09-14, including Sections 3.1, 4.1–4.3, and 5. The paper
does not link an official source repository. Searches by exact title, arXiv ID,
and authors did not identify one. This is a paper-derived implementation, not a
port of verified official code; no third-party repository is presented as official
or required at runtime.

The actual adjacent Slick checkout declares version 0.3.0. Its decorator was
inspected for method ownership (`instance` in Jinja), postprocessing, and provider
execution. Plain text deliberately omits `output_type`: in this checkout even
`output_type=str` appends JSON response instructions. No structured generation
or JSON parsing is needed for the paper's free-form individuals.

From the repository root, in an environment containing Slick:

```sh
python -m unittest tests.test_interactive_evolution
```

Checks use the shared `tests/providers.py` scripted provider and cover vote gates,
late feedback, crossover followed by mutation, mutation without crossover,
protected insertion, archival lineage, failure accounting, atomic invalid batches,
state reset, and template rendering from another working directory. These are
offline algorithm/interface checks, not a reproduction of human-subject results
or evidence of improved design quality with a live model.
