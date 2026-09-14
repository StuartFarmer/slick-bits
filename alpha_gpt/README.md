# Alpha-GPT

Problem-agnostic implementation of the AlphaBot algorithm in
[Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment](https://aclanthology.org/2025.emnlp-demos.14/).
It translates ideas into candidates, enhances them with genetic search, interprets
measured results, and uses feedback to guide subsequent rounds. Candidates are text;
your task and callbacks define whether they represent formulas, prompts, rules,
schedules, or another artifact.

## Use

Requires the adjacent Slick checkout and its dependencies. From this repository's
root, the existing `optimizer/.venv/bin/python` environment provides these.
Configure the process-global template root once at application startup:

```python
from pathlib import Path
from slick import prompts
import alpha_gpt

prompts.TEMPLATE_ROOT = Path(alpha_gpt.__file__).resolve().parent / "prompts"

async def optimize(task, provider, evaluate, mutate, crossover, retrieve=None, feedback=None):
    agent = alpha_gpt.AlphaGPT(
        task, provider, evaluate,
        mutate=mutate, crossover=crossover, retrieve=retrieve,
    )
    return await agent.run("Your initial idea", rounds=3, feedback=feedback)
```

The provider is an ordinary Slick provider supplied by your application. The core
does not construct model clients or run generated code. Independent generation
calls use exactly one `provider=` and explicit history; no Session is needed.
Use a fresh agent per run. Imports do not modify Slick's template root. Different
template roots require separate processes when running simultaneously.

Callback contracts:

| Callback | Contract |
| --- | --- |
| `async evaluate(content)` | Return `Evaluation(score, feedback="", metrics={}, qualified=True)`. The score determines selection; higher is better unless `maximize=False`. |
| `async mutate(content, directions, rng)` | Return one domain-valid candidate string using your grammar and the supplied `random.Random`. |
| `async crossover(left, right, directions, rng)` | Return one candidate combining the two parents. |
| `async retrieve(query)` | Return a relevance-ordered sequence of `Document(id, text)` for the requested stage and exact hierarchy path. Optional for interactive mode. |
| `async feedback(round)` | Return guidance for the next round, or `None` to stop. Optional; interactive mode stops after one review without it. |
| `redundant(candidate, selected)` | Optional synchronous predicate for behavioral equivalence or domain-specific correlation filtering. Both arguments are measured `Individual`s. |

`task` must describe the candidate interface, allowed fields/operators, constraints,
and objective. `evaluate` owns domain validation, data access, qualification, and
any execution isolation. Raise `CandidateRejected` for an expected invalid
candidate; unrelated exceptions propagate. Variation callbacks can also raise
`CandidateRejected` to skip an invalid offspring. The backend receives the model's
search directions; your operators decide how those directions affect the grammar.

## Complete non-financial example

This deliberately small task searches integer text for the value nearest 42. It
demonstrates the interface, not performance on a research benchmark. Call
`await find_integer(provider)` with your configured Slick provider.

```python
async def find_integer(provider):
    async def evaluate(content):
        try:
            value = int(content)
        except ValueError as error:
            raise alpha_gpt.CandidateRejected("Return only an integer") from error
        return alpha_gpt.Evaluation(abs(value - 42), feedback="Absolute error from 42")

    async def mutate(content, directions, rng):
        return str(int(content) + rng.choice([-4, -1, 1, 4]))

    async def crossover(left, right, directions, rng):
        return str((int(left) + int(right)) // 2)

    agent = alpha_gpt.AlphaGPT(
        "Return a decimal integer minimizing absolute error from 42.",
        provider, evaluate, mutate=mutate, crossover=crossover,
        maximize=False, seed_count=4, population_size=4, generations=5,
    )
    return await agent.run("Start with integers on both sides of 42")
```

Here `directions` is intentionally unused by the toy operators; production
operators can interpret it or accept a domain-specific policy. No artificial
provider or candidate execution engine is bundled in the implementation.

## Algorithm and modes

1. **Ideation:** retrieve successful examples, relevant field specifications and
   literature; polish the user's idea and feedback into a structured idea plus
   search directions and mutation/crossover probabilities.
2. **Implementation:** generate `seed_count` candidates with explanations. Evaluate
   each and attempt up to `repair_attempts` model repairs for domain rejections.
   Run `generations` genetic search generations. Each generation draws parents
   from its starting population using two-member tournaments with replacement.
   Try `population_size` offspring: crossover first, mutation second, independently
   sampled using the polished probabilities. Keep the best unique, qualified
   candidates from incumbents and offspring, optionally pruning redundancy.
3. **Review:** the analyst receives seed scores, selected results, metrics,
   evaluator feedback, and failures. Its summary and next direction are returned
   to the researcher. Human feedback changes the next round's ideation and search
   configuration; it does not change the caller's objective or evaluation data.

For **interactive mode**, pass an idea to `run`. For **autonomous mode**, call
`await agent.run(rounds=3)` with a hierarchical retrieval callback. Each round
performs RAG#0–3: retrieve successes, choose a high-level category, choose a scoped
subcategory, retrieve its field descriptions, then discover an idea. Only the
retrieved category IDs can be selected. Current measured candidates and the
previous review also inform discovery. The normal polish/search/review loop follows.

Retrieval requests use these scopes:

| Stage | Path |
| --- | --- |
| `successes`, `categories`, `literature` | `()` |
| `subcategories` | `(category_id,)` |
| `fields` in autonomous mode | `(category_id, subcategory_id)` |
| `fields` in interactive mode | `()` — relevant specifications for the supplied idea |

`retrieval_limit` caps documents per request (default 8). The retriever owns
relevance, chunk length, and indexing. Keep field/literature documents appropriately
chunked; a document count limit is not a token limit. Empty category lists cannot
support navigation and cause selection to fail rather than inventing fields.

## Official sources and Faiss

The [paper's arXiv version](https://arxiv.org/html/2308.00016v2) and
[ACL publication page](https://aclanthology.org/2025.emnlp-demos.14/) were checked on
2026-09-14. No verifiable official Alpha-GPT implementation link was found there
or through author/title searches. Similarly named third-party repositories were
not treated as author implementations.

The paper explicitly links the official
[facebookresearch/faiss repository](https://github.com/facebookresearch/faiss).
The optional `alpha_gpt.retrieval.FaissRetriever` uses that library directly,
following its [official index/add/search API](https://github.com/facebookresearch/faiss/wiki/Getting-started).
It builds an `IndexFlatL2` for each supplied hierarchy bucket, so retrieval searches
only the selected scope. Install its optional dependency with:

```sh
python -m pip install -r alpha_gpt/requirements-retrieval.txt
```

```python
from alpha_gpt import Document
from alpha_gpt.retrieval import FaissRetriever

# All vectors must use the same embedding model, dimensions and normalization.
# Your async embed(text) returns one embedding vector.
def make_retriever(embed):
    return FaissRetriever({
        ("categories", ()): ([Document("rules", "Decision rules")], [[1.0, 0.0]]),
        ("subcategories", ("rules",)): (
            [Document("thresholds", "Numeric thresholds")], [[0.9, 0.1]],
        ),
        ("fields", ("rules", "thresholds")): (
            [Document("duration", "Elapsed duration in seconds")], [[0.8, 0.2]],
        ),
    }, embed)
```

The two-dimensional vectors above illustrate the bucket layout; use real document
embeddings in an application. Add `successes`, `literature`, and unscoped `fields`
buckets as appropriate. Missing buckets return no documents. This is a static
index for caller-owned knowledge; new run results remain in the agent's archive
and population, and persistence/index updates belong to the application.

## Budgets, evidence, and fidelity

`max_calls` counts every model attempt, including repairs and reviews.
`max_evaluations` counts actual evaluator invocations. Exact candidate text is
cached across all rounds, including rejections; scores and qualification must
therefore stay stable for a run. Only finite scores and metrics enter selection.
There is no retry-until-full loop: duplicate and invalid offspring consume one
of the fixed offspring attempts and may leave the population underfilled.

`CandidateRejected`, blank content, nonfinite measurements, and failed
qualification produce explicit failure records. Invalid JSON/schema, incorrect
seed counts, unknown retrieved IDs, provider failures, unexpected callback errors,
and cancellation propagate. JSON/transport retries are not hidden inside the
agent. Raw prompts and responses are retained in `agent.calls` before parsing,
including error records; `archive`, `retrievals`, `rounds`, and `generations` retain
partial progress. Budget exhaustion returns a partial `Result`, which may contain
evaluated candidates without a completed analyst review. `rounds=0` does no work.

The paper specifies an architecture rather than a complete executable GP
algorithm. This implementation preserves its three-stage loop, knowledge
compilation, thought decompilation, hierarchical RAG, interactive refinement, and
seeded genetic enhancement. Tournament size, elitist replacement, exact-text
caching, finite budgets, repair policy, and structured generic prompts are explicit
implementation choices. Domain genetic operators are injected rather than replaced
by LLM mutation; domain syntax and semantic validity remain the evaluator's job.

No WebUI, market dataset, trading backtester, deployment infrastructure, GPU layer,
or proprietary alpha library is included. Final held-out evaluation belongs outside
the search loop; do not feed held-out results into `Evaluation.metrics`, retrieved
memory, or feedback if you intend to reserve them for a final unbiased assessment.
The offline checks establish algorithm behavior, not reproduction of the paper's
competition rankings or investment results.

Run checks from the repository root:

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_alpha_gpt
../slick/.venv/bin/ruff check alpha_gpt tests/test_alpha_gpt.py
../slick/.venv/bin/ruff format --check alpha_gpt tests/test_alpha_gpt.py
```

The Faiss integration test runs when `faiss-cpu` is installed; other tests use only
the shared scripted provider in `tests/providers.py` and make no paid model calls.
