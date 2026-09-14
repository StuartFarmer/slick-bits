"""Evolve caller-scored text with quality-uncertainty clusters and islands."""

import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from slick import Provider, Session, prompt


@dataclass(frozen=True)
class Evaluation:
    score: float
    signature: tuple[float, ...]

    def __post_init__(self):
        if not self.signature:
            raise ValueError("signature must not be empty")
        if not all(math.isfinite(value) for value in (self.score, *self.signature)):
            raise ValueError("score and signature must be finite")


@dataclass(frozen=True)
class Candidate:
    candidate: str
    signature: tuple[float, ...]
    score: float


@dataclass
class Sample:
    candidate: str
    island: int
    parents: tuple[str, str]
    evaluation: Evaluation | None = None
    error: str = ""


@dataclass
class Result:
    best: Candidate
    samples: list[Sample] = field(default_factory=list)
    resets: list[dict] = field(default_factory=list)

    @property
    def accepted(self) -> int:
        return sum(sample.evaluation is not None for sample in self.samples)


@dataclass
class Cluster:
    candidates: list[Candidate]
    visits: int = 0
    offspring_mean: float = 0.0
    offspring_count: int = 0

    def uiq(self, t, k):
        # The paper leaves cold starts unspecified: force exploration when k > 0.
        if self.visits == 0 and k > 0:
            return math.inf
        quality = self.offspring_mean if self.offspring_count else self.candidates[0].score
        return quality + k * math.sqrt(math.log(max(1, t)) / max(1, self.visits))

    def sample(self, rng, temperature):
        lengths = [len(p.candidate) for p in self.candidates]
        # Subtracting the largest logit cancels max(length) in the paper's formula.
        shortest = min(lengths)
        weights = [math.exp(-(n - shortest) / (shortest + 1e-6) / temperature) for n in lengths]
        return rng.choices(self.candidates, weights=weights, k=1)[0]


@dataclass
class Island:
    clusters: dict[tuple[float, ...], Cluster] = field(default_factory=dict)

    def add(self, candidate):
        cluster = self.clusters.get(candidate.signature)
        if cluster is None:
            self.clusters[candidate.signature] = Cluster([candidate])
        elif candidate.candidate not in {p.candidate for p in cluster.candidates}:
            cluster.candidates.append(candidate)

    def ranked(self, rng, t, k):
        clusters = list(self.clusters.values())
        rng.shuffle(clusters)  # Avoid insertion-order bias for equal UIQ (incl. inf).
        return sorted(clusters, key=lambda c: c.uiq(t, k), reverse=True)

    def select(self, rng, t, k, temperature):
        clusters = self.ranked(rng, t, k)[:2]
        parents = [c.sample(rng, temperature) for c in clusters]
        if len(parents) == 1:
            parents.append(clusters[0].sample(rng, temperature))
        # Lower-scoring example first, as in FunSearch's few-shot prompt.
        return sorted(parents, key=lambda p: p.score), clusters


class QUBE:
    """Maximize scores of textual candidates grouped by behavior signature.

    The seed is evaluated once before `samples` offspring attempts. Rejected
    candidates consume attempts but do not update offspring quality. Provider
    and seed evaluation failures abort; cancellation always propagates.
    Instances own one run. The caller configures templates and any Session.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        *,
        seed_candidate: str,
        samples: int = 100,
        islands: int = 10,
        k: float = 0.0001,
        reset_interval: int = 32768,
        temperature: float = 1.0,
        seed: int = 0,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.seed_candidate, self.samples = seed_candidate, samples
        self.islands = [Island() for _ in range(islands)]
        self.k, self.temperature, self.reset_interval = k, temperature, reset_interval
        self.rng = random.Random(seed)
        self.result: Result | None = None

    @prompt(template="generate.j2")
    async def generate(self, parent1: Candidate, parent2: Candidate, *, generated: str) -> str:
        """Generate one candidate from ordered quality-uncertainty parents."""
        return generated

    async def score(self, candidate: str) -> Evaluation:
        if not candidate.strip():
            raise ValueError("candidate must be nonblank text")
        return await self.evaluate(candidate)

    def reset(self, generated: int) -> list[dict]:
        """Replace the weaker half with samples from stronger island clusters."""
        if len(self.islands) < 2:
            return []
        best = [island.ranked(self.rng, generated + 1, self.k)[0] for island in self.islands]
        order = list(range(len(self.islands)))
        self.rng.shuffle(order)
        order.sort(key=lambda i: best[i].uiq(generated + 1, self.k))
        cut = len(order) // 2
        events = []
        for index in order[:cut]:
            donor = self.rng.choice(order[cut:])
            candidate = self.rng.choice(best[donor].candidates)
            self.islands[index] = Island()
            self.islands[index].add(candidate)
            events.append({"sample": generated, "island": index, "donor": donor})
        return events

    async def initialize(self) -> None:
        evaluation = await self.score(self.seed_candidate)
        self.signature_size = len(evaluation.signature)
        candidate = Candidate(self.seed_candidate, evaluation.signature, evaluation.score)
        self.result = Result(best=candidate)
        for island in self.islands:
            island.add(candidate)

    def select_parents(self, step: int) -> tuple[int, list[Candidate], list[Cluster]]:
        index = self.rng.randrange(len(self.islands))
        parents, clusters = self.islands[index].select(self.rng, step, self.k, self.temperature)
        # A single cluster supplying both examples receives one visit.
        for cluster in clusters:
            cluster.visits += 1
        return index, parents, clusters

    async def assess_sample(self, text: str, index: int, parents: list[Candidate]) -> Sample:
        sample = Sample(text, index, (parents[0].candidate, parents[1].candidate))
        self.result.samples.append(sample)
        try:
            evaluation = await self.score(text)
            if len(evaluation.signature) != self.signature_size:
                raise ValueError("signature dimension must match the seed")
            sample.evaluation = evaluation
        except ValueError as exc:
            sample.error = f"{type(exc).__name__}: {exc}"
        return sample

    def record(self, sample: Sample, clusters: list[Cluster]) -> None:
        if sample.evaluation is None:
            return
        evaluation = sample.evaluation
        candidate = Candidate(sample.candidate, evaluation.signature, evaluation.score)
        for cluster in clusters:
            cluster.offspring_count += 1
            count = cluster.offspring_count
            # Weight before adding: neither a sum nor a score difference must overflow.
            cluster.offspring_mean = (
                cluster.offspring_mean * ((count - 1) / count) + candidate.score / count
            )
        self.islands[sample.island].add(candidate)
        if candidate.score > self.result.best.score:
            self.result.best = candidate

    async def run(self, *, session: Session | None = None) -> Result:
        """Initialize a fresh instance, then generate, assess, and record each offspring."""
        execution = {"session": session} if session is not None else {"provider": self.provider}
        await self.initialize()
        for step in range(1, self.samples + 1):
            index, parents, clusters = self.select_parents(step)
            text = await self.generate(*parents, **execution)
            sample = await self.assess_sample(text, index, parents)
            self.record(sample, clusters)
            if self.reset_interval and step % self.reset_interval == 0:
                self.result.resets.extend(self.reset(step))
        return self.result
