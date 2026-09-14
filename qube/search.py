"""QUBE equations 1–2 and island evolution; scores are always maximized."""

import io
import math
import random
import tokenize
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Program:
    code: str
    signature: tuple[float, ...]
    score: float

    def __post_init__(self):
        if (
            not self.code
            or not self.signature
            or not all(math.isfinite(x) for x in (*self.signature, self.score))
        ):
            raise ValueError("program requires code and finite evaluation results")


@dataclass
class Cluster:
    programs: list[Program]
    visits: int = 0
    offspring_sum: float = 0.0
    offspring_count: int = 0

    def uiq(self, t, k):
        # The paper leaves cold starts unspecified: force exploration when k > 0.
        if self.visits == 0 and k > 0:
            return math.inf
        quality = (
            self.offspring_sum / self.offspring_count
            if self.offspring_count
            else self.programs[0].score
        )
        return quality + k * math.sqrt(math.log(max(1, t)) / max(1, self.visits))

    def sample(self, rng, temperature):
        lengths = [len(p.code) for p in self.programs]
        # Subtracting the largest logit cancels max(length) in the paper's formula.
        shortest = min(lengths)
        weights = [math.exp(-(n - shortest) / (shortest + 1e-6) / temperature) for n in lengths]
        return rng.choices(self.programs, weights=weights, k=1)[0]


@dataclass
class Island:
    clusters: dict[tuple[float, ...], Cluster] = field(default_factory=dict)

    def add(self, program):
        cluster = self.clusters.get(program.signature)
        if cluster is None:
            self.clusters[program.signature] = Cluster([program])
        elif program.code not in {p.code for p in cluster.programs}:
            cluster.programs.append(program)

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


class Database:
    def __init__(
        self,
        initial,
        *,
        islands=10,
        k=0.0001,
        reset_interval=32768,
        temperature=1.0,
        seed=0,
    ):
        if islands < 1 or reset_interval < 0 or k < 0 or not math.isfinite(k):
            raise ValueError("invalid island count, reset interval, or k")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("program temperature must be positive and finite")
        self.islands = [Island() for _ in range(islands)]
        for island in self.islands:
            island.add(initial)
        self.k, self.temperature, self.reset_interval = k, temperature, reset_interval
        self.rng = random.Random(seed)
        self.generated = self.accepted = 0
        self.best = initial

    def select(self, batch_size=1):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        index = self.rng.randrange(len(self.islands))
        parents, clusters = self.islands[index].select(
            self.rng, self.generated + 1, self.k, self.temperature
        )
        # One parent use per attempted child, even if both examples share a cluster.
        # Reserve pending uses now so concurrent samplers see exploration already due.
        for cluster in clusters:
            cluster.visits += batch_size
        return index, parents, clusters

    def record(self, index, clusters, program):
        self.generated += 1
        if program is None:
            return
        self.accepted += 1
        for cluster in clusters:
            cluster.offspring_sum += program.score
            cluster.offspring_count += 1
        self.islands[index].add(program)
        if program.score > self.best.score:
            self.best = program

    def reset(self):
        if len(self.islands) < 2:
            return []
        best = [island.ranked(self.rng, self.generated + 1, self.k)[0] for island in self.islands]
        order = list(range(len(self.islands)))
        self.rng.shuffle(order)
        order.sort(key=lambda i: best[i].uiq(self.generated + 1, self.k))
        cut = len(order) // 2
        events = []
        for index in order[:cut]:
            donor = self.rng.choice(order[cut:])
            program = self.rng.choice(best[donor].programs)
            self.islands[index] = Island()
            self.islands[index].add(program)
            events.append({"island": index, "donor": donor})
        return events


def tokens(code):
    return [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(code).readline)
        if token.type not in (tokenize.ENCODING, tokenize.ENDMARKER)
    ]


def edit_distance(a, b):
    # ponytail: quadratic token edit distance; use a compiled implementation if profiling warrants it.
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        next_row = [i]
        for j, y in enumerate(b, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = next_row
    return row[-1]


class Recent:
    def __init__(self, size=500):
        self.samples = deque(maxlen=size)

    def add(self, program, parents):
        if program is None:
            self.samples.append((None, None))
            return
        child = tokens(program.code)
        change = min(edit_distance(child, tokens(p.code)) for p in parents) / max(1, len(child))
        self.samples.append((program.score, change))

    def metrics(self):
        valid = [(s, c) for s, c in self.samples if s is not None]
        return {
            "recent_best_score": max((s for s, _ in valid), default=None),
            "recent_proportion_of_change": (
                sum(c for _, c in valid) / len(valid) if valid else None
            ),
        }
