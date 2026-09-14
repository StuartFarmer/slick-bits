"""APEX: sentence mutation with immediate beam updates and history-guided search."""

import math
import random
import re
from dataclasses import dataclass, replace

import numpy as np
from slick import prompt


@dataclass(frozen=True)
class Config:
    iterations: int = 50
    beam_size: int = 4
    alpha: float = 0.05
    regularization: float = 1.0
    random_probability: float = 0.5
    history_limit: int = 4
    history_distance: float = 0.5
    guided_mutation: bool = True
    seed: int = 0

    def __post_init__(self):
        for name, minimum in (("iterations", 0), ("beam_size", 1), ("history_limit", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("alpha", "regularization", "random_probability", "history_distance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.regularization == 0 or self.random_probability > 1:
            raise ValueError("regularization must be positive; probability must be <= 1")


LABEL = re.compile(r"(?:Q:|A:|Question[.:]|Answer[.:]|\([A-Za-z0-9]+\))\s*")


@dataclass(frozen=True)
class Document:
    parts: tuple[str, ...]
    mutable: tuple[int, ...]

    def __post_init__(self):
        if not self.mutable or len(set(self.mutable)) != len(self.mutable):
            raise ValueError("prompt needs distinct mutable fragment indices")
        if not all(isinstance(p, str) for p in self.parts):
            raise ValueError("prompt fragments must be strings")
        for i in self.mutable:
            if type(i) is not int or not 0 <= i < len(self.parts) or not self.parts[i].strip():
                raise ValueError("mutable fragments must be nonempty and in range")

    @property
    def text(self):
        return "".join(self.parts)

    def replace(self, index, sentence):
        if index not in self.mutable:
            raise ValueError("cannot replace an immutable fragment")
        parts = list(self.parts)
        parts[index] = sentence
        return replace(self, parts=tuple(parts))

    @classmethod
    def from_json(cls, fragments):
        if not isinstance(fragments, list) or not all(
            isinstance(p, dict)
            and isinstance(p.get("text"), str)
            and type(p.get("mutable")) is bool
            for p in fragments
        ):
            raise ValueError("expected a list of {text: string, mutable: boolean} fragments")
        return cls(
            tuple(p["text"] for p in fragments),
            tuple(i for i, p in enumerate(fragments) if p["mutable"]),
        )

    @classmethod
    def split(cls, text):
        parts, mutable = [], []
        # ponytail: punctuation heuristic misses abbreviations/complex markup;
        # use explicit fragment JSON when exact sentence boundaries matter.
        for line in text.splitlines(keepends=True):
            leading = re.match(r"\s*", line).group()
            parts.append(leading)
            rest = line[len(leading) :]
            label = LABEL.match(rest)
            if label:
                parts.append(label.group())
                rest = rest[label.end() :]
            for chunk in re.split(r'(?<=[.!?])([ \t]+)(?=[A-Z"\u201c])', rest):
                stripped = chunk.rstrip()
                if stripped and not stripped.isspace():
                    mutable.append(len(parts))
                    parts.append(stripped)
                parts.append(chunk[len(stripped) :])
        return cls(tuple(parts), tuple(mutable))


class Embeddings:
    """Cache unit vectors by sentence text; reject invalid encoder outputs."""

    def __init__(self, encode):
        self.encode, self.cache, self.dimension = encode, {}, None

    def __call__(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            vectors = np.asarray(self.encode(missing), dtype=float)
            if (
                vectors.ndim != 2
                or vectors.shape[0] != len(missing)
                or vectors.shape[1] == 0
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("encoder must return a finite (sentences, dimensions) matrix")
            norms = np.linalg.norm(vectors, axis=1)
            if not np.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("encoder returned a zero or overflowing vector")
            if self.dimension is not None and vectors.shape[1] != self.dimension:
                raise ValueError("encoder dimensions changed")
            self.dimension = vectors.shape[1]
            self.cache.update(zip(missing, vectors / norms[:, None]))
        return np.array([self.cache[t] for t in texts])


class LinUCB:
    def __init__(self, regularization=1.0, alpha=0.05):
        self.regularization, self.alpha = regularization, alpha
        self.features, self.rewards = [], []

    def update(self, feature, reward):
        self.features.append(feature.copy())
        self.rewards.append(reward)

    def values(self, x):
        # Dual ridge / Woodbury: same equations, solve T-by-T instead of d-by-d.
        variance = np.sum(x * x, axis=1) / self.regularization
        mean = np.zeros(len(x))
        if self.features:
            h = np.array(self.features)
            cross = x @ h.T
            kernel = h @ h.T + self.regularization * np.eye(len(h))
            solved = np.linalg.solve(kernel, np.column_stack((self.rewards, cross.T)))
            mean = cross @ solved[:, 0]
            variance -= np.sum(cross * solved[:, 1:].T, axis=1) / self.regularization
        return mean + self.alpha * np.sqrt(np.maximum(variance, 0))


def retrieve(feature, history, embeddings, limit, distance):
    """Nearest nonzero-reward edits; negative edits are presented in reverse."""
    entries = [entry for entry in history if entry["reward"] != 0]
    if not entries or not limit:
        return []
    vectors = embeddings([entry["before"] for entry in entries])
    distances = np.linalg.norm(vectors - feature, axis=1)
    chosen = sorted(
        (i for i, d in enumerate(distances) if d < distance), key=lambda i: distances[i]
    )[:limit]
    return [
        (entries[i]["before"], entries[i]["after"])
        if entries[i]["reward"] > 0
        else (entries[i]["after"], entries[i]["before"])
        for i in chosen
    ]


@prompt(template="mutate.j2")
async def mutate(sentence: str, examples: list, guided: bool, *, generated: str) -> str:
    """Generate a sentence rephrasing; the optimizer validates and logs the raw text."""
    return generated


async def optimize(document, evaluate, provider, encode, config=Config(), on_step=None):
    """Maximize a training-only async score. No held-out data enters this function."""
    rng = random.Random(config.seed)
    embeddings = Embeddings(encode)
    bandit = LinUCB(config.regularization, config.alpha)
    cache, history = {}, []

    async def score(text):
        if text not in cache:
            value = float(await evaluate(text))
            if not math.isfinite(value):
                raise ValueError("evaluator returned a non-finite score")
            cache[text] = value
        return cache[text]

    initial_score = await score(document.text)
    beam = [(document, initial_score)]
    if on_step:
        on_step(dict(iteration=0, prompt=document.text, score=initial_score, evaluations=1))
    for iteration in range(1, config.iterations + 1):
        parent, parent_score = rng.choice(beam)
        vectors = embeddings([parent.parts[i] for i in parent.mutable])
        if rng.random() < config.random_probability:
            arm = rng.randrange(len(vectors))
            selection = "random"
        else:
            values = bandit.values(vectors)
            arm = int(
                rng.choice(
                    np.flatnonzero(
                        np.isclose(values, values.max(), atol=1e-12, rtol=1e-12)
                    ).tolist()
                )
            )
            selection = "linucb"
        index, feature = parent.mutable[arm], vectors[arm]
        before = parent.parts[index]
        examples = (
            retrieve(feature, history, embeddings, config.history_limit, config.history_distance)
            if config.guided_mutation
            else []
        )
        after = (await mutate(before, examples, config.guided_mutation, provider=provider)).strip()
        valid = bool(after) and len(after.splitlines()) == 1 and "```" not in after
        valid = valid and not re.search(r"</?(?:original|rephrased)\b", after, re.IGNORECASE)
        valid = valid and not LABEL.match(after)
        status = "candidate" if valid else "invalid"
        candidate = parent.replace(index, after) if valid else parent
        candidate_score = await score(candidate.text) if valid else parent_score
        reward = candidate_score - parent_score
        if valid:
            bandit.update(feature, reward)
            if all(candidate.text != p.text for p, _ in beam):
                beam.append((candidate, candidate_score))
                beam.sort(key=lambda item: item[1], reverse=True)
                del beam[config.beam_size :]
        entry = dict(
            iteration=iteration,
            index=index,
            before=before,
            after=after,
            reward=reward,
            parent_score=parent_score,
            score=candidate_score,
            best_score=beam[0][1],
            selection=selection,
            status=status,
            examples=examples,
            evaluations=len(cache),
            prompt=candidate.text,
            parent_prompt=parent.text,
        )
        history.append(entry)
        if on_step:
            on_step(entry)
    pool = [dict(prompt=p.text, score=s) for p, s in beam]
    return dict(
        initial_score=initial_score,
        best=pool[0],
        beam=pool,
        history=history,
        evaluations=len(cache),
    )
