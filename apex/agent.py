"""APEX: sentence mutation with immediate beam updates and history-guided search."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from slick import Session, prompt
from slick.providers import Provider


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


LABEL = re.compile(r"(?:Q:|A:|Question[.:]|Answer[.:]|\([A-Za-z0-9]+\))\s*")


@dataclass(frozen=True)
class Document:
    parts: tuple[str, ...]
    mutable: tuple[int, ...]

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
    def split(cls, text):
        parts, mutable = [], []
        # ponytail: punctuation heuristic misses abbreviations/complex markup;
        # use explicit Document fragments when exact sentence boundaries matter.
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


@dataclass(frozen=True)
class Mutation:
    """Retain the raw response even when a sentence edit is rejected."""

    raw: str
    text: str
    valid: bool

    @classmethod
    def parse(cls, response: str) -> "Mutation":
        text = response.strip()
        valid = bool(text) and len(text.splitlines()) == 1 and "```" not in text
        valid = valid and not re.search(r"</?(?:original|rephrased)\b", text, re.IGNORECASE)
        valid = valid and not LABEL.match(text)
        return cls(response, text, bool(valid))


class APEX:
    """Optimize sentence edits against an injected higher-is-better score.

    The evaluator and synchronous batch encoder belong to the caller. Scores
    are cached per run, so evaluation must be repeatable. Invalid sentence
    responses consume an iteration without changing the beam or bandit;
    provider, encoder, and evaluation exceptions propagate without retries.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        encode: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.encode = encode

    @prompt(template="mutate.j2")
    async def mutate(self, sentence: str, *, generated: str) -> Mutation:
        """Check an independent sentence variation and retain rejected raw output."""
        return Mutation.parse(generated)

    @prompt(template="mutate_guided.j2")
    async def mutate_guided(
        self, sentence: str, examples: list[tuple[str, str]], *, generated: str
    ) -> Mutation:
        """Check a sentence rephrasing guided by previous successful edits."""
        return Mutation.parse(generated)

    async def _score(self, text: str) -> float:
        if text not in self.scores:
            value = float(await self.evaluate(text))
            if not math.isfinite(value):
                raise ValueError("evaluator returned a non-finite score")
            self.scores[text] = value
        return self.scores[text]

    def _select_sentence(self, document: Document):
        vectors = self.embeddings([document.parts[i] for i in document.mutable])
        if self.rng.random() < self.config.random_probability:
            arm = self.rng.randrange(len(vectors))
            selection = "random"
        else:
            values = self.bandit.values(vectors)
            arm = self.rng.choice(
                np.flatnonzero(np.isclose(values, values.max(), atol=1e-12, rtol=1e-12)).tolist()
            )
            selection = "linucb"
        return document.mutable[arm], vectors[arm], selection

    async def _mutate_sentence(self, sentence, feature, execution):
        if self.config.guided_mutation:
            examples = retrieve(
                feature,
                self.history,
                self.embeddings,
                self.config.history_limit,
                self.config.history_distance,
            )
            return await self.mutate_guided(sentence, examples, **execution), examples
        return await self.mutate(sentence, **execution), []

    async def _apply_mutation(self, parent, parent_score, index, feature, mutation):
        candidate = parent.replace(index, mutation.text) if mutation.valid else parent
        score = await self._score(candidate.text) if mutation.valid else parent_score
        reward = score - parent_score
        if mutation.valid:
            self.bandit.update(feature, reward)
            if all(candidate.text != p.text for p, _ in self.beam):
                self.beam.append((candidate, score))
                self.beam.sort(key=lambda item: item[1], reverse=True)
                del self.beam[self.config.beam_size :]
        return dict(
            index=index,
            before=parent.parts[index],
            after=mutation.text,
            response=mutation.raw,
            reward=reward,
            parent_score=parent_score,
            score=score,
            best_score=self.beam[0][1],
            status="candidate" if mutation.valid else "invalid",
            evaluations=len(self.scores),
            prompt=candidate.text,
            parent_prompt=parent.text,
        )

    async def run(
        self, document: Document, *, config: Config = Config(), session: Session | None = None
    ) -> dict:
        """Select, mutate, score, and immediately update a bounded beam.

        Use explicit Document fragments for exact immutable boundaries, or
        Document.split(text) for heuristic sentence splitting. Run one search
        at a time per agent. A supplied Session owns the sequential conversation.
        """
        execution = {"session": session} if session is not None else {"provider": self.provider}
        self.config = config
        self.rng = random.Random(config.seed)
        self.embeddings = Embeddings(self.encode)
        self.bandit = LinUCB(config.regularization, config.alpha)
        self.scores, self.history = {}, []
        initial_score = await self._score(document.text)
        self.beam = [(document, initial_score)]
        for iteration in range(1, config.iterations + 1):
            if not document.mutable:
                break
            parent, parent_score = self.rng.choice(self.beam)
            index, feature, selection = self._select_sentence(parent)
            mutation, examples = await self._mutate_sentence(
                parent.parts[index], feature, execution
            )
            entry = await self._apply_mutation(parent, parent_score, index, feature, mutation)
            self.history.append(
                dict(iteration=iteration, selection=selection, examples=examples, **entry)
            )
        pool = [dict(prompt=p.text, score=s) for p, s in self.beam]
        return dict(
            initial_score=initial_score,
            best=pool[0],
            beam=pool,
            history=self.history,
            evaluations=len(self.scores),
        )
