"""Train an efficient prompt retriever from language-model-ranked demonstrations."""

from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Example:
    input: str
    output: str


def contrastive_loss(queries: np.ndarray, contexts: np.ndarray, positives: np.ndarray) -> tuple:
    """All-in-batch dot-product NLL and derivatives with respect to both encoders' outputs."""
    logits = queries @ contexts.T
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -float(log_probs[np.arange(len(queries)), positives].mean())
    derivative = np.exp(log_probs)
    derivative[np.arange(len(queries)), positives] -= 1
    derivative /= len(queries)
    return loss, derivative @ contexts, derivative.T @ queries


def bm25(
    corpus: Sequence[Sequence[str]], query: Sequence[str], k1: float = 1.5, b: float = 0.75
) -> np.ndarray:
    """BM25Okapi including its average-IDF floor for common terms."""
    counts = [Counter(tokens) for tokens in corpus]
    frequencies = Counter(token for row in counts for token in row)
    size = len(corpus)
    idfs = {
        token: np.log(size - count + 0.5) - np.log(count + 0.5)
        for token, count in frequencies.items()
    }
    average = np.mean(list(idfs.values())) if idfs else 0.0
    idfs = {token: value if value >= 0 else 0.25 * average for token, value in idfs.items()}
    lengths = np.array([len(row) for row in corpus])
    mean_length = max(float(lengths.mean()), 1.0)
    scores = np.zeros(size)
    for token in query:
        frequency = np.array([row[token] for row in counts])
        scores += (
            idfs.get(token, 0.0)
            * frequency
            * (k1 + 1)
            / (frequency + k1 * (1 - b + b * lengths / mean_length))
        )
    return scores


class EPR:
    """Own BM25 candidate mining, LM supervision, contrastive training and dot-product retrieval.

    evaluate(query, demonstration) returns finite answer NLL, lower being better.
    encode(parameters, texts, role) returns embeddings and their parameter Jacobians
    of shape (texts, embedding dimensions, parameters); role is query or context.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Example, Example], Awaitable[float]],
        encode: Callable[
            [np.ndarray, Sequence[str], str], Awaitable[tuple[np.ndarray, np.ndarray]]
        ],
        tokenize: Callable[[str], Sequence[str]] = str.split,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.encode, self.tokenize = encode, tokenize

    @prompt(template="answer.j2")
    async def answer(self, input: str, demonstrations: Sequence[Example], *, generated: str) -> str:
        return generated

    @staticmethod
    def _context(example):
        return example.input + "\t" + example.output

    async def _supervise(self, pool, candidate_count, positive_count):
        corpus = [self.tokenize(self._context(example)) for example in pool]
        supervision = []
        for index, query in enumerate(pool):
            scores = bm25(corpus, corpus[index])
            candidates = [int(i) for i in np.argsort(-scores, kind="stable") if i != index][
                :candidate_count
            ]
            losses = []
            for candidate in candidates:
                self.evaluations += 1
                loss = float(await self.evaluate(query, pool[candidate]))
                if not np.isfinite(loss):
                    raise ValueError("demonstration answer loss must be finite")
                losses.append((candidate, loss))
            losses.sort(key=lambda row: row[1])
            supervision.append(
                {
                    "ranked": losses,
                    "positives": [i for i, _ in losses[:positive_count]],
                    "hard_negatives": [i for i, _ in losses[-positive_count:]],
                }
            )
        return supervision

    async def _encode(self, texts, role):
        self.encoder_calls += 1
        values, jacobian = await self.encode(self.parameters, texts, role)
        values, jacobian = np.asarray(values), np.asarray(jacobian)
        if not np.isfinite(values).all() or not np.isfinite(jacobian).all():
            raise ValueError("encoder values and Jacobians must be finite")
        return values, jacobian

    async def _train_batch(self, pool, supervision, indices, hard_negatives, random_negatives, rng):
        contexts, positives = [], []
        for index in indices:
            positives.append(len(contexts))
            contexts.append(int(rng.choice(supervision[index]["positives"])))
            contexts.extend(rng.integers(len(pool), size=random_negatives).tolist())
            hard = supervision[index]["hard_negatives"]
            contexts.extend(
                rng.choice(hard, min(hard_negatives, len(hard)), replace=False).tolist()
            )
        queries, q_jac = await self._encode([pool[i].input for i in indices], "query")
        embeddings, c_jac = await self._encode(
            [self._context(pool[i]) for i in contexts], "context"
        )
        loss, q_grad, c_grad = contrastive_loss(queries, embeddings, np.array(positives))
        gradient = np.einsum("nd,ndp->p", q_grad, q_jac) + np.einsum("nd,ndp->p", c_grad, c_jac)
        return loss, gradient, contexts

    async def retrieve(self, input: str, *, count: int = 5) -> tuple[Example, ...]:
        query, _ = await self._encode([input], "query")
        indices = np.argsort(-(self.index @ query[0]), kind="stable")[:count][::-1]
        return tuple(self.pool[i] for i in indices)

    async def run(
        self,
        pool: Sequence[Example],
        parameters: np.ndarray,
        *,
        candidate_count: int = 50,
        positive_count: int = 5,
        steps: int = 100,
        batch_size: int = 8,
        hard_negatives: int = 1,
        random_negatives: int = 0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        seed: int = 0,
    ) -> dict:
        self.parameters = np.array(parameters, dtype=float, copy=True)
        self.pool = tuple(pool)
        self.evaluations = self.encoder_calls = 0
        rng = np.random.default_rng(seed)
        supervision = await self._supervise(pool, candidate_count, positive_count)
        first, second = np.zeros_like(self.parameters), np.zeros_like(self.parameters)
        history = []
        for step in range(1, steps + 1):
            indices = rng.choice(len(pool), min(batch_size, len(pool)), replace=False)
            loss, gradient, contexts = await self._train_batch(
                pool, supervision, indices, hard_negatives, random_negatives, rng
            )
            gradient *= min(1, 2 / (np.linalg.norm(gradient) + 1e-12))
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            update = first / (1 - 0.9**step) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            self.parameters *= 1 - learning_rate * weight_decay
            self.parameters -= learning_rate * update
            history.append({"loss": loss, "queries": indices.tolist(), "contexts": contexts})
        self.index, _ = await self._encode([self._context(example) for example in pool], "context")
        return {
            "parameters": self.parameters.copy(),
            "supervision": supervision,
            "history": history,
            "index": self.index.copy(),
            "evaluations": self.evaluations,
            "encoder_calls": self.encoder_calls,
        }
