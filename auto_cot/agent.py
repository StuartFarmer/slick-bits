"""Construct diverse automatic chain-of-thought demonstrations by clustering."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Demonstration:
    question: str
    rationale: str
    answer: str


def cluster(points: np.ndarray, k: int, rng: np.random.Generator) -> tuple:
    """K-means++ followed by Lloyd updates; coincident centers collapse."""
    centers = [points[int(rng.integers(len(points)))]]
    for _ in range(1, k):
        distance = np.min(np.sum((points[:, None] - np.array(centers)) ** 2, axis=2), axis=1)
        if distance.sum() == 0:
            break
        centers.append(points[int(rng.choice(len(points), p=distance / distance.sum()))])
    centers = np.array(centers)
    for _ in range(100):
        labels = np.argmin(np.sum((points[:, None] - centers) ** 2, axis=2), axis=1)
        updated = np.array(
            [points[labels == i].mean(axis=0) for i in range(len(centers)) if np.any(labels == i)]
        )
        if updated.shape == centers.shape and np.allclose(updated, centers):
            break
        centers = updated
    labels = np.argmin(np.sum((points[:, None] - centers) ** 2, axis=2), axis=1)
    return centers, labels


class AutoCoT:
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Sequence[Demonstration]], Awaitable[float]],
        embed: Callable[[Sequence[str]], Awaitable[np.ndarray]],
    ):
        self.task, self.provider, self.evaluate, self.embed = task, provider, evaluate, embed

    @prompt(template="reason.j2")
    async def reason(self, question: str, *, generated: str) -> str:
        return generated.strip()

    @prompt(template="extract.j2")
    async def extract(self, question: str, rationale: str, *, generated: str) -> str:
        return generated.strip()

    @prompt(template="answer.j2")
    async def answer(self, question: str, *, generated: str) -> str:
        return generated

    def _select(self, traces, centers, labels, points, max_words, max_lines, arithmetic):
        selected, indices = [], []
        for i, center in enumerate(centers):
            members = np.flatnonzero(labels == i)
            order = members[
                np.argsort(np.sum((points[members] - center) ** 2, axis=1), kind="stable")
            ]
            for index in order:
                trace = traces[index]
                rationale = trace.rationale.replace("\n\n", "\n").strip()
                if (
                    len(trace.question.split()) > max_words
                    or len(rationale.split("\n")) > max_lines
                    or not rationale.endswith(".")
                    or not trace.answer
                ):
                    continue
                if arithmetic and (
                    trace.answer not in rationale[:-1].split(".")[-1]
                    and trace.answer not in rationale.split()[-10:]
                ):
                    continue
                selected.append(
                    Demonstration(trace.question, rationale.replace("\n", " "), trace.answer)
                )
                indices.append(int(index))
                break
        return selected, indices

    async def run(
        self,
        questions: Sequence[str],
        *,
        clusters: int = 8,
        max_question_words: int = 60,
        max_rationale_lines: int = 5,
        arithmetic: bool = False,
        seed: int = 0,
    ) -> dict:
        """Generate zero-shot traces, choose one admissible centroid neighbor per cluster."""
        self.generations = self.evaluations = 0
        points = np.asarray(await self.embed(questions), dtype=float)
        if not np.isfinite(points).all():
            raise ValueError("embeddings must be finite")
        centers, labels = cluster(
            points, min(clusters, len(questions)), np.random.default_rng(seed)
        )
        traces = []
        for question in questions:
            self.generations += 1
            rationale = await self.reason(question, provider=self.provider)
            self.generations += 1
            answer = await self.extract(question, rationale, provider=self.provider)
            traces.append(Demonstration(question, rationale, answer))
        self.demonstrations, indices = self._select(
            traces, centers, labels, points, max_question_words, max_rationale_lines, arithmetic
        )
        self.evaluations += 1
        score = float(await self.evaluate(tuple(self.demonstrations)))
        if not math.isfinite(score):
            raise ValueError("demonstration score must be finite")
        return {
            "demonstrations": tuple(self.demonstrations),
            "indices": indices,
            "traces": traces,
            "score": score,
            "generations": self.generations,
            "evaluations": self.evaluations,
        }
