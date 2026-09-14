"""Choose an annotation pool with diversity-discounted vote-k and uncertainty strata."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Example:
    input: str
    output: str


def vote_graph(embeddings: np.ndarray, k: int) -> list[list[int]]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)
    similarity = unit @ unit.T
    supporters = [[] for _ in embeddings]
    for voter in range(len(embeddings)):
        order = np.argsort(-similarity[voter], kind="stable")
        for candidate in [i for i in order if i != voter][:k]:
            supporters[candidate].append(voter)
    return supporters


class SelectiveAnnotation:
    """Own vote coverage, initial annotation, uncertainty stratification and retrieval.

    annotate is a label-oracle primitive, not an optimizer. uncertainty returns a
    finite higher-is-less-confident value for one input with locally retrieved demos.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Sequence[Example]], Awaitable[float]],
        embed: Callable[[Sequence[str]], Awaitable[np.ndarray]],
        annotate: Callable[[str], Awaitable[str]],
        uncertainty: Callable[[Sequence[Example], str], Awaitable[float]] | None = None,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.embed, self.annotate, self.uncertainty = embed, annotate, uncertainty

    @prompt(template="answer.j2")
    async def answer(self, input: str, demonstrations: Sequence[Example], *, generated: str) -> str:
        return generated

    def _pick(self, candidates, selected, coverage):
        def score(index):
            return sum(
                10.0 ** (-coverage[voter])
                for voter in self.supporters[index]
                if voter not in selected
            )

        # Source resolves initial ties by descending raw vote degree.
        return max(candidates, key=score)

    def _cover(self, index, selected, coverage):
        selected.append(int(index))
        coverage[self.supporters[index]] += 1

    async def _annotate(self, inputs, indices):
        for index in indices:
            self.annotations += 1
            self.labeled[index] = Example(inputs[index], await self.annotate(inputs[index]))

    def _retrieve(self, query, indices, neighbors):
        denominator = np.linalg.norm(self.embeddings[indices], axis=1) * np.linalg.norm(query)
        similarity = np.divide(
            self.embeddings[indices] @ query,
            denominator,
            out=np.zeros(len(indices)),
            where=denominator > 0,
        )
        ordered = [
            indices[i]
            for i in np.argsort(-similarity, kind="stable")
            if not np.isclose(similarity[i], 1, atol=1e-12, rtol=0)
        ]
        return tuple(self.labeled[i] for i in ordered[:neighbors][::-1])

    async def run(
        self,
        inputs: Sequence[str],
        *,
        budget: int,
        initial: int = 10,
        neighbors: int = 5,
        vote_neighbors: int = 150,
        mode: Literal["votek", "fast_votek"] = "votek",
    ) -> dict:
        self.annotations = self.uncertainty_calls = self.evaluations = 0
        self.labeled = {}
        self.embeddings = np.asarray(await self.embed(inputs), dtype=float)
        if not np.isfinite(self.embeddings).all():
            raise ValueError("embeddings must be finite")
        self.supporters = vote_graph(self.embeddings, min(vote_neighbors, len(inputs) - 1))
        ranked = sorted(range(len(inputs)), key=lambda i: len(self.supporters[i]), reverse=True)
        selected, coverage = [], np.zeros(len(inputs), dtype=int)
        initial_size = budget if mode == "fast_votek" else min(initial, budget)
        for _ in range(initial_size):
            candidate = self._pick([i for i in ranked if i not in selected], selected, coverage)
            self._cover(candidate, selected, coverage)
        await self._annotate(inputs, selected)
        uncertainty_scores = []
        if len(selected) < budget:
            initial_indices = selected.copy()
            for index in range(len(inputs)):
                if index in selected:
                    continue
                demos = self._retrieve(self.embeddings[index], initial_indices, neighbors)
                self.uncertainty_calls += 1
                score = float(await self.uncertainty(demos, inputs[index]))
                if not math.isfinite(score):
                    raise ValueError("uncertainty must be finite")
                uncertainty_scores.append((index, score))
            uncertainty_scores.sort(key=lambda item: item[1], reverse=True)
            width = max(1, int(len(inputs) * 0.9 / (budget - len(selected))))
            for start in range(0, len(uncertainty_scores), width):
                if len(selected) == budget:
                    break
                candidates = [
                    i for i, _ in uncertainty_scores[start : start + width] if i not in selected
                ]
                if candidates:
                    self._cover(self._pick(candidates, selected, coverage), selected, coverage)
            # Small pools can exhaust uncertainty strata; continue the same vote objective.
            while len(selected) < budget:
                self._cover(
                    self._pick([i for i in ranked if i not in selected], selected, coverage),
                    selected,
                    coverage,
                )
            await self._annotate(inputs, selected[len(initial_indices) :])
        self.evaluations += 1
        examples = tuple(self.labeled[i] for i in selected)
        score = float(await self.evaluate(examples))
        if not math.isfinite(score):
            raise ValueError("selected pool score must be finite")
        return {
            "indices": selected,
            "examples": examples,
            "score": score,
            "supporters": self.supporters,
            "coverage": coverage,
            "uncertainties": uncertainty_scores,
            "annotations": self.annotations,
            "uncertainty_calls": self.uncertainty_calls,
            "evaluations": self.evaluations,
        }
