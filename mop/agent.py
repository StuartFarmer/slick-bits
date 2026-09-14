"""Build a mixture of expert prompts using clustered demonstrations and regional assignment."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Example:
    input: str
    target: str


@dataclass(frozen=True)
class Expert:
    instruction: str
    demonstrations: tuple[Example, ...]
    score: float


class MoP:
    """Own expert-count selection, routing, complement generation and regional search.

    evaluate(instruction, demonstrations, examples) returns a finite higher score
    for better task execution. embed receives input strings without target labels.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str, Sequence[Example], Sequence[Example]], Awaitable[float]],
        embed: Callable[[Sequence[str]], Awaitable[np.ndarray]],
    ):
        self.task, self.provider, self.evaluate, self.embed = task, provider, evaluate, embed

    @prompt(template="induce.j2")
    async def induce(self, examples: Sequence[Example], *, generated: str) -> str:
        """Generate a compensating instruction from the other experts' demonstrations."""
        if not generated.strip():
            raise ValueError(f"empty expert instruction; response={generated!r}")
        return generated.strip()

    @prompt(template="answer.j2")
    async def answer(
        self, instruction: str, demonstrations: Sequence[Example], input: str, *, generated: str
    ) -> str:
        """Execute one routed expert with its fixed instruction and demonstrations."""
        return generated

    def _kmeans(self, points, k):
        centers = [points[int(self.rng.integers(len(points)))]]
        for _ in range(1, k):
            distance = np.min(
                np.sum((points[:, None] - np.array(centers)[None]) ** 2, axis=2), axis=1
            )
            if not distance.sum():
                break
            centers.append(points[int(self.rng.choice(len(points), p=distance / distance.sum()))])
        centers = np.array(centers)
        for _ in range(100):
            labels = np.argmin(np.sum((points[:, None] - centers[None]) ** 2, axis=2), axis=1)
            updated = np.array(
                [
                    points[labels == i].mean(axis=0)
                    for i in range(len(centers))
                    if np.any(labels == i)
                ]
            )
            if updated.shape == centers.shape and np.allclose(updated, centers):
                break
            centers = updated
        labels = np.argmin(np.sum((points[:, None] - centers[None]) ** 2, axis=2), axis=1)
        inertia = np.sum((points - centers[labels]) ** 2)
        return centers, labels, inertia

    def _partition(self, examples, embeddings, max_experts, penalty):
        total_inertia = np.sum((embeddings - embeddings.mean(axis=0)) ** 2)
        options = []
        for k in range(2, min(max_experts, len(examples)) + 1):
            centers, labels, inertia = self._kmeans(embeddings, k)
            objective = (
                inertia / total_inertia + penalty * len(centers) if total_inertia else penalty
            )
            options.append((objective, centers, labels))
        if not options:
            centers, labels, _ = self._kmeans(embeddings, 1)
        else:
            _, centers, labels = min(options, key=lambda option: option[0])
        self.centers = centers
        self.partitions, self.trimmed, self.regional_centers = [], [], []
        size = max(1, len(examples) // max_experts)
        for i in range(len(centers)):
            indices = np.flatnonzero(labels == i).tolist()
            if len(indices) > size:
                saved = self.rng.choice(indices, size, replace=False).tolist()
                self.trimmed.extend(examples[j] for j in indices if j not in saved)
            else:
                saved = (indices * (size // len(indices) + 1))[:size]
            self.partitions.append(tuple(examples[j] for j in saved))
            self.regional_centers.append(embeddings[saved].mean(axis=0))
        self.cluster_history = [{"objective": o[0], "experts": len(o[1])} for o in options]

    async def _score(self, instruction, demos, examples, phase):
        self.evaluations += 1
        score = float(await self.evaluate(instruction, demos, examples))
        if not np.isfinite(score):
            raise ValueError("expert score must be finite")
        self.measurements.append({"instruction": instruction, "score": score, "phase": phase})
        return score

    async def _generate_candidates(self, validation, candidates_per_region, pool_size):
        candidates = []
        for region in range(len(self.partitions)):
            complement = [
                example
                for i, partition in enumerate(self.partitions)
                if i != region
                for example in partition
            ]
            complement += self.trimmed
            if not complement:
                complement = list(self.partitions[region])
            for _ in range(candidates_per_region):
                self.optimizer_calls += 1
                candidates.append(await self.induce(complement, provider=self.provider))
        ranked = []
        for candidate in dict.fromkeys(candidates):
            ranked.append((candidate, await self._score(candidate, (), validation, "candidate")))
        return [
            candidate
            for candidate, _ in sorted(ranked, key=lambda row: row[1], reverse=True)[:pool_size]
        ]

    async def _assign(self, candidates, validation, embeddings, regional_examples):
        self.experts = []
        norms = np.linalg.norm(embeddings, axis=1)
        for region, demos in enumerate(self.partitions):
            center = self.regional_centers[region]
            denominator = norms * np.linalg.norm(center)
            similarity = np.divide(
                embeddings @ center,
                denominator,
                out=np.zeros(len(embeddings)),
                where=denominator > 0,
            )
            selected = np.argsort(-similarity, kind="stable")[:regional_examples]
            examples = [validation[i] for i in selected]
            ranked = []
            for candidate in candidates:
                ranked.append((candidate, await self._score(candidate, demos, examples, "region")))
            instruction, score = max(ranked, key=lambda row: row[1])
            self.experts.append(Expert(instruction, demos, score))

    async def route(self, input: str) -> int:
        """Choose the nearest training cluster using only the query's input embedding."""
        embedding = np.asarray(await self.embed([input]))[0]
        if not np.isfinite(embedding).all():
            raise ValueError("query embedding must be finite")
        return int(np.argmin(np.sum((self.centers - embedding) ** 2, axis=1)))

    async def predict(self, input: str) -> str:
        expert = self.experts[await self.route(input)]
        return await self.answer(
            expert.instruction, expert.demonstrations, input, provider=self.provider
        )

    async def run(
        self,
        demonstrations: Sequence[Example],
        validation: Sequence[Example],
        *,
        max_experts: int = 4,
        cluster_penalty: float = 0.02,
        candidates_per_region: int = 3,
        pool_size: int = 5,
        regional_examples: int = 20,
        seed: int = 0,
    ) -> dict:
        """Partition demos, generate shared compensating instructions, and assign by region."""
        self.rng = np.random.default_rng(seed)
        self.optimizer_calls = self.evaluations = 0
        self.measurements = []
        embeddings = np.asarray(await self.embed([e.input for e in demonstrations]))
        eval_embeddings = np.asarray(await self.embed([e.input for e in validation]))
        if not np.isfinite(embeddings).all() or not np.isfinite(eval_embeddings).all():
            raise ValueError("example embeddings must be finite")
        self._partition(demonstrations, embeddings, max_experts, cluster_penalty)
        candidates = await self._generate_candidates(validation, candidates_per_region, pool_size)
        await self._assign(
            candidates, validation, eval_embeddings, min(len(validation), regional_examples)
        )
        return {
            "experts": self.experts,
            "centers": self.centers.copy(),
            "candidates": candidates,
            "clusters": self.cluster_history,
            "measurements": self.measurements,
            "optimizer_calls": self.optimizer_calls,
            "evaluations": self.evaluations,
        }
