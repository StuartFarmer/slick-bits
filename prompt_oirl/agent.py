"""Fit an offline binary reward proxy and select prompts separately for each query."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

Embed = Callable[[str], Awaitable[np.ndarray]]


@dataclass(frozen=True)
class Observation:
    query: str
    prompt: str
    correct: float


@dataclass(frozen=True)
class Tree:
    value: float
    feature: int = -1
    threshold: float = 0.0
    left: "Tree | None" = None
    right: "Tree | None" = None


def predict_tree(tree: Tree, row: np.ndarray) -> float:
    while tree.feature >= 0:
        tree = tree.left if row[tree.feature] < tree.threshold else tree.right
    return tree.value


def logistic_derivatives(logits: np.ndarray, labels: np.ndarray):
    probabilities = expit(logits)
    return probabilities - labels, probabilities * (1 - probabilities)


class PromptOIRL:
    """Exact greedy Newton trees approximate the released XGBoost binary-logistic proxy."""

    def __init__(self, task: str, embed: Embed):
        self.task = task
        self.embed = embed
        self.trees = []

    async def run(
        self,
        observations: Sequence[Observation],
        queries: Sequence[str],
        candidates: Sequence[str],
        *,
        rounds: int = 2000,
        learning_rate: float = 0.001,
        max_depth: int = 10,
        l2: float = 1.0,
        min_child_weight: float = 1.0,
    ) -> dict:
        self.trees = []
        self.learning_rate = learning_rate
        self.max_depth, self.l2, self.min_child_weight = max_depth, l2, min_child_weight
        self.embeddings = {}
        features, labels = await self.offline_data(observations)
        losses = self.fit(features, labels, rounds)
        selections = [await self.select(query, candidates) for query in queries]
        return {"selections": selections, "losses": losses, "trees": tuple(self.trees)}

    async def embedding(self, text):
        if text not in self.embeddings:
            value = np.array(await self.embed(text), dtype=float, copy=True)
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite embedding")
            self.embeddings[text] = value
        return self.embeddings[text]

    async def offline_data(self, observations):
        rows, labels = [], []
        for observation in observations:
            if not np.isfinite(observation.correct) or not 0 <= observation.correct <= 1:
                raise ValueError("Offline correctness must be a finite binary reward")
            rows.append(
                np.concatenate(
                    (
                        await self.embedding(observation.query),
                        await self.embedding(observation.prompt),
                    )
                )
            )
            labels.append(observation.correct)
        return np.array(rows), np.array(labels)

    def fit(self, features, labels, rounds):
        logits = np.zeros(len(labels))
        losses = []
        for _ in range(rounds):
            gradient, hessian = logistic_derivatives(logits, labels)
            tree = self.grow(features, gradient, hessian, 0)
            self.trees.append(tree)
            logits += self.learning_rate * np.array([predict_tree(tree, row) for row in features])
            loss = float(np.mean(np.logaddexp(0, logits) - labels * logits))
            if not np.isfinite(loss):
                raise ValueError("Nonfinite proxy loss")
            losses.append(loss)
        return losses

    def grow(self, features, gradient, hessian, depth):
        g, h = gradient.sum(), hessian.sum()
        leaf = Tree(float(-g / (h + self.l2)))
        if depth == self.max_depth or len(features) < 2:
            return leaf
        best_gain, best = 0.0, None
        for feature in range(features.shape[1]):
            order = np.argsort(features[:, feature], kind="stable")
            values = features[order, feature]
            left_g, left_h = np.cumsum(gradient[order])[:-1], np.cumsum(hessian[order])[:-1]
            right_g, right_h = g - left_g, h - left_h
            gains = 0.5 * (
                left_g**2 / (left_h + self.l2)
                + right_g**2 / (right_h + self.l2)
                - g**2 / (h + self.l2)
            )
            usable = (
                (values[:-1] < values[1:])
                & (left_h >= self.min_child_weight)
                & (right_h >= self.min_child_weight)
            )
            gains = np.where(usable, gains, -np.inf)
            split = int(np.argmax(gains))
            if gains[split] > best_gain:
                best_gain = gains[split]
                # The right observed value avoids floating midpoint overflow/rounding.
                best = feature, values[split + 1], order[: split + 1], order[split + 1 :]
        if best is None:
            return leaf
        feature, threshold, left, right = best
        return Tree(
            leaf.value,
            feature,
            float(threshold),
            self.grow(features[left], gradient[left], hessian[left], depth + 1),
            self.grow(features[right], gradient[right], hessian[right], depth + 1),
        )

    async def select(self, query: str, candidates: Sequence[str]) -> dict:
        query_embedding = await self.embedding(query)
        scores = []
        for prompt in candidates:
            row = np.concatenate((query_embedding, await self.embedding(prompt)))
            score = expit(self.learning_rate * sum(predict_tree(tree, row) for tree in self.trees))
            scores.append(float(score))
        index = int(np.argmax(scores))
        return {
            "query": query,
            "prompt": candidates[index],
            "probability": scores[index],
            "scores": scores,
        }
