"""Boost weighted discrete verbalizers selected from frozen language-model token scores."""

import itertools
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from slick.prompts import Prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Learner:
    template: str
    verbalizer: tuple[int, ...]
    weight: float


def verbalizer_candidates(scores, labels, weights, classes, top_tokens, normalize_classes):
    """Weighted one-versus-rest L1 token association, class ownership and Cartesian search."""
    multipliers = np.full((classes, len(labels)), -1 / (classes - 1) if normalize_classes else -1.0)
    multipliers[np.arange(classes)[:, None] == labels] = 1
    association = (multipliers * weights * len(labels)) @ scores
    owners = np.argmax(association, axis=0)
    candidates = []
    for label in range(classes):
        masked = np.where(owners == label, association[label], -10000.0)
        candidates.append(np.argsort(-masked, kind="stable")[:top_tokens].tolist())
    pairs = list(itertools.product(*candidates))
    if classes == 2:
        pairs += [tuple(reversed(pair)) for pair in pairs.copy()]
    return pairs


class PromptBoosting:
    """Own weighted verbalizer search, SAMME updates and best validation-prefix selection.

    token_scores(provider, contexts) returns frozen model scores over its entire
    vocabulary. evaluate receives validation predictions, with higher scores better.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[np.ndarray], Awaitable[float]],
        token_scores: Callable[[Provider, Sequence[str]], Awaitable[np.ndarray]],
    ):
        self.task, self.provider, self.evaluate, self.token_scores = (
            task,
            provider,
            evaluate,
            token_scores,
        )

    async def _scores(self, template, inputs):
        contexts = [
            Prompt("context.j2")(task=self.task, context=template.replace("{input}", x))
            for x in inputs
        ]
        self.model_calls += 1
        scores = np.asarray(await self.token_scores(self.provider, contexts))
        if not np.isfinite(scores).all():
            raise ValueError("token scores must be finite")
        return scores

    def _weak_learner(self, scores, labels, weights, classes, top_tokens, trials, normalize, rng):
        pairs = verbalizer_candidates(scores, labels, weights, classes, top_tokens, normalize)
        sampled = rng.choice(len(pairs), min(trials, len(pairs)), replace=False)
        best = None
        for index in sampled:
            pair = pairs[index]
            predictions = np.argmax(scores[:, pair], axis=1)
            wrong = predictions != labels
            error = float(weights @ wrong)
            self.candidates_scored += 1
            if best is None or error < best[0]:
                best = error, pair, wrong
        return best

    async def predict(self, inputs: Sequence[str]) -> np.ndarray:
        votes = np.zeros((len(inputs), self.classes))
        cache = {}
        for learner in self.learners:
            if learner.template not in cache:
                cache[learner.template] = await self._scores(learner.template, inputs)
            predictions = np.argmax(cache[learner.template][:, learner.verbalizer], axis=1)
            votes[np.arange(len(inputs)), predictions] += learner.weight
        return np.argmax(votes, axis=1)

    async def run(
        self,
        templates: Sequence[str],
        training: Sequence[str],
        labels: np.ndarray,
        validation: Sequence[str],
        *,
        classes: int,
        rounds: int = 20,
        top_tokens: int = 5,
        candidates_per_round: int = 20000,
        learning_rate: float = 1.0,
        normalize_classes: bool = False,
        random_templates: bool = True,
        seed: int = 0,
    ) -> dict:
        rng = np.random.default_rng(seed)
        self.classes = classes
        self.model_calls = self.evaluations = self.candidates_scored = 0
        weights = np.full(len(training), 1 / len(training))
        train_cache, valid_cache = {}, {}
        learners, history = [], []
        votes = np.zeros((len(validation), classes))
        best_score, best_count = -math.inf, 0
        template_order = (
            rng.integers(len(templates), size=rounds)
            if random_templates
            else np.arange(rounds) % len(templates)
        )
        for step in range(rounds):
            template = templates[template_order[step]]
            if template not in train_cache:
                train_cache[template] = await self._scores(template, training)
                valid_cache[template] = await self._scores(template, validation)
            error, pair, wrong = self._weak_learner(
                train_cache[template],
                labels,
                weights,
                classes,
                top_tokens,
                candidates_per_round,
                normalize_classes,
                rng,
            )
            before = weights.copy()
            if error >= 1 - 1 / classes:
                history.append({"error": error, "accepted": False, "weights": before})
                continue
            bounded_error = max(error, np.finfo(float).eps)
            alpha = learning_rate * (
                math.log((1 - bounded_error) / bounded_error) + math.log(classes - 1)
            )
            weights *= np.exp(alpha * wrong)
            weights /= weights.sum()
            learners.append(Learner(template, pair, alpha))
            predictions = np.argmax(valid_cache[template][:, pair], axis=1)
            votes[np.arange(len(validation)), predictions] += alpha
            self.evaluations += 1
            score = float(await self.evaluate(np.argmax(votes, axis=1)))
            if not math.isfinite(score):
                raise ValueError("ensemble score must be finite")
            if score >= best_score:
                best_score, best_count = score, len(learners)
            history.append(
                {
                    "error": error,
                    "accepted": True,
                    "weights": before,
                    "updated_weights": weights.copy(),
                    "alpha": alpha,
                    "score": score,
                }
            )
            if error == 0:
                break
        if not learners:
            raise ValueError("no weak learner outperformed chance")
        self.learners = learners[:best_count]
        return {
            "learners": self.learners,
            "all_learners": learners,
            "score": best_score,
            "history": history,
            "model_calls": self.model_calls,
            "evaluations": self.evaluations,
            "candidates_scored": self.candidates_scored,
        }
