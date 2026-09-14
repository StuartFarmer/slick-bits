"""Evolve named prompts using reflection, instance-wise Pareto selection, and merge.

Adapted from gepa-ai/gepa; see README.md and LICENSE for provenance.
Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors.
"""

import math
import random
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Generic, TypeVar

from slick import prompt
from slick.providers import Provider

Instance = TypeVar("Instance")


@dataclass(frozen=True)
class Evaluation:
    """One complete system rollout; higher finite scores are better."""

    score: float
    output: str = ""
    feedback: str = ""
    trace: str = ""
    module_feedback: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    prompts: dict[str, str]
    scores: tuple[float, ...]
    parents: tuple[int, ...] = ()

    @property
    def score(self) -> float:
        return fmean(self.scores)


class InvalidInstruction(ValueError):
    """Reflection did not produce a nonempty fenced instruction."""


def pareto_weights(scores: Sequence[Sequence[float]]) -> dict[int, int]:
    """Port upstream's coverage pruning and instance-win sampling weights.

    A candidate is redundant when all its wins are covered by remaining
    candidates, which is stronger pruning than strict vector dominance.
    """
    fronts = []
    for column in zip(*scores):
        best = max(column)
        fronts.append({i for i, score in enumerate(column) if score == best})
    order = dict.fromkeys(i for front in fronts for i in front)
    remaining = set(order)
    # Removing coverage cannot make a previously indispensable candidate redundant.
    for i in sorted(order, key=lambda i: fmean(scores[i])):
        wins = [front for front in fronts if i in front]
        if all((front & remaining) - {i} for front in wins):
            remaining.remove(i)
    return dict(Counter(i for front in fronts for i in front if i in remaining))


class GEPA(Generic[Instance]):
    """Own a sequential GEPA search over arbitrary caller-executed systems.

    The evaluator receives a fresh prompt dictionary and one opaque instance.
    It owns system execution, isolation, and feedback. Provider/evaluator errors
    propagate; malformed reflections are logged and rejected without retries.
    Configure Slick's template root at startup. Calls use independent provider
    contexts; validation content is never sent to the reflection model.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Mapping[str, str], Instance], Awaitable[Evaluation]],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.raw_responses: list[str] = []

    @prompt(template="reflect.j2")
    async def reflect(
        self, module: str, instruction: str, examples: Sequence[dict], *, generated: str
    ) -> str:
        """Extract the paper's textual instruction, retaining rejected raw output."""
        self.raw_responses.append(generated)
        # First opening / last closing fence permits code fences inside a prompt.
        match = re.search(r"```[^\n]*\n(.*)\n```", generated, re.DOTALL)
        if match is None or not match[1].strip():
            raise InvalidInstruction("Expected a nonempty fenced instruction")
        return match[1].strip()

    async def _evaluate(self, prompts, instances):
        if self.rollouts + len(instances) > self.budget:
            raise RuntimeError("Insufficient rollout budget for complete evaluation")
        results = []
        for instance in instances:
            self.rollouts += 1  # Failed evaluator calls still consume an attempt.
            result = await self.evaluate(dict(prompts), instance)
            if not math.isfinite(result.score):
                raise ValueError("Measured scores must be finite")
            results.append(result)
        return results

    def _register(self, prompts, scores, parents=()):
        self.population.append(Candidate(dict(prompts), tuple(scores), parents))
        self.next_modules.append(max((self.next_modules[i] for i in parents), default=0))
        return len(self.population) - 1

    async def _mutate(self, parent_idx, batch):
        parent = self.population[parent_idx]
        module = self.modules[self.next_modules[parent_idx]]
        self.next_modules[parent_idx] = (self.next_modules[parent_idx] + 1) % len(self.modules)
        record = {"operation": "mutation", "parents": (parent_idx,), "module": module}
        self.history.append(record)
        before = await self._evaluate(parent.prompts, batch)
        examples = [
            {
                "input": str(item),
                "output": result.output,
                "score": result.score,
                "feedback": result.feedback,
                "trace": result.trace,
                "module_feedback": result.module_feedback.get(module, ""),
            }
            for item, result in zip(batch, before)
        ]
        record["before"] = fmean(result.score for result in before)
        self.reflection_calls += 1
        try:
            instruction = await self.reflect(
                module, parent.prompts[module], examples, provider=self.provider
            )
        except InvalidInstruction as exc:
            record.update(status="invalid", error=str(exc))
            return False
        child = {**parent.prompts, module: instruction}
        if child == parent.prompts:
            record["status"] = "unchanged"
            return False
        after = await self._evaluate(child, batch)
        record["after"] = fmean(result.score for result in after)
        if record["after"] <= record["before"]:
            record["status"] = "not_improved"
            return False
        validation = await self._evaluate(child, self.validation)
        idx = self._register(child, [result.score for result in validation], (parent_idx,))
        record.update(status="accepted", candidate=idx)
        return True

    def _ancestors(self, idx):
        ancestors = set()
        pending = list(self.population[idx].parents)
        while pending:
            parent = pending.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                pending.extend(self.population[parent].parents)
        return ancestors

    def _propose_merge(self):
        frontier = list(pareto_weights([c.scores for c in self.population]))
        if len(frontier) < 2:
            return None
        for _ in range(10):
            i, j = sorted(self.rng.sample(frontier, 2))
            first, second = self.population[i], self.population[j]
            ai, aj = self._ancestors(i), self._ancestors(j)
            if i in aj or j in ai:
                continue
            ancestors = [
                a
                for a in sorted(ai & aj)
                if (i, j, a) not in self.tried_merges
                and self.population[a].score <= min(first.score, second.score)
                and any(
                    (self.population[a].prompts[m] in (first.prompts[m], second.prompts[m]))
                    and first.prompts[m] != second.prompts[m]
                    for m in self.modules
                )
            ]
            if not ancestors:
                continue
            a = self.rng.choice(ancestors)
            self.tried_merges.add((i, j, a))
            child = {}
            for m in self.modules:
                original, left, right = (
                    self.population[a].prompts[m],
                    first.prompts[m],
                    second.prompts[m],
                )
                if left == original:
                    child[m] = right
                elif right == original or left == right:
                    child[m] = left
                elif first.score == second.score:
                    child[m] = self.rng.choice([left, right])
                else:
                    child[m] = left if first.score > second.score else right
            if any(child == candidate.prompts for candidate in self.population):
                continue
            return child, i, j, a
        return None

    def _merge_batch(self, first, second):
        size = min(5, len(self.validation))
        buckets = [[], [], []]
        for idx, (left, right) in enumerate(zip(first.scores, second.scores)):
            bucket = 0 if left > right else 1 if right > left else 2
            buckets[bucket].append(idx)
        selected = []
        for bucket in buckets:
            count = min(len(bucket), math.ceil(size / 3), size - len(selected))
            selected.extend(self.rng.sample(bucket, count))
        unused = sorted(set(range(len(self.validation))) - set(selected))
        selected.extend(self.rng.sample(unused, size - len(selected)))
        return selected

    async def _try_merge(self):
        if self.rollouts + len(self.validation) > self.budget:
            return False
        proposal = self._propose_merge()
        if proposal is None:
            return False
        child, i, j, ancestor = proposal
        first, second = self.population[i], self.population[j]
        batch = self._merge_batch(first, second)
        self.merge_attempts += 1
        record = {
            "operation": "merge",
            "parents": (i, j),
            "ancestor": ancestor,
            "validation_indices": tuple(batch),
        }
        self.history.append(record)
        results = await self._evaluate(child, [self.validation[k] for k in batch])
        record["before"] = max(fmean(c.scores[k] for k in batch) for c in (first, second))
        record["after"] = fmean(result.score for result in results)
        if record["after"] < record["before"]:
            record["status"] = "not_improved"
            return False
        # Reuse these measurements so each validation instance costs one rollout.
        scores = dict(zip(batch, (result.score for result in results)))
        remaining = [k for k in range(len(self.validation)) if k not in scores]
        results = await self._evaluate(child, [self.validation[k] for k in remaining])
        scores.update(zip(remaining, (result.score for result in results)))
        idx = self._register(child, [scores[k] for k in range(len(self.validation))], (i, j))
        record.update(status="accepted", candidate=idx)
        return True

    async def run(
        self,
        seed_prompts: Mapping[str, str],
        train: Sequence[Instance],
        validation: Sequence[Instance],
        *,
        budget: int = 1000,
        minibatch_size: int = 3,
        max_merges: int = 0,
        seed: int = 0,
    ) -> dict:
        """Return the best mean-validation candidate, ancestry, and attempt records.

        Supply nonempty prompts and datasets, a positive minibatch size, and a
        budget covering baseline validation. Each mutation reserves enough for
        parent + child minibatches and full validation. Unused tail budget stays
        unspent. Set max_merges=5 for GEPA+Merge. Runs reset state; no overlap.
        """
        self.rng = random.Random(seed)
        self.budget, self.validation = budget, tuple(validation)
        self.modules = tuple(seed_prompts)
        self.rollouts = self.reflection_calls = self.merge_attempts = 0
        self.population, self.next_modules, self.history, self.raw_responses = [], [], [], []
        self.tried_merges = set()
        baseline = await self._evaluate(seed_prompts, self.validation)
        self._register(seed_prompts, [result.score for result in baseline])
        batch_size = min(minibatch_size, len(train))
        while self.rollouts + 2 * batch_size + len(validation) <= budget:
            weights = pareto_weights([candidate.scores for candidate in self.population])
            parent = self.rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
            batch = self.rng.sample(list(train), batch_size)
            accepted = await self._mutate(parent, batch)
            if accepted and self.merge_attempts < max_merges:
                await self._try_merge()
        return {
            "best": max(self.population, key=lambda candidate: candidate.score),
            "population": tuple(self.population),
            "pareto_weights": pareto_weights([candidate.scores for candidate in self.population]),
            "history": self.history,
            "raw_responses": self.raw_responses,
            "rollouts": self.rollouts,
            "reflection_calls": self.reflection_calls,
            "merge_attempts": self.merge_attempts,
        }
