"""Diverse prompt mutations and novelty-weighted failure rules from MAPS."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel
from slick import prompt
from slick.providers import Provider


def checked(text):
    if not text.strip():
        raise ValueError(f"empty generated text; response={text!r}")
    return text.strip()


def distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        following = [i]
        for j, b in enumerate(right, 1):
            following.append(min(following[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = following
    return row[-1]


class Suggestions(BaseModel):
    suggestions: list[str]


@dataclass
class Cluster:
    representative: str
    size: int
    examples: list[dict]


@dataclass
class Evaluation:
    score: float
    failures: list[dict] = field(default_factory=list)


@dataclass
class Candidate:
    instruction: str
    prompt: str
    evaluation: Evaluation


class MAPS:
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        cluster: Callable[[Sequence[dict]], Sequence[Cluster]],
        domain_context: str = "",
    ):
        self.task, self.provider = task, provider
        self.evaluate, self.cluster, self.domain_context = evaluate, cluster, domain_context

    @prompt(template="suggest.j2", output_type=Suggestions)
    async def suggest(self, prompts: list[str], count: int, *, generated: Suggestions) -> list[str]:
        suggestions = [checked(item) for item in generated.suggestions]
        if len(suggestions) != count or len(set(suggestions)) != count:
            raise ValueError(f"expected {count} distinct mutation suggestions: {generated!r}")
        return suggestions

    @prompt(template="mutate.j2")
    async def mutate(self, prompts: list[str], suggestion: str, *, generated: str) -> str:
        return checked(generated)

    @prompt(template="reflect.j2")
    async def reflect(self, examples: list[dict], *, generated: str) -> str:
        return checked(generated)

    @prompt(template="rule.j2")
    async def induce_rule(self, reflection: str, *, generated: str) -> str:
        return checked(generated)

    def render(self, instruction, rules):
        return "\n\n".join(
            part for part in [instruction, self.domain_context, "\n".join(rules)] if part
        )

    async def _measure(self, instruction, rules):
        text = self.render(instruction, rules)
        if text not in self.archive:
            self.evaluations += 1
            result = await self.evaluate(text)
            if not math.isfinite(result.score):
                raise ValueError("fitness must be finite")
            self.archive[text] = Candidate(instruction, text, result)
        return self.archive[text]

    def _select_cluster(self, failures):
        clusters = list(self.cluster(failures))
        weights = []
        for cluster in clusters:
            novelty = min(
                (
                    distance(cluster.representative, old)
                    / max(1, len(cluster.representative) + len(old))
                    for old in self.tried
                ),
                default=1,
            )
            weights.append(cluster.size * novelty)
        if not clusters or not any(weights):
            return None, weights
        return self.rng.choices(clusters, weights=weights, k=1)[0], weights

    async def _rules(self, parents, count):
        failures = [row for parent in parents for row in parent.evaluation.failures]
        cluster, weights = self._select_cluster(failures)
        if cluster is None:
            return
        self.optimizer_calls += 1
        reflection = await self.reflect(cluster.examples, provider=self.provider)
        proposals = []
        for _ in range(count):
            self.optimizer_calls += 1
            rule = await self.induce_rule(reflection, provider=self.provider)
            self.tried.append(cluster.representative)
            measured = await self._measure(parents[0].instruction, [*self.rules, rule])
            proposals.append((rule, measured))
        if not proposals:
            return
        rule, candidate = max(proposals, key=lambda pair: pair[1].evaluation.score)
        accepted = candidate.evaluation.score > parents[0].evaluation.score
        if accepted:
            self.rules.append(rule)
        self.history.append(
            {
                "cluster": cluster,
                "weights": weights,
                "rule": rule,
                "accepted": accepted,
                "proposals": proposals,
            }
        )

    async def run(
        self,
        initial_prompts: Sequence[str],
        *,
        iterations: int = 5,
        beam_size: int = 3,
        mutations: int = 3,
        rule_candidates: int = 3,
        seed: int = 0,
    ) -> dict:
        self.rng, self.archive = random.Random(seed), {}
        self.rules, self.tried, self.history = [], [], []
        self.evaluations = self.optimizer_calls = 0
        population = [await self._measure(text, []) for text in dict.fromkeys(initial_prompts)]
        for _ in range(iterations):
            parents = sorted(population, key=lambda item: item.evaluation.score, reverse=True)[
                :beam_size
            ]
            prompts = [item.instruction for item in parents]
            self.optimizer_calls += 1
            suggestions = await self.suggest(prompts, mutations, provider=self.provider)
            children = []
            for suggestion in suggestions:
                self.optimizer_calls += 1
                children.append(await self.mutate(prompts, suggestion, provider=self.provider))
            await self._rules(parents, rule_candidates)
            population = [
                await self._measure(text, self.rules)
                for text in dict.fromkeys([*prompts, *children])
            ]
        return {
            "best": max(population, key=lambda item: item.evaluation.score),
            "population": population,
            "rules": self.rules,
            "tried": self.tried,
            "history": self.history,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
        }
