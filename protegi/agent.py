"""Optimize task instructions with textual gradients, beam search and bandit selection."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from slick import prompt
from slick.providers import Provider

Example = TypeVar("Example")
Selection = Literal["s-sr", "ucb", "ucb-e", "sr", "sh", "uniform"]


@dataclass(frozen=True)
class Evaluation:
    """Task metric and mistake descriptions for exactly the supplied examples."""

    score: float
    errors: tuple[str, ...] = ()


def tagged_items(response: str, count: int) -> list[str]:
    items = [s.strip() for s in re.findall(r"<START>(.*?)<END>", response, re.DOTALL)]
    if (
        len(items) != count
        or response.count("<START>") != count
        or response.count("<END>") != count
        or not all(items)
    ):
        raise ValueError(f"expected {count} nonempty <START>...<END> items; response={response!r}")
    return items


class ProTeGi(Generic[Example]):
    """Keep search, sampling and beam decisions local; delegate only task evaluation.

    Evaluate returns a finite, higher-is-better metric and error descriptions for
    exactly the supplied batch. Bandits estimate example-weighted batch rewards;
    for non-additive metrics such as F1 this is an approximation, not pooled F1.
    Errors propagate; retries and execution isolation belong to the caller.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str, Sequence[Example]], Awaitable[Evaluation]],
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate

    @prompt(template="gradients.j2")
    async def gradients(
        self, instruction: str, errors: Sequence[str], count: int, *, generated: str
    ) -> list[str]:
        """Generate textual reasons for failures on a sampled error batch."""
        return tagged_items(generated, count)

    @prompt(template="revise.j2")
    async def revise(
        self, instruction: str, errors: Sequence[str], gradient: str, count: int, *, generated: str
    ) -> list[str]:
        """Move opposite to one textual gradient."""
        return tagged_items(generated, count)

    @prompt(template="paraphrase.j2")
    async def paraphrase(self, instruction: str, *, generated: str) -> str:
        """Sample a semantic-preserving Monte Carlo variation."""
        if not generated.strip():
            raise ValueError(f"empty paraphrase; response={generated!r}")
        return generated.strip()

    async def _assess(self, instruction, examples, phase):
        self.evaluations += 1
        self.example_evaluations += len(examples)
        assessment = await self.evaluate(instruction, examples)
        if not math.isfinite(assessment.score):
            raise ValueError("evaluation score must be finite")
        self.measurements.append(
            {
                "prompt": instruction,
                "score": assessment.score,
                "phase": phase,
                "size": len(examples),
            }
        )
        return assessment

    async def _expand(
        self,
        beam,
        examples,
        minibatch_size,
        gradient_batches,
        gradients_per_batch,
        errors_per_gradient,
        steps_per_gradient,
        paraphrases,
        max_expansion,
    ):
        batch = self.rng.sample(list(examples), min(len(examples), minibatch_size))
        candidates = list(beam)
        for instruction in beam:
            assessment = await self._assess(instruction, batch, "gradient")
            revisions = []
            for _ in range(gradient_batches):
                errors = self.rng.sample(
                    list(assessment.errors), min(len(assessment.errors), errors_per_gradient)
                )
                self.optimizer_calls += 1
                gradients = await self.gradients(
                    instruction, errors, gradients_per_batch, provider=self.provider
                )
                for gradient in gradients:
                    self.optimizer_calls += 1
                    revisions.extend(
                        await self.revise(
                            instruction,
                            errors,
                            gradient,
                            steps_per_gradient,
                            provider=self.provider,
                        )
                    )
            variants = []
            for parent in revisions + [instruction]:
                for _ in range(paraphrases):
                    self.optimizer_calls += 1
                    variants.append(await self.paraphrase(parent, provider=self.provider))
            expanded = list(dict.fromkeys(revisions + variants))
            if len(expanded) > max_expansion:
                expanded = self.rng.sample(expanded, max_expansion)
            candidates.extend(expanded)
        return list(dict.fromkeys(candidates))

    async def _successive_rejects(self, candidates, examples, beam_size, budget, prompts_per_round):
        survivors = list(candidates)
        rejections = []
        rounds = len(survivors) - beam_size
        if rounds <= 0:
            return survivors, rejections
        sample_size = math.ceil(budget / (rounds * prompts_per_round))
        for _ in range(rounds):
            batch = self.rng.sample(list(examples), min(len(examples), sample_size))
            selected = self.rng.sample(survivors, min(len(survivors), prompts_per_round))
            scores = [(await self._assess(p, batch, "selection")).score for p in selected]
            rejected = selected[scores.index(min(scores))]
            survivors.remove(rejected)
            rejections.append(rejected)
        return survivors, rejections

    async def _bandits(self, candidates, examples, beam_size, budget, samples, exploration, mode):
        means = dict.fromkeys(candidates, 0.0)
        counts = dict.fromkeys(candidates, 0)
        remaining = budget
        step = 1
        while remaining > 0:
            unseen = [p for p in candidates if counts[p] == 0]
            if unseen:
                chosen = unseen[0]
                # Reserve at least one observation for every remaining unseen arm.
                size = min(samples, len(examples), remaining - len(unseen) + 1)
            else:
                factor = math.log(step) if mode == "ucb" else exploration
                chosen = max(
                    candidates,
                    key=lambda p: means[p] + exploration * math.sqrt(factor / counts[p]),
                )
                size = min(samples, len(examples), remaining)
            if size <= 0:
                raise RuntimeError("selection budget cannot sample every candidate")
            batch = self.rng.sample(list(examples), size)
            reward = (await self._assess(chosen, batch, "selection")).score
            counts[chosen] += size
            weight = size / counts[chosen]
            means[chosen] = means[chosen] * (1 - weight) + reward * weight
            remaining -= size
            step += 1
        if not all(counts.values()):
            raise RuntimeError("selection budget cannot sample every candidate")
        ranked = sorted(candidates, key=means.__getitem__, reverse=True)
        return ranked[:beam_size], ranked[beam_size:]

    async def _eliminate(self, candidates, examples, beam_size, budget, mode):
        survivors = list(candidates)
        rejected = []
        means = dict.fromkeys(candidates, 0.0)
        counts = dict.fromkeys(candidates, 0)
        n = len(candidates)
        harmonic = 0.5 + sum(1 / i for i in range(2, n + 1))
        halving_rounds = math.ceil(math.log2(n / beam_size))
        remaining, previous_target, phase = budget, 0, 1
        while len(survivors) > beam_size:
            if mode == "sr":
                target = math.ceil((budget - n) / (harmonic * (n + 1 - phase)))
                size = max(1, target - previous_target)
                previous_target = target
                keep = len(survivors) - 1
            elif mode == "sh":
                size = max(1, budget // (len(survivors) * halving_rounds))
                keep = max(beam_size, math.ceil(len(survivors) / 2))
            else:  # uniform allocation
                size = budget // n
                keep = beam_size
            size = min(size, len(examples), remaining // len(survivors))
            if size <= 0:
                if not all(counts[p] for p in survivors):
                    raise RuntimeError("selection budget cannot sample every candidate")
                # No complete comparison fits: finish using measured estimates.
                ranked = sorted(survivors, key=means.__getitem__, reverse=True)
                rejected.extend(ranked[beam_size:])
                return ranked[:beam_size], rejected
            batch = self.rng.sample(list(examples), size)
            for p in survivors:
                reward = (await self._assess(p, batch, "selection")).score
                counts[p] += size
                weight = size / counts[p]
                means[p] = means[p] * (1 - weight) + reward * weight
                remaining -= size
            ranked = sorted(survivors, key=means.__getitem__, reverse=True)
            rejected.extend(ranked[keep:])
            survivors = ranked[:keep]
            phase += 1
        return survivors, rejected

    async def _select(
        self,
        candidates,
        examples,
        beam_size,
        budget,
        prompts_per_round,
        selection,
        samples_per_eval,
        exploration,
    ):
        if len(candidates) <= beam_size:
            return list(candidates), []
        if selection == "s-sr":
            return await self._successive_rejects(
                candidates, examples, beam_size, budget, prompts_per_round
            )
        if selection in {"ucb", "ucb-e"}:
            return await self._bandits(
                candidates, examples, beam_size, budget, samples_per_eval, exploration, selection
            )
        return await self._eliminate(candidates, examples, beam_size, budget, selection)

    async def run(
        self,
        initial_prompts: Sequence[str],
        examples: Sequence[Example],
        *,
        iterations: int = 3,
        beam_size: int = 4,
        minibatch_size: int = 64,
        gradient_batches: int = 1,
        gradients_per_batch: int = 4,
        errors_per_gradient: int = 4,
        steps_per_gradient: int = 1,
        paraphrases: int = 2,
        max_expansion: int = 8,
        selection_budget: int = 100,
        prompts_per_round: int = 4,
        selection: Selection = "s-sr",
        samples_per_eval: int = 5,
        exploration: float = 2.0,
        validation_examples: Sequence[Example] | None = None,
        seed: int = 0,
    ) -> dict:
        """Expand and reject down to the beam, then measure final survivors together.

        selection_budget bounds per-step selection example evaluations, except
        legacy s-sr uses a nominal budget whose ceilings may exceed it. Gradient
        evaluation and final ranking are counted separately. Validation examples
        are used only to rank the final beam; absent them, use all training data.
        Use a separate test set to measure the chosen prompt's generalization.
        """
        self.rng = random.Random(seed)
        self.evaluations = self.example_evaluations = self.optimizer_calls = 0
        self.measurements = []
        beam = list(dict.fromkeys(initial_prompts))
        history = []
        for _ in range(iterations):
            candidates = await self._expand(
                beam,
                examples,
                minibatch_size,
                gradient_batches,
                gradients_per_batch,
                errors_per_gradient,
                steps_per_gradient,
                paraphrases,
                max_expansion,
            )
            beam, rejected = await self._select(
                candidates,
                examples,
                beam_size,
                selection_budget,
                prompts_per_round,
                selection,
                samples_per_eval,
                exploration,
            )
            history.append({"beam": beam.copy(), "rejected": rejected})
        final_examples = examples if validation_examples is None else validation_examples
        population = [
            {"prompt": p, "score": (await self._assess(p, final_examples, "final")).score}
            for p in beam
        ]
        return {
            "best": max(population, key=lambda p: p["score"]),
            "population": population,
            "history": history,
            "measurements": self.measurements,
            "evaluations": self.evaluations,
            "example_evaluations": self.example_evaluations,
            "optimizer_calls": self.optimizer_calls,
        }
