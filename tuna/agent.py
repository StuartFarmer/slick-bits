"""Tune a student sequentially on teacher-likelihood and contextual response rankings."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field, StrictInt
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Example:
    instruction: str
    reference: str


@dataclass(frozen=True)
class TeacherResponse:
    text: str
    token_logprobs: np.ndarray


class Ranking(BaseModel, extra="forbid"):
    order: list[StrictInt] = Field(min_length=1)


Teacher = Callable[[str, int], Awaitable[TeacherResponse]]
Generate = Callable[[np.ndarray, str, float, int], Awaitable[str]]
Forward = Callable[[np.ndarray, str, str], Awaitable[tuple[np.ndarray, np.ndarray]]]


def ranking_loss(scores: np.ndarray, reference: float, margin: float, mle_weight: float):
    """Paper's all-pairs rank-gap hinge plus negative reference mean log likelihood."""
    loss, gradient = -mle_weight * reference, np.zeros_like(scores)
    for better in range(len(scores)):
        for worse in range(better + 1, len(scores)):
            violation = scores[worse] - scores[better] + margin * (worse - better)
            if violation > 0:
                loss += violation
                gradient[better] -= 1
                gradient[worse] += 1
    return float(loss), gradient, -mle_weight


def rouge_l(first: str, second: str) -> float:
    a, b = first.split(), second.split()
    row = np.zeros(len(b) + 1, dtype=int)
    for token in a:
        previous = row.copy()
        for j, other in enumerate(b, 1):
            row[j] = previous[j - 1] + 1 if token == other else max(previous[j], row[j - 1])
    return 2 * row[-1] / (len(a) + len(b))


class TUNA:
    """Model generation/likelihood are primitives; rankings, losses and Adam are local."""

    def __init__(
        self, task: str, provider: Provider, teacher: Teacher, generate: Generate, forward: Forward
    ):
        self.task, self.provider = task, provider
        self.teacher, self.generate, self.forward = teacher, generate, forward

    @prompt(template="rank.j2", output_type=Ranking)
    async def rank(
        self, instruction: str, responses: list[str], *, generated: Ranking
    ) -> tuple[int, ...]:
        if sorted(generated.order) != list(range(len(responses))):
            raise ValueError("Ranking must contain each candidate ID exactly once")
        return tuple(generated.order)

    async def run(
        self,
        parameters: np.ndarray,
        examples: Sequence[Example],
        *,
        candidates: int = 4,
        probabilistic_epochs: int = 1,
        contextual_epochs: int = 1,
        probabilistic_rate: float = 1e-5,
        contextual_rate: float = 1e-6,
        length_penalty: float = 1.3,
        margin: float = 0.1,
        mle_weight: float = 1.0,
        diversity_threshold: float = 0.8,
        resample_attempts: int = 3,
        seed: int = 0,
    ) -> dict:
        self.parameters = np.array(parameters, dtype=float, copy=True)
        self.rng = np.random.default_rng(seed)
        self.generations = 0
        probabilistic = await self.probabilistic_data(examples, candidates, length_penalty)
        first_losses = await self.fit(
            probabilistic, probabilistic_epochs, probabilistic_rate, margin, mle_weight
        )
        contextual = await self.contextual_data(
            examples, candidates, diversity_threshold, resample_attempts
        )
        second_losses = await self.fit(
            contextual, contextual_epochs, contextual_rate, margin, mle_weight
        )
        return {
            "parameters": self.parameters.copy(),
            "probabilistic_rankings": probabilistic,
            "contextual_rankings": contextual,
            "probabilistic_losses": first_losses,
            "contextual_losses": second_losses,
            "generations": self.generations,
            "ranking_calls": len(examples),
        }

    async def probabilistic_data(self, examples, count, penalty):
        records = []
        for example in examples:
            scored = []
            for _ in range(count):
                response = await self.teacher(example.instruction, int(self.rng.integers(2**31)))
                self.generations += 1
                logs = np.array(response.token_logprobs, copy=True)
                if not response.text.strip() or len(logs) == 0 or not np.isfinite(logs).all():
                    raise ValueError("Invalid teacher response or log probabilities")
                scored.append((float(logs.sum() / len(logs) ** penalty), response.text))
            responses = [text for _, text in sorted(scored, key=lambda pair: pair[0], reverse=True)]
            records.append((example, responses))
        return records

    async def contextual_data(self, examples, count, threshold, attempts):
        records = []
        for example in examples:
            responses = []
            for _ in range(count):
                trials = []
                for attempt in range(attempts):
                    text = await self.generate(
                        self.parameters.copy(),
                        example.instruction,
                        1.0 + 0.1 * attempt,
                        int(self.rng.integers(2**31)),
                    )
                    self.generations += 1
                    if not text.strip():
                        raise ValueError("Empty student response")
                    similarity = max(
                        (rouge_l(text, previous) for previous in responses), default=0.0
                    )
                    trials.append((similarity, text))
                    if similarity < threshold:
                        break
                responses.append(min(trials, key=lambda trial: trial[0])[1])
            order = await self.rank(example.instruction, responses, provider=self.provider)
            records.append((example, [responses[index] for index in order]))
        return records

    async def likelihood(self, instruction, response):
        logs, jacobian = await self.forward(self.parameters.copy(), instruction, response)
        if len(logs) == 0 or not np.isfinite(logs).all() or not np.isfinite(jacobian).all():
            raise ValueError("Invalid response likelihood or Jacobian")
        return float(np.mean(logs)), np.mean(jacobian, axis=0)

    async def fit(self, records, epochs, rate, margin, mle_weight):
        moment, variance, step, losses = (
            np.zeros_like(self.parameters),
            np.zeros_like(self.parameters),
            0,
            [],
        )
        for _ in range(epochs):
            for index in self.rng.permutation(len(records)):
                example, responses = records[index]
                likelihoods = [
                    await self.likelihood(example.instruction, response) for response in responses
                ]
                scores = np.array([item[0] for item in likelihoods])
                jacobian = np.array([item[1] for item in likelihoods])
                reference, reference_jacobian = await self.likelihood(
                    example.instruction, example.reference
                )
                loss, derivative, dr = ranking_loss(scores, reference, margin, mle_weight)
                gradient = derivative @ jacobian + dr * reference_jacobian
                if not np.isfinite(loss) or not np.isfinite(gradient).all():
                    raise ValueError("Nonfinite ranking objective")
                step += 1
                moment = 0.9 * moment + 0.1 * gradient
                variance = 0.999 * variance + 0.001 * gradient**2
                self.parameters -= (
                    rate
                    * (moment / (1 - 0.9**step))
                    / (np.sqrt(variance / (1 - 0.999**step)) + 1e-8)
                )
                if not np.isfinite(self.parameters).all():
                    raise ValueError("Nonfinite tuned parameters")
                losses.append(loss)
        return losses
