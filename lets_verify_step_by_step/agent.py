"""Fixed-generator best-of-N search and active process supervision."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from slick import prompt
from slick.providers import Provider

Label = Literal[-1, 0, 1]


@dataclass(frozen=True)
class StepProbabilities:
    positive: float
    neutral: float
    negative: float


@dataclass(frozen=True)
class Solution:
    text: str
    steps: tuple[str, ...]


@dataclass(frozen=True)
class ScoredSolution:
    solution: Solution
    probabilities: tuple[StepProbabilities, ...]
    log_score: float

    @property
    def score(self) -> float:
        return math.exp(self.log_score)


@dataclass(frozen=True)
class Result:
    best: ScoredSolution
    samples: tuple[ScoredSolution, ...]
    calls: int
    correct: bool | None


@dataclass(frozen=True)
class TrainingExample:
    problem: str
    steps: tuple[str, ...]
    label: Label


Reward = Callable[[str, tuple[str, ...]], Awaitable[Sequence[StepProbabilities]]]
Evaluate = Callable[[str, Solution], Awaitable[bool]]
LabelSteps = Callable[[str, tuple[str, ...]], Awaitable[Sequence[Label]]]
Fit = Callable[[tuple[TrainingExample, ...], int], Awaitable[Reward]]


def checked_probabilities(scores: Sequence[StepProbabilities]) -> tuple[StepProbabilities, ...]:
    """Check measured reward outputs, not caller configuration."""
    result = tuple(scores)
    if not result:
        raise ValueError("The reward model returned no step probabilities")
    for score in result:
        values = (score.positive, score.neutral, score.negative)
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise ValueError("Reward probabilities must be finite and between zero and one")
        if not math.isclose(sum(values), 1, abs_tol=1e-6):
            raise ValueError("Each step's label probabilities must sum to one")
    return result


def process_examples(
    problem: str, steps: Sequence[str], labels: Sequence[Label]
) -> tuple[TrainingExample, ...]:
    """Supervise each labeled prefix through the first negative, inclusive."""
    if not labels or len(labels) > len(steps):
        raise ValueError("Labels must cover a nonempty solution prefix")
    examples = []
    for index, label in enumerate(labels):
        if label not in (-1, 0, 1):
            raise ValueError("Step labels must be -1, 0, or 1")
        examples.append(TrainingExample(problem, tuple(steps[: index + 1]), label))
        if label == -1:
            return tuple(examples)
    if len(labels) != len(steps):
        raise ValueError("An unfinished label sequence must end at the first error")
    return tuple(examples)


def synthetic_labels(
    scores: Sequence[StepProbabilities], *, threshold: float = 0.2
) -> tuple[Label, ...]:
    """Appendix H: negative probability strictly above the threshold means error."""
    labels = []
    for score in checked_probabilities(scores):
        label = -1 if score.negative > threshold else 1
        labels.append(label)
        if label == -1:
            break
    return tuple(labels)


def synthetic_outcome(scores: Sequence[StepProbabilities], *, threshold: float = 0.2) -> bool:
    """The same teacher marks a whole solution correct only if every step passes."""
    return -1 not in synthetic_labels(scores, threshold=threshold)


class VerifyStepByStep:
    """Own sampling, reward ranking, and optional data collection/training.

    reward(problem, steps) must return prefix-causal label probabilities for every
    step, including the final answer. It never receives a reference answer.
    evaluate owns task-specific final-answer extraction and grading. Providers
    remain fixed throughout learning; fit replaces only the reward callback.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        reward: Reward,
        evaluate: Evaluate | None = None,
        *,
        neutral_is_correct: bool = True,
        reduction: Literal["product", "minimum"] = "product",
    ):
        self.task = task
        self.provider = provider
        self.reward = reward
        self.evaluate = evaluate
        self.neutral_is_correct = neutral_is_correct
        self.reduction = reduction
        self.calls: list[dict] = []
        self.samples: list[ScoredSolution] = []
        self.training: list[TrainingExample] = []

    @prompt(template="generate.j2")
    async def generate(self, problem: str, *, generated: str) -> Solution:
        """Preserve raw text and split the paper's newline-delimited steps."""
        steps = tuple(line for line in generated.splitlines() if line.strip())
        if not steps:
            raise ValueError("Generated solution contains no steps")
        return Solution(generated, steps)

    async def run(self, problem: str, *, n: int = 16) -> Result:
        """Sample n independent solutions; evaluate only after selecting the best.

        Ties keep the earliest sample. No generation repair, early stopping,
        shared Session, or generator updates. Failures are logged and propagated.
        """
        self.calls, self.samples = [], []
        await self.sample_solutions(problem, n)
        best = max(self.samples, key=lambda sample: sample.log_score)
        correct = None
        if self.evaluate is not None:
            correct = await self.evaluate(problem, best.solution)
        return Result(best, tuple(self.samples), len(self.calls), correct)

    async def sample_solutions(self, problem: str, n: int) -> tuple[ScoredSolution, ...]:
        """Append exactly n independent calls and their scored solutions on success."""
        pool = []
        for _ in range(n):
            record = {"operation": "generate", "problem": problem}
            self.calls.append(record)
            try:
                solution = await self.generate(problem, provider=self)
                probabilities = checked_probabilities(await self.reward(problem, solution.steps))
                if len(probabilities) != len(solution.steps):
                    raise ValueError("Reward model must score every step, including the answer")
                correctness = [
                    min(1.0, p.positive + (p.neutral if self.neutral_is_correct else 0))
                    for p in probabilities
                ]
                logs = [math.log(p) if p else -math.inf for p in correctness]
                log_score = math.fsum(logs) if self.reduction == "product" else min(logs)
                sample = ScoredSolution(solution, probabilities, log_score)
                pool.append(sample)
                self.samples.append(sample)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
        return tuple(pool)

    async def select_for_labeling(
        self, problem: str, pool: Sequence[ScoredSolution], k: int, *, wrong_fraction: float = 0.8
    ) -> tuple[ScoredSolution, ...]:
        """Section 4.2: top wrong answers, then top remaining of either outcome.

        Requires evaluate. Floor the wrong-answer quota; shortages are filled
        from the remaining ranked pool. Distinct sample positions are retained,
        even when the generator emits identical text.
        """
        ranked = sorted(range(len(pool)), key=lambda i: -pool[i].log_score)
        correct = [await self.evaluate(problem, sample.solution) for sample in pool]
        selected = [i for i in ranked if not correct[i]][: int(k * wrong_fraction)]
        used = set(selected)
        selected.extend(i for i in ranked if i not in used)
        return tuple(pool[i] for i in selected[:k])

    async def learn(
        self,
        problems: Sequence[str],
        label: LabelSteps,
        fit: Fit,
        *,
        pool_size: int = 1000,
        k: int = 10,
        rounds: int = 1,
        wrong_fraction: float = 0.8,
        epochs: int = 2,
    ) -> tuple[TrainingExample, ...]:
        """Collect labels, then fit a reward model to all accumulated prefixes.

        Start with a seeded selector. label can be human or a separate teacher;
        it may stop at the first negative. fit(examples, epochs) owns tokenizer,
        checkpoint, optimizer and label-token-only cross entropy. It returns the
        trained reward callable. Use one round for the Section 4.2 ablation;
        extra rounds opt into iterative selection from Section 2.4.
        """
        self.calls, self.samples, self.training = [], [], []
        for _ in range(rounds):
            await self.collect_labels(problems, label, pool_size, k, wrong_fraction)
            self.reward = await fit(tuple(self.training), epochs)
        return tuple(self.training)

    async def collect_labels(
        self,
        problems: Sequence[str],
        label: LabelSteps,
        pool_size: int,
        k: int,
        wrong_fraction: float,
    ) -> None:
        """Keep selector weights fixed until all problems in the round are labeled."""
        for problem in problems:
            pool = await self.sample_solutions(problem, pool_size)
            selected = await self.select_for_labeling(
                problem, pool, k, wrong_fraction=wrong_fraction
            )
            for sample in selected:
                labels = await label(problem, sample.solution.steps)
                self.training.extend(process_examples(problem, sample.solution.steps, labels))

    async def acall(self, context, *, tools=None, tool_results=None):
        """Record raw responses before postprocessing; sequential calls only."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("The fixed generator must return text without tool requests")
        return response, requests
