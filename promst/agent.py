"""Search prompt ancestry with categorized execution feedback and learned screening."""

import math
import random
import statistics
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from slick import prompt
from slick.providers import Provider


@dataclass
class Feedback:
    category: str
    text: str


@dataclass
class Evaluation:
    score: float
    feedback: Sequence[Feedback] = ()


@dataclass
class Heuristic:
    predict: Callable[[str], Awaitable[Sequence[float]]]
    errors: Sequence[float]


@dataclass
class Candidate:
    prompt: str
    score: float
    feedback: Sequence[Feedback]
    ancestors: list[str] = field(default_factory=list)


def checked(text):
    if not text.strip():
        raise ValueError(f"empty generated text; response={text!r}")
    return text.strip()


class PROMST:
    """Inject execution with human-designed feedback rules and optional model training.

    fit(history) returns an ensemble predictor and its held-out model errors.
    Search uses mean prediction + population variance + mean error as the paper's
    admission bound, and never replaces real fitness with predicted fitness.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        fit: Callable[[Sequence[Candidate]], Awaitable[Heuristic | None]] | None = None,
    ):
        self.task, self.provider, self.evaluate, self.fit = task, provider, evaluate, fit
        self.responses = []

    @prompt(template="summarize_feedback.j2")
    async def summarize_feedback(
        self, instruction: str, category: str, feedback: list[str], *, generated: str
    ) -> str:
        self.responses.append({"operation": "summarize_feedback", "response": generated})
        return checked(generated)

    @prompt(template="revise.j2")
    async def revise(
        self, instruction: str, feedback: list[dict], ancestors: list[str], *, generated: str
    ) -> str:
        self.responses.append({"operation": "revise", "response": generated})
        return checked(generated)

    async def _score(self, text, ancestors):
        self.evaluations += 1
        result = await self.evaluate(text)
        score = float(result.score)
        if not math.isfinite(score):
            raise ValueError("fitness must be finite")
        return Candidate(text, score, result.feedback, ancestors)

    async def _propose(self, parent, feedback_count):
        sampled = self.rng.sample(list(parent.feedback), min(feedback_count, len(parent.feedback)))
        groups = {}
        for item in sampled:
            groups.setdefault(item.category, []).append(item.text)
        summaries = []
        for category, feedback in groups.items():
            self.optimizer_calls += 1
            summary = await self.summarize_feedback(
                parent.prompt, category, feedback, provider=self.provider
            )
            summaries.append({"category": category, "summary": summary})
        self.optimizer_calls += 1
        return await self.revise(
            parent.prompt, summaries, parent.ancestors + [parent.prompt], provider=self.provider
        )

    async def _screen(self, text, model, threshold):
        self.prediction_calls += 1
        predictions = [float(value) for value in await model.predict(text)]
        errors = [float(value) for value in model.errors]
        if not predictions or not errors:
            raise ValueError("heuristic predictions and measured errors must be nonempty")
        if not all(math.isfinite(value) for value in predictions + errors):
            raise ValueError("heuristic predictions and measured errors must be finite")
        bound = (
            statistics.mean(predictions)
            + statistics.pvariance(predictions)
            + statistics.mean(errors)
        )
        return bound >= threshold, bound

    async def _expand(self, parent, depth, children, feedback_count, model, threshold):
        if not parent.feedback:
            return []
        proposed, seen = [], set(self.archive)
        budget = 3 * children if model is not None else children
        for _ in range(budget):
            self.attempts += 1
            record = {
                "parent": parent.prompt,
                "prompt": None,
                "depth": depth,
                "rejection": "generation_pending",
            }
            self.history.append(record)
            text = await self._propose(parent, feedback_count)
            record.update(prompt=text, rejection="duplicate" if text in seen else "")
            if text in seen:
                continue
            seen.add(text)
            if model is not None:
                admitted, bound = await self._screen(text, model, threshold)
                record.update(bound=bound, threshold=threshold)
                if not admitted:
                    record["rejection"] = "heuristic"
                    continue
            proposed.append(text)
            if len(proposed) == children:
                break
        return proposed

    async def _generation(
        self, beam, level, children, score_start, feedback_count, threshold_factor
    ):
        model = None
        if self.fit is not None and level >= score_start and any(p.feedback for p in beam):
            self.fit_calls += 1
            model = await self.fit(tuple(self.archive.values()))
        threshold = threshold_factor * max(item.score for item in self.archive.values())
        for parent in beam:
            proposals = await self._expand(
                parent, level, children, feedback_count, model, threshold
            )
            for text in proposals:
                self.archive[text] = await self._score(text, parent.ancestors + [parent.prompt])

    async def run(
        self,
        initial_prompt: str,
        *,
        depth: int = 20,
        beam_size: int = 5,
        children: int = 8,
        first_children: int = 20,
        patience: int | None = 3,
        score_start: int = 4,
        feedback_count: int = 10,
        threshold_factor: float = 0.8,
        seed: int = 0,
    ) -> dict:
        self.rng, self.history, self.responses = random.Random(seed), [], []
        self.evaluations = self.optimizer_calls = self.prediction_calls = self.fit_calls = 0
        self.attempts = 0
        initial = await self._score(initial_prompt, [])
        self.archive = {initial.prompt: initial}
        beam = [initial]
        generation_best, stale = [initial.score], 0
        stop_reason = "depth"
        for level in range(1, depth):
            await self._generation(
                beam,
                level,
                first_children if level == 1 else children,
                score_start,
                feedback_count,
                threshold_factor,
            )
            beam = sorted(self.archive.values(), key=lambda item: item.score, reverse=True)[
                :beam_size
            ]
            stale = 0 if beam[0].score > generation_best[-1] else stale + 1
            generation_best.append(beam[0].score)
            if not any(parent.feedback for parent in beam):
                stop_reason = "exhausted"
                break
            if patience is not None and stale >= patience:
                stop_reason = "stagnation"
                break
        return {
            "best": beam[0],
            "beam": beam,
            "archive": list(self.archive.values()),
            "history": self.history.copy(),
            "responses": self.responses.copy(),
            "generation_best": generation_best,
            "stop_reason": stop_reason,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "attempts": self.attempts,
            "prediction_calls": self.prediction_calls,
            "fit_calls": self.fit_calls,
        }
