"""Exemplar-guided reflection with rewarded feedback and exemplar memories."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel
from slick import prompt
from slick.providers import Provider


def checked(text):
    if not text.strip():
        raise ValueError(f"empty generated text; response={text!r}")
    return text.strip()


class Exemplar(BaseModel):
    question: str
    answer: str
    rationale: str


class Exemplars(BaseModel):
    exemplars: list[Exemplar]


class Feedbacks(BaseModel):
    feedbacks: list[str]


@dataclass
class Memory:
    content: str | Exemplar
    priority: float = 1.0


@dataclass
class Candidate:
    prompt: str
    score: float
    exemplars: dict[str, list[Exemplar]]


class ERM:
    """The evaluator receives query-specific exemplars alongside the instruction.

    Queries are caller-chosen validation inputs (or stable serialized input keys).
    The evaluator renders each query with its associated exemplars; {} means none.
    Similarity and solution verification are caller-owned semantic operations.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str, dict[str, list[Exemplar]]], Awaitable[float]],
        failures: Callable[[str], Awaitable[Sequence[dict]]],
        verify: Callable[[Exemplar, dict], Awaitable[bool]],
        similarity: Callable[[str, str], float],
        queries: Sequence[str],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.failures, self.verify, self.similarity = failures, verify, similarity
        self.queries = queries

    @prompt(template="exemplars.j2", output_type=Exemplars)
    async def make_exemplars(
        self, instruction: str, failures: list[dict], *, generated: Exemplars
    ) -> list[Exemplar]:
        for exemplar in generated.exemplars:
            if not all(
                part.strip() for part in (exemplar.question, exemplar.answer, exemplar.rationale)
            ):
                raise ValueError(f"empty generated exemplar field: {generated!r}")
        return generated.exemplars

    @prompt(template="reflect.j2", output_type=Feedbacks)
    async def reflect(
        self, instruction: str, failures: list[dict], exemplars: list[dict], *, generated: Feedbacks
    ) -> list[str]:
        return [checked(item) for item in generated.feedbacks]

    @prompt(template="revise.j2")
    async def revise(
        self, instruction: str, failures: list[dict], feedback: str, *, generated: str
    ) -> str:
        return checked(generated)

    @prompt(template="revise_memory.j2")
    async def revise_memory(
        self, instruction: str, failures: list[dict], feedbacks: list[str], *, generated: str
    ) -> str:
        return checked(generated)

    def _sample(self, entries, logits, count):
        entries, logits = list(entries), list(logits)
        chosen = []
        for _ in range(min(count, len(entries))):
            maximum = max(logits)
            weights = [math.exp(value - maximum) for value in logits]
            index = self.rng.choices(range(len(entries)), weights=weights, k=1)[0]
            chosen.append(entries.pop(index))
            logits.pop(index)
        return chosen

    def retrieve(
        self, query: str, *, count: int = 5, training: bool = False, temperature: float = 1.0
    ) -> list[Exemplar]:
        values = [
            entry.priority * self.similarity(query, entry.content.question)
            for entry in self.exemplar_memory
        ]
        if training:
            chosen = self._sample(
                self.exemplar_memory, [value / temperature for value in values], count
            )
        else:
            chosen = [
                entry
                for _, entry in sorted(
                    zip(values, self.exemplar_memory), key=lambda pair: pair[0], reverse=True
                )[:count]
            ]
        return [entry.content.model_copy(deep=True) for entry in chosen]

    def _reward(self, entries, useful, beta, threshold):
        for entry in entries:
            entry.priority = (1 - beta) * entry.priority + beta * int(useful)
        self.feedback_memory = [
            entry for entry in self.feedback_memory if entry.priority >= threshold
        ]
        self.exemplar_memory = [
            entry for entry in self.exemplar_memory if entry.priority >= threshold
        ]

    async def _factory(self, parent, failures, replacement):
        self.optimizer_calls += 1
        exemplars = await self.make_exemplars(parent.prompt, failures, provider=self.provider)
        accepted = []
        by_question = {row["question"]: row for row in failures}
        for exemplar in exemplars:
            source = by_question.get(exemplar.question)
            valid = source is not None and exemplar.answer == source["answer"]
            if valid:
                valid = await self.verify(exemplar, source)
            duplicate = next(
                (
                    entry
                    for entry in self.exemplar_memory
                    if entry.content.question == exemplar.question
                ),
                None,
            )
            stored = valid and (duplicate is None or self.rng.random() < replacement)
            if valid:
                accepted.append(exemplar.model_dump())
            if stored:
                if duplicate is not None:
                    self.exemplar_memory.remove(duplicate)
                self.exemplar_memory.append(Memory(exemplar))
            self.factory_history.append({"exemplar": exemplar, "verified": valid, "stored": stored})
        return accepted

    async def _measure(self, text, exemplars):
        self.evaluations += 1
        score = float(await self.evaluate(text, exemplars))
        if not math.isfinite(score):
            raise ValueError("fitness must be finite")
        return Candidate(text, score, exemplars)

    async def _expand(
        self,
        parent,
        failures,
        exemplars,
        retrieval,
        use_memory,
        memory_count,
        temperature,
        threshold,
        beta,
        gain,
        similarity_cutoff,
    ):
        reference = await self._measure(parent.prompt, retrieval)
        self.optimizer_calls += 1
        feedbacks = await self.reflect(parent.prompt, failures, exemplars, provider=self.provider)
        children = [reference]
        # Retrieve before storing this round's new feedback: reuse historical memory.
        historical = (
            self._sample(
                self.feedback_memory,
                [entry.priority / temperature for entry in self.feedback_memory],
                memory_count,
            )
            if use_memory
            else []
        )
        for feedback in feedbacks:
            self.optimizer_calls += 1
            text = await self.revise(parent.prompt, failures, feedback, provider=self.provider)
            child = await self._measure(text, retrieval)
            useful = child.score - reference.score > gain
            redundant = any(
                self.similarity(feedback, entry.content) >= similarity_cutoff
                for entry in self.feedback_memory
            )
            if useful and not redundant:
                self.feedback_memory.append(Memory(feedback))
            self.feedback_history.append(
                {
                    "feedback": feedback,
                    "gain": child.score - reference.score,
                    "stored": useful and not redundant,
                }
            )
            children.append(child)
        if historical:
            self.optimizer_calls += 1
            text = await self.revise_memory(
                parent.prompt,
                failures,
                [entry.content for entry in historical],
                provider=self.provider,
            )
            child = await self._measure(text, retrieval)
            self._reward(historical, child.score - reference.score > gain, beta, threshold)
            children.append(child)
        return children

    async def run(
        self,
        initial_prompt: str,
        *,
        iterations: int = 5,
        beam_size: int = 2,
        batch_size: int = 8,
        memory_period: int = 1,
        memory_count: int = 2,
        exemplar_count: int = 5,
        temperature: float = 1.0,
        replacement: float = 0.5,
        beta: float = 0.5,
        threshold: float = 0.2,
        min_gain: float = 0,
        similarity_cutoff: float = 0.9,
        seed: int = 0,
    ) -> dict:
        self.rng = random.Random(seed)
        self.feedback_memory, self.exemplar_memory = [], []
        self.factory_history, self.feedback_history = [], []
        self.evaluations = self.optimizer_calls = 0
        best = await self._measure(initial_prompt, {})
        population = [best]
        for step in range(iterations):
            prepared, solved = [], []
            for parent in population:
                failures = list(await self.failures(parent.prompt))
                batch = self.rng.sample(failures, min(batch_size, len(failures)))
                if batch:
                    exemplars = await self._factory(parent, batch, replacement)
                    prepared.append((parent, batch, exemplars))
                else:
                    solved.append(parent)
            if not prepared:
                break
            retrieval = {
                query: self.retrieve(
                    query, count=exemplar_count, training=True, temperature=temperature
                )
                for query in self.queries
            }
            children = [await self._measure(parent.prompt, retrieval) for parent in solved]
            for parent, batch, exemplars in prepared:
                children.extend(
                    await self._expand(
                        parent,
                        batch,
                        exemplars,
                        retrieval,
                        (step + 1) % memory_period == 0,
                        memory_count,
                        temperature,
                        threshold,
                        beta,
                        min_gain,
                        similarity_cutoff,
                    )
                )
            population = sorted(children, key=lambda item: item.score, reverse=True)[:beam_size]
            current = population[0]
            if current.score > best.score:
                best = current
            baseline = await self._measure(current.prompt, {})
            used = {item.question for items in retrieval.values() for item in items}
            selected = [entry for entry in self.exemplar_memory if entry.content.question in used]
            self._reward(selected, current.score - baseline.score > min_gain, beta, threshold)
        return {
            "best": best,
            "population": population,
            "feedback_memory": self.feedback_memory,
            "exemplar_memory": self.exemplar_memory,
            "factory_history": self.factory_history,
            "feedback_history": self.feedback_history,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
        }
