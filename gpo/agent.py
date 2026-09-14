"""Gradient-inspired prompt optimization with relevant-history generation."""

import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from slick import prompt
from slick.providers import Provider


def unpack(text):
    match = re.search(r"<START>(.*?)<END>", text, re.DOTALL)
    if not match or not match[1].strip():
        raise ValueError(f"missing nonempty START/END proposal; response={text!r}")
    return match[1].strip()


@dataclass
class Candidate:
    prompt: str
    score: float


@dataclass
class Gradient:
    text: str
    gain: float


class GPO:
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        failures: Callable[[str], Awaitable[Sequence[dict]]] | None = None,
        *,
        embed: Callable[[Sequence[str]], Awaitable[np.ndarray]] | None = None,
    ):
        self.task, self.provider = task, provider
        self.evaluate, self.failures = evaluate, failures
        self.embed = embed

    @prompt(template="generate.j2")
    async def generate(
        self,
        instruction: str,
        history: list[Candidate],
        words: int,
        examples: Sequence[str],
        *,
        generated: str,
    ) -> str:
        return unpack(generated)

    @prompt(template="gradient.j2")
    async def gradient(self, instruction: str, failures: list[dict], *, generated: str) -> str:
        return unpack(generated)

    @prompt(template="edit.j2")
    async def edit(
        self,
        instruction: str,
        feedback: list[str],
        words: int,
        examples: list[dict],
        *,
        generated: str,
    ) -> str:
        return unpack(generated)

    @prompt(template="edit_history.j2")
    async def edit_history(
        self,
        instruction: str,
        feedback: str,
        history: list[Candidate],
        words: int,
        examples: list[dict],
        *,
        generated: str,
    ) -> str:
        return unpack(generated)

    def _momentum(self, items, size, selection, key):
        if selection == "importance":
            return sorted(items, key=key, reverse=True)[:size]
        return list(reversed(items[-size:])) if size else []

    async def _relevant_parameters(self, size):
        current, previous = self.trajectory[-1], self.trajectory[-2::-1]
        if not previous or size == 1:
            return [current]
        self.embedding_calls += 1
        vectors = np.asarray(
            await self.embed([current.prompt, *[p.prompt for p in previous]]), dtype=float
        )
        if not np.isfinite(vectors).all():
            raise ValueError("measured embeddings must be finite")
        scales = np.max(np.abs(vectors), axis=1, keepdims=True)
        vectors = vectors / np.where(scales == 0, 1, scales)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1, norms)
        order = np.argsort(-(vectors[1:] @ vectors[0]), kind="stable")[: size - 1]
        return [current, *[previous[i] for i in order]]

    def _words(self, step, total, initial, final, schedule, warmup):
        if step < warmup:
            return int(initial * step / warmup)
        progress = (step - warmup) / (total - warmup)
        if schedule == "linear":
            return int(initial + (final - initial) * progress)
        if schedule == "cosine":
            return int(final + (initial - final) * (1 + math.cos(math.pi * progress)) / 2)
        return initial

    async def _measure(self, text):
        if text not in self.archive:
            self.evaluations += 1
            score = float(await self.evaluate(text))
            if not math.isfinite(score):
                raise ValueError("fitness must be finite")
            self.archive[text] = Candidate(text, score)
        return self.archive[text]

    def _advance(self, current, pool, words, feedback, params):
        ranked = sorted(pool.values(), key=lambda item: item.score, reverse=True)
        unseen = [item for item in ranked if item.prompt not in self.visited]
        chosen = (unseen or ranked)[0]
        self.history.append(
            {
                "parent": current,
                "chosen": chosen,
                "words": words,
                "feedback": feedback,
                "parameters": params,
            }
        )
        self.visited.add(chosen.prompt)
        self.trajectory.append(chosen)
        return chosen

    async def _generation_step(self, current, candidates, words, selection, memory, examples):
        params = (
            await self._relevant_parameters(memory)
            if selection == "relevance"
            else self._momentum(self.trajectory, memory, selection, lambda item: item.score)
        )
        pool = {current.prompt: current}
        for _ in range(candidates):
            self.optimizer_calls += 1
            text = await self.generate(
                current.prompt, params, words, examples, provider=self.provider
            )
            pool[text] = await self._measure(text)
        return self._advance(current, pool, words, [], params)

    async def _feedback_step(self, current, candidates, words, momentum, selection, memory):
        failures = list(await self.failures(current.prompt))
        self.optimizer_calls += 1
        gradient = await self.gradient(current.prompt, failures, provider=self.provider)
        previous = self._momentum(self.gradients, memory - 1, selection, lambda item: item.gain)
        feedback = [gradient, *[item.text for item in previous]]
        params = self._momentum(self.trajectory, memory, selection, lambda item: item.score)
        pool = {current.prompt: current}
        for _ in range(candidates):
            self.optimizer_calls += 1
            if momentum == "parameter":
                text = await self.edit_history(
                    current.prompt, gradient, params, words, failures, provider=self.provider
                )
            else:
                text = await self.edit(
                    current.prompt, feedback, words, failures, provider=self.provider
                )
            pool[text] = await self._measure(text)
        chosen = self._advance(current, pool, words, feedback, params)
        self.gradients.append(Gradient(gradient, chosen.score - current.score))
        return chosen

    async def run(
        self,
        initial_prompt: str,
        *,
        iterations: int = 5,
        candidates: int = 8,
        recipe: str = "generate",
        momentum: str = "parameter",
        selection: str = "relevance",
        examples: Sequence[str] = (),
        memory: int = 3,
        initial_words: int = 50,
        final_words: int = 10,
        schedule: str = "cosine",
        warmup: int = 0,
    ) -> dict:
        self.archive, self.history, self.gradients = {}, [], []
        self.evaluations = self.optimizer_calls = self.embedding_calls = 0
        current = await self._measure(initial_prompt)
        self.visited, self.trajectory = {initial_prompt}, [current]
        for step in range(1, iterations + 1):
            words = self._words(step, iterations, initial_words, final_words, schedule, warmup)
            if recipe == "generate":
                current = await self._generation_step(
                    current, candidates, words, selection, memory, examples
                )
            else:
                current = await self._feedback_step(
                    current, candidates, words, momentum, selection, memory
                )
        return {
            "best": max(self.archive.values(), key=lambda item: item.score),
            "current": current,
            "history": self.history,
            "gradients": self.gradients,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "embedding_calls": self.embedding_calls,
        }
