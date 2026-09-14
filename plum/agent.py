"""Optimize prompts with Plum's archive GA, hill climb, annealing, tabu, or harmony search."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from slick import prompt
from slick.providers import Provider

from grips.agent import Edit, Spans, edit_text, update_bank


class Plum:
    """Own a metaheuristic search over the same constituent edits as GrIPS.

    The GA variant uses tournament selection from a growing archive and appends
    the best neighbor even when it loses to the global incumbent. No crossover
    is added to this source's GA-M variant. Evaluation is uncached, higher is better.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        spans: Spans,
    ):
        self.task, self.provider = task, provider
        self.evaluate, self.spans = evaluate, spans

    @prompt(template="paraphrase.j2")
    async def paraphrase(self, phrase: str, *, generated: str) -> str:
        if not generated.strip():
            raise ValueError(f"empty paraphrase; response={generated!r}")
        return generated.strip()

    async def _paraphrase(self, phrase):
        self.optimizer_calls += 1
        return await self.paraphrase(phrase, provider=self.provider)

    async def _score(self, text):
        self.evaluations += 1
        score = float(await self.evaluate(text))
        if not math.isfinite(score):
            raise ValueError("fitness must be finite")
        return {"prompt": text, "score": score}

    async def _edit(self, text, operations=None):
        return await edit_text(
            text,
            self.bank,
            self.operations if operations is None else operations,
            self.compose,
            self.spans,
            self._paraphrase,
            self.rng,
        )

    async def _harmony(self, segments, memory_rate, pitch_rate):
        chunks, deleted, added = [], [], []
        for slot in range(segments):
            source = self.rng.choice(self.population)["prompt"]
            spans = list(self.spans(source))
            begin = math.ceil(slot * len(spans) / segments)
            end = math.ceil((slot + 1) * len(spans) / segments)
            # Preserve every constituent; upstream's end-1 slice dropped one per segment.
            chunk = " ".join(source[a:b] for a, b in spans[begin:end])
            if not chunk:
                continue
            in_memory = self.rng.random() < memory_rate
            if not in_memory or self.rng.random() < pitch_rate:
                edit = await self._edit(chunk, ("substitute",) if in_memory else None)
                chunk = edit.text
                deleted.extend(edit.deleted)
                added.extend(edit.added)
            chunks.append(chunk)
        return Edit(" ".join(chunks), deleted, added)

    async def _neighbors(
        self, base, count, algorithm, segments, memory_rate, pitch_rate, tabu_accept
    ):
        neighbors = []
        for _ in range(count):
            edit = (
                await self._harmony(segments, memory_rate, pitch_rate)
                if algorithm == "hs"
                else await self._edit(base["prompt"])
            )
            self.attempts += 1
            if not edit.text:
                self.rejections.append({"attempt": self.attempts, "reason": "empty edit"})
                continue
            if algorithm == "tabu" and edit.text in self.tabu and self.rng.random() > tabu_accept:
                self.rejections.append({"attempt": self.attempts, "reason": "tabu"})
                continue
            neighbors.append((await self._score(edit.text), edit))
        return neighbors

    async def run(
        self,
        initial_prompt: str,
        *,
        algorithm: Literal["ga", "hc", "sa", "tabu", "hs"] = "ga",
        iterations: int = 50,
        candidates: int = 10,
        tournament: int = 5,
        compose: int = 1,
        patience: int = 7,
        operations: Sequence[str] = ("delete", "swap", "substitute", "add"),
        temperature: float = 10.0,
        cooling: float = 5.0,
        tabu_size: int = 5,
        tabu_accept: float = 0.5,
        harmony_size: int = 10,
        segments: int = 5,
        memory_rate: float = 0.4,
        pitch_rate: float = 0.5,
        seed: int = 0,
    ) -> dict:
        self.rng, self.bank, self.tabu = random.Random(seed), [], []
        self.operations, self.compose = operations, compose
        self.evaluations = self.optimizer_calls = self.attempts = 0
        self.rejections = []
        current = best = await self._score(initial_prompt)
        self.population = [current]
        history, stale = [], 0
        for step in range(iterations):
            base = (
                max(self.rng.choices(self.population, k=tournament), key=lambda x: x["score"])
                if algorithm == "ga"
                else current
            )
            neighbors = await self._neighbors(
                base, candidates, algorithm, segments, memory_rate, pitch_rate, tabu_accept
            )
            accepted = False
            if neighbors:
                trial, edit = max(neighbors, key=lambda pair: pair[0]["score"])
                delta = trial["score"] - current["score"]
                heat = temperature * math.exp(-(step + 1) / cooling)
                accepted = delta > 0 or (
                    algorithm == "sa" and heat > 0 and self.rng.random() < math.exp(delta / heat)
                )
                if algorithm == "ga":
                    self.population.append(trial)
                elif algorithm == "tabu":
                    self.tabu = (self.tabu + [trial["prompt"]])[-tabu_size:]
                elif algorithm == "hs":
                    self.population = sorted(
                        self.population + [p for p, _ in neighbors],
                        key=lambda p: p["score"],
                        reverse=True,
                    )[:harmony_size]
                    retained = {p["prompt"] for p in self.population}
                    for item, change in neighbors:
                        if item["prompt"] in retained:
                            update_bank(self.bank, change)
                if accepted:
                    current = trial
                    if algorithm != "hs":
                        update_bank(self.bank, edit)
                stale = 0 if delta > 0 else stale + 1
                if trial["score"] > best["score"]:
                    best = trial
            else:
                stale += 1
            history.append(
                {
                    "step": step,
                    "base": base.copy(),
                    "current": current.copy(),
                    "accepted": accepted,
                    "population_size": len(self.population),
                }
            )
            if stale >= patience:
                break
        return {
            "algorithm": algorithm,
            "best": best,
            "current": current,
            "population": self.population.copy(),
            "history": history,
            "tabu": self.tabu.copy(),
            "deleted": self.bank.copy(),
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "attempts": self.attempts,
            "rejections": self.rejections.copy(),
        }
