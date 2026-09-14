"""Search instructions with constituent edits and greedy or annealed acceptance."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from slick import prompt
from slick.providers import Provider

Spans = Callable[[str], Sequence[tuple[int, int]]]


@dataclass
class Edit:
    text: str
    deleted: list[str]
    added: list[str]


async def edit_text(text, bank, operations, compose, spans, paraphrase, rng):
    """Apply sampled constituent edits; offsets refer to the current exact text.

    This operation is also used by Plum, whose official code reuses GrIPS edits.
    The accepted-deletion bank is frozen while proposing a neighborhood.
    """
    deleted, added = [], []
    available = [op for op in operations if op != "add" or bank]
    for _ in range(compose):
        locations = list(spans(text))
        if not locations or not available:
            break
        operation = rng.choice(available)
        start, end = rng.choice(locations)
        phrase = text[start:end]
        if operation == "delete":
            text = text[:start] + text[end:]
            deleted.append(phrase)
        elif operation == "substitute":
            text = text[:start] + await paraphrase(phrase) + text[end:]
        elif operation == "swap":
            if len(locations) > 1:
                (a, b), (c, d) = sorted(rng.sample(locations, 2))
                text = text[:a] + text[c:d] + text[b:c] + text[a:b] + text[d:]
        elif operation == "add":
            position = rng.choice([0] + [end for _, end in locations])
            restored = rng.choice(bank)
            text = text[:position] + " " + restored + " " + text[position:]
            added.append(restored)
        else:
            raise KeyError(operation)
        text = text.strip()
    return Edit(text, deleted, added)


def update_bank(bank, edit):
    for phrase in edit.added:
        if phrase in bank:
            bank.remove(phrase)
    bank.extend(edit.deleted)


class GrIPS:
    """Inject disjoint constituent offsets, a paraphrase provider, and fitness.

    Higher finite scores are better. Scores are not cached. Provider, parser,
    and evaluator errors propagate; empty edited candidates consume an attempt.
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

    async def _neighbors(self, current, candidates, operations, compose):
        neighbors = []
        for _ in range(candidates):
            edit = await edit_text(
                current["prompt"],
                self.bank,
                operations,
                compose,
                self.spans,
                self._paraphrase,
                self.rng,
            )
            self.attempts += 1
            if edit.text:
                neighbors.append((await self._score(edit.text), edit))
            else:
                self.rejections.append({"attempt": self.attempts, "reason": "empty edit"})
        return neighbors

    async def run(
        self,
        initial_prompt: str,
        *,
        iterations: int = 10,
        candidates: int = 5,
        compose: int = 1,
        patience: int = 2,
        operations: Sequence[str] = ("delete", "swap", "substitute", "add"),
        anneal: bool = False,
        temperature: float = 10.0,
        cooling: float = 5.0,
        seed: int = 0,
    ) -> dict:
        self.rng, self.bank = random.Random(seed), []
        self.evaluations = self.optimizer_calls = self.attempts = 0
        self.rejections = []
        current = best = await self._score(initial_prompt)
        history, stale = [], 0
        for step in range(iterations):
            neighbors = await self._neighbors(current, candidates, operations, compose)
            accepted, improved = False, False
            if neighbors:
                trial, edit = max(neighbors, key=lambda pair: pair[0]["score"])
                delta = trial["score"] - current["score"]
                improved = delta > 0
                heat = temperature * math.exp(-step / cooling)
                accepted = improved or (
                    anneal and heat > 0 and self.rng.random() < math.exp(delta / heat)
                )
                if accepted:
                    current = trial
                    update_bank(self.bank, edit)
                    if current["score"] > best["score"]:
                        best = current
            stale = 0 if improved else stale + 1
            history.append({"step": step, "accepted": accepted, "current": current.copy()})
            if stale >= patience and not accepted:
                break
        return {
            "best": best,
            "current": current,
            "history": history,
            "deleted": self.bank.copy(),
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "attempts": self.attempts,
            "rejections": self.rejections.copy(),
        }
