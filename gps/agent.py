"""Select and paraphrase prompt populations using Genetic Prompt Search."""

import math
import random
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from slick import prompt
from slick.providers import Provider

LANGUAGES = (
    "Chinese",
    "Japanese",
    "Korean",
    "French",
    "Spanish",
    "Italian",
    "Russian",
    "German",
    "Arabic",
    "Greek",
    "Cantonese",
)
TAGS = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.DOTALL)
SENTINELS = re.compile(r"<extra_id_(\d+)>")


def preserves_placeholders(parent: str, child: str) -> bool:
    """Preserve literal slots and the ordered control/comment tags, without rendering."""
    before, after = TAGS.findall(parent), TAGS.findall(child)
    return Counter(before) == Counter(after) and [
        tag for tag in before if not tag.startswith("{{")
    ] == [tag for tag in after if not tag.startswith("{{")]


def mask_prompt(parent: str, rng: random.Random, fraction: float) -> tuple[str, list[str]]:
    """Mask sampled whitespace words outside literal Jinja tags."""
    protected = [match.span() for match in TAGS.finditer(parent)]
    # ponytail: whitespace words approximate T5 tokens; use tokenizer offsets for exact replication.
    words = [
        match.span()
        for match in re.finditer(r"\S+", parent)
        if not any(start < match.end() and end > match.start() for start, end in protected)
    ]
    if not words:
        return parent, []
    spans = sorted(rng.sample(words, max(1, math.ceil(len(words) * fraction))))
    pieces, end = [], 0
    for start, stop in spans:
        pieces.append(parent[end:start])
        end = stop
    pieces.append(parent[end:])
    masked = "".join(piece + f"<extra_id_{i}>" for i, piece in enumerate(pieces[:-1]))
    return masked + pieces[-1], pieces


def fill_prompt(raw: str, pieces: list[str]) -> str:
    """Rebuild immutable unmasked text from a T5-style sentinel completion."""
    parts = SENTINELS.split(raw.strip())
    count = len(pieces) - 1
    if parts[0].strip() or parts[-1].strip() or parts[1::2] != [str(i) for i in range(count + 1)]:
        raise ValueError("invalid cloze")
    fills = parts[2:-1:2]
    if any(not fill.strip() for fill in fills):
        raise ValueError("invalid cloze")
    return "".join(piece + fill.strip() for piece, fill in zip(pieces, fills)) + pieces[-1]


class GPS:
    """Select top-k parents per generation and return top-k across all generations.

    Children replace the generation; an entirely rejected generation falls back
    to its parents. The final archive preserves each generation's top-k winners.
    The provider replaces the original generation models; task-specific validity
    belongs to the optional accept(parent, child) callback.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        accept: Callable[[str, str], bool] = preserves_placeholders,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.accept = accept

    @prompt(template="mutate.j2")
    async def mutate(self, parent: str, *, generated: str) -> str:
        return generated

    @prompt(template="translate.j2")
    async def translate(self, parent: str, language: str, *, generated: str) -> str:
        return generated

    @prompt(template="back_translate.j2")
    async def back_translate(self, translation: str, language: str, *, generated: str) -> str:
        return generated

    @prompt(template="cloze.j2")
    async def cloze(self, masked: str, sentinel_count: int, *, generated: str) -> str:
        return generated

    async def _score(self, text):
        self.evaluations += 1
        score = float(await self.evaluate(text))
        if not math.isfinite(score):
            raise ValueError("fitness must be finite")
        return {"prompt": text, "score": score}

    async def _generate(self, parent, strategy, index, languages, mask_fraction, record):
        if strategy == "sentence_continuation":
            self.optimizer_calls += 1
            record["raw"] = await self.mutate(parent, provider=self.provider)
            return record["raw"].strip()
        if strategy == "back_translation":
            language = languages[index % len(languages)]
            record["language"] = language
            self.optimizer_calls += 1
            translation = await self.translate(parent, language, provider=self.provider)
            record["translation"] = translation
            if not translation.strip() or not preserves_placeholders(parent, translation):
                record["rejection"] = "translation constraint"
                return ""
            self.optimizer_calls += 1
            record["raw"] = await self.back_translate(translation, language, provider=self.provider)
            return record["raw"].strip()
        if strategy == "cloze":
            masked, pieces = mask_prompt(parent, self.rng, mask_fraction)
            record["masked"] = masked
            if not pieces:
                record["rejection"] = "no editable tokens"
                return ""
            self.optimizer_calls += 1
            record["raw"] = await self.cloze(masked, len(pieces) - 1, provider=self.provider)
            try:
                return fill_prompt(record["raw"], pieces)
            except ValueError:
                record["rejection"] = "invalid cloze"
                return ""
        raise KeyError(strategy)

    async def _offspring(self, parents, counts, generation, strategy, languages, mask_fraction):
        children = []
        for parent, count in zip(parents, counts):
            for index in range(count):
                self.attempts += 1
                record = {
                    "generation": generation,
                    "parent": parent["prompt"],
                    "strategy": strategy,
                    "raw": None,
                    "rejection": "pending",
                }
                self.history.append(record)
                child = await self._generate(
                    parent["prompt"], strategy, index, languages, mask_fraction, record
                )
                if record["rejection"] != "pending":
                    continue
                reason = (
                    "empty"
                    if not child
                    else "duplicate"
                    if child in self.seen
                    else "template constraint"
                    if not self.accept(parent["prompt"], child)
                    else ""
                )
                record.update(child=child, rejection=reason)
                if reason:
                    continue
                self.seen.add(child)
                children.append(await self._score(child))
        return children or parents.copy()

    async def run(
        self,
        initial_prompts: Sequence[str],
        *,
        generations: int = 7,
        top_k: int | None = None,
        offspring_per_parent: int | None = None,
        pool_size: int = 30,
        strategy: Literal[
            "sentence_continuation", "back_translation", "cloze"
        ] = "sentence_continuation",
        languages: Sequence[str] = LANGUAGES,
        mask_fraction: float = 0.15,
        seed: int = 0,
        rescore_final: bool = False,
    ) -> dict:
        """Score G0 and reproduce generations-1 times; optionally rescore the archive.

        Scores are finite and higher is better. Default SC/cloze pools allocate
        pool_size attempts across parents; BT uses each language once per parent.
        offspring_per_parent overrides either budget. Rejections consume attempts
        without retries. Provider/evaluator errors propagate, retaining run state.
        """
        self.evaluations = self.optimizer_calls = self.attempts = 0
        self.history = []
        self.rng = random.Random(seed)
        self.seen = set(initial_prompts)
        population = [await self._score(text) for text in dict.fromkeys(initial_prompts)]
        top_k = len(population) if top_k is None else top_k
        parents = sorted(population, key=lambda x: x["score"], reverse=True)[:top_k]
        archive = {item["prompt"]: item for item in parents}
        snapshots = [{"population": population.copy(), "selected": parents.copy()}]
        for generation in range(1, generations):
            if offspring_per_parent is not None:
                counts = [offspring_per_parent] * len(parents)
            elif strategy == "back_translation":
                counts = [len(languages)] * len(parents)
            else:
                quotient, remainder = divmod(pool_size, len(parents))
                counts = [quotient + (i < remainder) for i in range(len(parents))]
            population = await self._offspring(
                parents, counts, generation, strategy, languages, mask_fraction
            )
            parents = sorted(population, key=lambda x: x["score"], reverse=True)[:top_k]
            archive.update((item["prompt"], item) for item in parents)
            snapshots.append({"population": population.copy(), "selected": parents.copy()})
        final_pool = list(archive.values())
        if rescore_final:
            final_pool = [await self._score(item["prompt"]) for item in final_pool]
        finalists = sorted(final_pool, key=lambda x: x["score"], reverse=True)[:top_k]
        return {
            "best": finalists[0],
            "finalists": finalists,
            "population": population,
            "history": self.history.copy(),
            "archive": list(archive.values()),
            "generations": snapshots,
            "evaluations": self.evaluations,
            "optimizer_calls": self.optimizer_calls,
            "attempts": self.attempts,
        }
