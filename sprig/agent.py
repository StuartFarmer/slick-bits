"""Search system-prompt components using edit enumeration and UCB credit."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Paraphrases(BaseModel, extra="forbid"):
    texts: list[Text]


@dataclass(frozen=True)
class Individual:
    components: tuple[str, ...]
    score: float

    @property
    def prompt(self) -> str:
        return " ".join(self.components)


class SPRIG:
    """Batch evaluator uses one shared training subset per numbered generation.

    Larger scores win. Paraphrasing uses the injected Slick provider. Generation,
    batch alignment, and nonfinite measurement failures propagate without retries.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Sequence[str], int], Awaitable[Sequence[float]]],
        corpus: Sequence[str],
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.corpus = tuple(corpus)

    @prompt(template="rephrase.j2", output_type=Paraphrases)
    async def rephrase(self, component: str, count: int, *, generated: Paraphrases) -> list[str]:
        if len(generated.texts) != count:
            raise ValueError("paraphraser returned the wrong number of components")
        return generated.texts

    def _choose_components(self, count, exploration):
        total = sum(len(values) for values in self.credit.values())

        def ucb(component):
            values = self.credit[component]
            return (
                (
                    math.fsum(values) / len(values)
                    + exploration * math.sqrt(math.log(max(total, 1)) / len(values))
                )
                if values
                else math.inf
            )

        return sorted(self.corpus, key=ucb, reverse=True)[:count]

    async def _expand(self, beam, add_count, paraphrases, exploration):
        edits = []
        additions = self._choose_components(add_count, exploration)
        for parent in beam:
            parts = parent.components
            for position in range(len(parts) + 1):
                for component in additions:
                    edits.append(
                        (parts[:position] + (component,) + parts[position:], parts, component, 1)
                    )
            for position, component in enumerate(parts):
                edits.append((parts[:position] + parts[position + 1 :], parts, component, -1))
                if paraphrases:
                    for variant in await self.rephrase(
                        component, paraphrases, provider=self.provider
                    ):
                        self.origins.setdefault(variant, self.origins.get(component, component))
                        edits.append(
                            (parts[:position] + (variant,) + parts[position + 1 :], parts, None, 0)
                        )
            for first, second in combinations(range(len(parts)), 2):
                swapped = list(parts)
                swapped[first], swapped[second] = swapped[second], swapped[first]
                edits.append((tuple(swapped), parts, None, 0))
        return edits

    async def _select(self, beam, edits, generation, beam_size):
        candidates = list(dict.fromkeys(edit[0] for edit in edits))
        if not candidates:
            return beam
        all_parts = list(dict.fromkeys([p.components for p in beam] + candidates))
        measured = list(await self.evaluate([" ".join(p) for p in all_parts], generation))
        if len(measured) != len(all_parts) or any(not math.isfinite(s) for s in measured):
            raise ValueError("batch evaluator must return one finite score per prompt")
        self.evaluations += len(measured)
        scores = dict(zip(all_parts, measured, strict=True))
        for child, parent, component, direction in edits:
            if direction:
                origin = self.origins.get(component, component)
                gain = direction * (scores[child] - scores[parent])
                if not math.isfinite(gain):
                    raise ValueError("component credit must be finite")
                self.credit.setdefault(origin, []).append(gain)
        return sorted(
            (Individual(parts, scores[parts]) for parts in candidates),
            key=lambda p: p.score,
            reverse=True,
        )[:beam_size]

    async def run(
        self,
        initial: Sequence[str] = (),
        *,
        rounds: int = 10,
        beam_size: int = 10,
        add_count: int = 60,
        paraphrases: int = 10,
        exploration: float = math.sqrt(2),
    ) -> dict:
        self.credit = {component: [] for component in self.corpus}
        self.origins = {component: component for component in self.corpus}
        self.evaluations = 0
        beam = [Individual(tuple(initial), 0.0)]
        beam = await self._select(beam, [(tuple(initial), tuple(initial), None, 0)], -1, beam_size)
        for generation in range(rounds):
            edits = await self._expand(beam, add_count, paraphrases, exploration)
            beam = await self._select(beam, edits, generation, beam_size)
        return {
            "beam": beam,
            "best": beam[0],
            "credit": self.credit,
            "evaluations": self.evaluations,
        }
