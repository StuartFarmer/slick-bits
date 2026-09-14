"""Match simulated behavior frequencies by editing selected agents' persona fields."""

import random
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Annotated

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Edit(BaseModel, extra="forbid"):
    fields: dict[str, Text]


class PersonaEvolve:
    def __init__(
        self,
        task: str,
        provider: Provider,
        simulate: Callable[[dict[str, dict[str, str]]], Awaitable[dict[str, str]]],
        target: Mapping[str, float],
        *,
        editable: Sequence[str],
        context: Mapping[str, str] | None = None,
        words_per_field: int = 25,
        seed: int = 0,
    ):
        self.task, self.provider, self.simulate = task, provider, simulate
        self.target, self.editable = dict(target), tuple(editable)
        self.context = dict(context or {})
        self.words_per_field, self.seed = words_per_field, seed

    @prompt(template="rewrite.j2", output_type=Edit)
    async def rewrite(
        self, persona: dict, observed: str, target: str, context: str, *, generated: Edit
    ) -> dict[str, str]:
        if set(generated.fields) != set(self.editable):
            raise ValueError("rewrite must contain exactly the editable fields")
        if any(len(text.split()) > self.words_per_field for text in generated.fields.values()):
            raise ValueError("persona field exceeds word limit")
        return {**persona, **generated.fields}

    def assignments(self, behaviors, rng):
        counts, grouped = Counter(behaviors.values()), defaultdict(list)
        n = len(behaviors)
        for name, category in behaviors.items():
            grouped[category].append(name)
        gaps = {category: target - counts[category] / n for category, target in self.target.items()}
        destinations = [category for category, gap in gaps.items() if gap > 0]
        weights = [max(0.001, gaps[category]) for category in destinations]
        assignments = []
        for category, gap in gaps.items():
            if gap < 0 and destinations:
                count = min(int(abs(gap) * n), len(grouped[category]))
                for name in rng.sample(grouped[category], count):
                    assignments.append(
                        (name, category, rng.choices(destinations, weights=weights)[0])
                    )
        return gaps, assignments

    async def revise_population(self, personas, assignments):
        updated = {name: dict(persona) for name, persona in personas.items()}
        for name, observed, target in assignments:
            updated[name] = await self.rewrite(
                personas[name], observed, target, self.context.get(name, ""), provider=self.provider
            )
        return updated

    async def run(self, initial: Mapping[str, Mapping[str, str]], *, rounds: int = 10) -> dict:
        personas = {name: dict(persona) for name, persona in initial.items()}
        rng, history = random.Random(self.seed), []
        for iteration in range(rounds + 1):
            behaviors = await self.simulate({name: dict(p) for name, p in personas.items()})
            if set(behaviors) != set(personas):
                raise ValueError("simulation must classify every supplied persona")
            if any(b not in self.target and b != "UNKNOWN" for b in behaviors.values()):
                raise ValueError("simulation returned an unknown behavior label")
            gaps, assignments = self.assignments(behaviors, rng)
            history.append({"behaviors": dict(behaviors), "gaps": gaps, "edits": assignments})
            if iteration == rounds or not assignments:
                break
            personas = await self.revise_population(personas, assignments)
        return {"personas": personas, "history": history, "simulation_calls": len(history)}
