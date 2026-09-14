"""Adapt LLM search intensity, island allocation, and solution tactics.

Adapted from SkyDiscover's AdaEvolve implementation; see NOTICE and README.md.
Modified for Slick and arbitrary caller-evaluated text, following the paper's equations.
"""

import math
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Proposal(BaseModel, extra="forbid"):
    content: str = Field(min_length=1)


class Tactic(BaseModel, extra="forbid"):
    idea: Text
    description: Text
    what_to_optimize: Text
    cautions: str
    approach_type: Text


class Tactics(BaseModel, extra="forbid"):
    ideas: list[Tactic] = Field(min_length=1, max_length=3)


class InvalidCandidate(ValueError):
    """Generated content cannot be used by the search."""


@dataclass(frozen=True)
class Evaluation:
    """Caller-measured score and diagnostics; error marks an invalid candidate."""

    score: float = 0.0
    feedback: str = ""
    error: str = ""


@dataclass(frozen=True)
class Candidate:
    id: int
    content: str
    score: float
    feedback: str = ""
    parent_id: int | None = None


@dataclass
class Island:
    archive: list[Candidate]
    maximize: bool = True
    signal: float = 0.0
    reward: float = 0.0
    decayed_visits: float = 0.0
    visits: int = 0

    @property
    def best(self) -> Candidate:
        return (max if self.maximize else min)(self.archive, key=lambda p: p.score)


@dataclass(frozen=True)
class Config:
    initial_islands: int = 2
    decay: float = 0.9
    epsilon: float = 1e-8
    intensity_min: float = 0.1
    intensity_max: float = 0.7
    migration_interval: int = 15
    spawn_threshold: float = 0.02
    meta_threshold: float = 0.12
    spawn_cooldown: int = 15
    max_islands: int = 8
    meta_warmup: int = 15
    tactic_uses: int = 5
    inspirations: int = 3


def text_distance(first: str, second: str) -> float:
    """Port SkyDiscover's text diversity: 70% token Jaccard, 30% length."""
    # ponytail: lexical proxy, inject semantic distance if task-level diversity requires it.
    tokens = [
        {t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+\.?[0-9]*", s) if len(t) >= 2}
        for s in (first, second)
    ]
    union = tokens[0] | tokens[1]
    jaccard = 1 - len(tokens[0] & tokens[1]) / len(union) if union else 0.0
    return 0.7 * jaccard + 0.3 * abs(len(first) - len(second)) / max(len(first), len(second), 1)


def checked_content(proposal: Proposal) -> str:
    if not proposal.content.strip():
        raise InvalidCandidate("Generated content is blank")
    return proposal.content


class AdaEvolve:
    """Sequential AdaEvolve with independent model contexts and external evaluation.

    Configure Slick's template root at application startup. Use a fresh instance
    per run; the provider owns transport retries and the evaluator owns execution
    isolation. Inspect attempts, events, islands, and tactic_history after a run.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[Evaluation]],
        *,
        evaluator_context: str = "",
        meta_provider: Provider | None = None,
        maximize: bool = True,
        config: Config = Config(),
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.evaluator_context = evaluator_context
        self.meta_provider = provider if meta_provider is None else meta_provider
        self.maximize, self.config = maximize, config
        self.direction = 1 if maximize else -1
        self.score_direction = "Higher" if maximize else "Lower"
        self.attempts: list[dict] = []
        self.events: list[dict] = []
        self.tactic_history: list[dict] = []
        self.active_tactics: list[int] = []
        self.mutation_calls = self.meta_calls = self.evaluations = 0
        self.last_spawn = self.iteration = 0

    @prompt(template="explore.j2", output_type=Proposal)
    async def explore(
        self, parent: Candidate, inspirations: list[Candidate], *, generated: Proposal
    ) -> str:
        return checked_content(generated)

    @prompt(template="exploit.j2", output_type=Proposal)
    async def exploit(
        self, parent: Candidate, inspirations: list[Candidate], *, generated: Proposal
    ) -> str:
        return checked_content(generated)

    @prompt(template="explore_guided.j2", output_type=Proposal)
    async def explore_guided(
        self,
        parent: Candidate,
        inspirations: list[Candidate],
        tactic: Tactic,
        *,
        generated: Proposal,
    ) -> str:
        return checked_content(generated)

    @prompt(template="exploit_guided.j2", output_type=Proposal)
    async def exploit_guided(
        self,
        parent: Candidate,
        inspirations: list[Candidate],
        tactic: Tactic,
        *,
        generated: Proposal,
    ) -> str:
        return checked_content(generated)

    @prompt(template="generate_tactics.j2", output_type=Tactics)
    async def generate_tactics(
        self, best: Candidate, recent: list[dict], tried: list[dict], *, generated: Tactics
    ) -> list[Tactic]:
        names = [idea.idea.casefold() for idea in generated.ideas]
        if len(set(names)) != len(names):
            raise InvalidCandidate("Tactic ideas must be distinct")
        return generated.ideas

    async def acall(self, context, *, tools=None, tool_results=None):
        """Record raw responses before Slick parses them; used only by _generate."""
        raw, calls = await self._call_provider.acall(context)
        self._call_record["raw"] = raw
        return raw, calls

    async def _generate(self, operation, args, provider, record):
        self.attempts.append(record)
        self._call_provider, self._call_record = provider, record
        if operation.__name__ == "generate_tactics":
            self.meta_calls += 1
        else:
            self.mutation_calls += 1
        try:
            return await operation(*args, provider=self)
        except (ValidationError, InvalidCandidate) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if "raw" not in record:
                raise  # The provider failed before generated-output validation began.
            return None
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def assess(self, content: str, record: dict) -> Candidate | None:
        record["content"] = content
        self.evaluations += 1
        try:
            result = await self.evaluate(content)
        except TimeoutError as exc:
            record["error"] = f"TimeoutError: {exc}"
            return None
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        record["feedback"] = result.feedback
        if result.error or not math.isfinite(result.score):
            record["error"] = result.error or "Measured score must be finite"
            return None
        record["score"] = result.score
        return Candidate(record["id"], content, result.score, result.feedback, record.get("parent"))

    def intensity(self, island: Island) -> float:
        c = self.config
        return c.intensity_min + (c.intensity_max - c.intensity_min) / (
            1 + math.sqrt(island.signal + c.epsilon)
        )

    def select_island(self) -> int:
        unvisited = [k for k, island in enumerate(self.islands) if island.visits == 0]
        if unvisited:
            return self.rng.choice(unvisited)
        total = sum(island.visits for island in self.islands)
        return max(
            range(len(self.islands)),
            key=lambda k: (
                self.islands[k].reward / self.islands[k].decayed_visits
                + math.sqrt(2 * math.log(total) / self.islands[k].visits)
            ),
        )

    def sample(self, island: Island) -> tuple[Candidate, list[Candidate], bool]:
        explore = self.rng.random() < self.intensity(island)
        ranked = sorted(island.archive, key=lambda p: self.direction * p.score, reverse=True)
        parent = self.rng.choice(
            island.archive if explore else ranked[: math.ceil(len(ranked) / 4)]
        )
        inspirations = [p for p in ranked if p.id != parent.id]
        if explore:
            inspirations.sort(key=lambda p: text_distance(parent.content, p.content), reverse=True)
        return parent, inspirations[: self.config.inspirations], explore

    def update_state(
        self, island: Island, child: Candidate | None, *, migrant: bool = False
    ) -> None:
        """Algorithm 6; migrants update local signal and baseline without UCB credit."""
        c = self.config
        local_before, global_before = island.best.score, self.best.score
        gain = max(self.direction * (child.score - local_before), 0.0) if child else 0.0
        if not migrant or gain > 0:
            delta = gain / (abs(local_before) + c.epsilon)
            island.signal = c.decay * island.signal + (1 - c.decay) * delta**2
        if not migrant:
            island.reward = c.decay * island.reward + gain / (abs(global_before) + c.epsilon)
            island.decayed_visits = c.decay * island.decayed_visits + 1
            island.visits += 1
        if child is not None:
            if all(p.id != child.id for p in island.archive):
                island.archive.append(child)
            if self.direction * (child.score - global_before) > 0:
                self.best = child

    async def mutate(self, k: int) -> None:
        island = self.islands[k]
        parent, inspirations, explore = self.sample(island)
        args = [parent, inspirations]
        operation = self.explore if explore else self.exploit
        tactic_index = self.active_tactics[0] if self.active_tactics else None
        if tactic_index is not None:
            args.append(Tactic.model_validate(self.tactic_history[tactic_index]["tactic"]))
            operation = self.explore_guided if explore else self.exploit_guided
        record = {
            "id": self.iteration,
            "iteration": self.iteration,
            "island": k,
            "operation": operation.__name__,
            "parent": parent.id,
            "intensity": self.intensity(island),
            "tactic": tactic_index,
        }
        before = self.best.score
        content = await self._generate(operation, args, self.provider, record)
        child = await self.assess(content, record) if content is not None else None
        self.update_state(island, child)
        record["best_score"] = self.best.score
        if tactic_index is not None:
            outcome = self.tactic_history[tactic_index]
            outcome["uses"] += 1
            outcome["improvement"] += self.direction * (self.best.score - before)
            outcome["best_score"] = self.best.score
            self.active_tactics.pop(0)
            if outcome["uses"] < self.config.tactic_uses:
                self.active_tactics.append(tactic_index)

    def migrate(self) -> None:
        # Snapshot first: an incoming migrant must not travel multiple hops in one round.
        outgoing = [island.best for island in self.islands]
        for k, candidate in enumerate(outgoing):
            self.update_state(self.islands[(k + 1) % len(self.islands)], candidate, migrant=True)
        self.events.append({"iteration": self.iteration, "operation": "migration"})

    def finish_iteration(self) -> None:
        c = self.config
        if c.migration_interval and self.iteration % c.migration_interval == 0:
            self.migrate()
        if (
            self.iteration - self.last_spawn >= c.spawn_cooldown
            and len(self.islands) < c.max_islands
            and all(i.visits > 0 and i.signal <= c.spawn_threshold for i in self.islands)
        ):
            archive = {p.id: p for i in self.islands for p in i.archive}
            seed = self.rng.choice(list(archive.values()))
            self.islands.append(Island([seed], maximize=self.maximize))
            self.last_spawn = self.iteration
            self.events.append({"iteration": self.iteration, "operation": "spawn", "seed": seed.id})

    async def guide(self) -> None:
        c = self.config
        if (
            self.active_tactics
            or self.iteration < c.meta_warmup
            or not all(i.signal <= c.meta_threshold for i in self.islands)
        ):
            return
        recent = [dict(a) for a in self.attempts if a["operation"] != "generate_tactics"][-10:]
        record = {"iteration": self.iteration, "operation": "generate_tactics"}
        ideas = await self._generate(
            self.generate_tactics,
            [self.best, recent, self.tactic_history],
            self.meta_provider,
            record,
        )
        if ideas is not None:
            for idea in ideas:
                self.active_tactics.append(len(self.tactic_history))
                self.tactic_history.append(
                    {
                        "tactic": idea.model_dump(),
                        "uses": 0,
                        "starting_score": self.best.score,
                        "best_score": self.best.score,
                        "improvement": 0.0,
                    }
                )

    async def run(
        self, initial: str, *, iterations: int = 100, seed: int = 0, max_calls: int | None = None
    ) -> Candidate:
        """Run up to iterations mutations; max_calls includes mutation and meta calls."""
        self.rng = random.Random(seed)
        self.seed_evaluation = {"id": 0, "operation": "seed"}
        self.best = await self.assess(initial, self.seed_evaluation)
        if self.best is None:
            raise RuntimeError(f"Initial candidate is invalid: {self.seed_evaluation['error']}")
        self.islands = [
            Island([self.best], maximize=self.maximize) for _ in range(self.config.initial_islands)
        ]
        budget = math.inf if max_calls is None else max_calls
        for self.iteration in range(1, iterations + 1):
            if self.mutation_calls + self.meta_calls >= budget:
                break
            await self.mutate(self.select_island())
            self.finish_iteration()
            # Reserve a mutation after meta-guidance; never generate unusable final tactics.
            if self.iteration < iterations and self.mutation_calls + self.meta_calls + 1 < budget:
                await self.guide()
        return self.best
