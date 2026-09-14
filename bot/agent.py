"""Boost heterogeneous thought trees by accumulating chain analysis and revision advice."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, Field
from slick import prompt
from slick.providers import Provider


class Score(BaseModel, extra="forbid"):
    value: float = Field(ge=0, le=1, allow_inf_nan=False)


class Scores(BaseModel, extra="forbid", frozen=True):
    node: float = Field(ge=0, le=1, allow_inf_nan=False)
    edge: float = Field(ge=0, le=1, allow_inf_nan=False)


@dataclass(frozen=True)
class Thought:
    text: str
    node: float
    edge: float

    @property
    def weight(self) -> float:
        return self.node + self.edge


Chain = tuple[Thought, ...]
Evaluator = Callable[[tuple[str, ...]], Awaitable[Scores]]


@dataclass(frozen=True)
class Tree:
    strategy: str
    leaves: tuple[Chain, ...]
    expansions: int
    temperature: float | None = None
    top_p: float | None = None

    @property
    def best(self) -> Chain:
        return max(self.leaves, key=lambda chain: sum(step.weight for step in chain))


@dataclass(frozen=True)
class Experience:
    chain: Chain
    feedback: str


@dataclass(frozen=True)
class Result:
    answer: str
    chain: Chain
    experiences: tuple[Experience, ...]
    calls: int


class BoostingOfThoughts:
    """Own one BoT run; callers own provider configuration and external evaluation.

    Without an evaluator the model scores both nodes and edges. An evaluator
    receives the entire candidate path as text and supplies normalized scores.
    An optional tree_provider(temperature, top_p) constructs independently
    configured generators. The main provider handles scoring, feedback and joins.
    One run at a time per instance; no implicit retries or shared Sessions.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Evaluator | None = None,
        *,
        tree_provider: Callable[[float, float], Provider] | None = None,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.tree_provider = tree_provider
        self.calls: list[dict] = []
        self.assessments: list[dict] = []
        self.experiences: list[Experience] = []
        self.forests: list[tuple[Tree, ...]] = []

    @prompt(template="generate.j2")
    async def generate(self, chain: Chain, *, generated: str) -> str:
        """Generate one next thought, using accumulated experience."""
        if not generated.strip():
            raise ValueError("generated thought is blank")
        return generated.strip()

    @prompt(template="score_thought.j2", output_type=Score)
    async def score_thought(self, chain: Chain, thought: str, *, generated: Score) -> float:
        """Assess correctness and usefulness in the context of the full path."""
        return generated.value

    @prompt(template="score_edge.j2", output_type=Score)
    async def score_edge(self, chain: Chain, thought: str, *, generated: Score) -> float:
        """Assess confidence in the transition from the parent state."""
        return generated.value

    @prompt(template="similarity.j2", output_type=Score)
    async def similarity(self, left: str, right: str, *, generated: Score) -> float:
        """Assess whether two thoughts represent sufficiently similar states."""
        return generated.value

    @prompt(template="analyze.j2")
    async def analyze(self, chain: Chain, *, generated: str) -> str:
        """Produce chain and per-step error reports, advice and confidence."""
        if not generated.strip():
            raise ValueError("generated feedback is blank")
        return generated

    @prompt(template="finish.j2")
    async def finish(self, chain: Chain, *, generated: str) -> str:
        """Produce the task's final artifact from the final chain and all experience."""
        if not generated.strip():
            raise ValueError("generated answer is blank")
        return generated

    async def run(
        self,
        *,
        iterations: int = 10,
        trees: int = 15,
        max_depth: int = 5,
        max_expansions: int | None = None,
        aggregation: Literal["greedy", "best_first"] = "greedy",
        growth_range: tuple[float, float] = (0.3, 0.8),
        similarity_threshold: float = 0.7,
        seed: int = 0,
    ) -> Result:
        """Build, aggregate, analyze and accumulate for exactly `iterations` rounds.

        Depth counts generated thoughts, excluding the empty root. One expansion
        generates and scores two children. By default all eligible nodes expand;
        max_expansions caps expansions per tree when a smaller search is wanted.
        Model confidence never terminates the outer loop as a claim of success.
        """
        self.calls, self.assessments, self.experiences, self.forests = [], [], [], []
        rng = random.Random(seed)
        limit = 2**max_depth - 1 if max_expansions is None else max_expansions
        chain: Chain = ()
        for _ in range(iterations):
            forest = await self._forest(trees, max_depth, limit, growth_range, rng)
            self.forests.append(forest)
            chain = await self.aggregate(
                [tree.best for tree in forest], aggregation, max_depth, similarity_threshold
            )
            feedback = await self._call(self.analyze, chain)
            self.experiences.append(Experience(chain, feedback))
        answer = await self._call(self.finish, chain)
        return Result(answer, chain, tuple(self.experiences), len(self.calls))

    async def _call(self, operation, *args, source=None):
        """Capture raw output before Slick parsing or postprocessing can reject it."""
        record = {"operation": operation.__name__}
        self.calls.append(record)
        source = self.provider if source is None else source

        async def acall(context):
            record["prompt"] = context
            text, requests = await source.acall(context)
            record["response"] = text
            if requests:
                raise ValueError("BoT requires text responses, not tool requests")
            return text, requests

        try:
            return await operation(*args, provider=SimpleNamespace(acall=acall))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _assess(self, chain: Chain, thought: str) -> Thought:
        if self.evaluate is None:
            node = await self._call(self.score_thought, chain, thought)
            edge = await self._call(self.score_edge, chain, thought)
            return Thought(thought, node, edge)
        texts = tuple(step.text for step in chain) + (thought,)
        record = {"chain": texts}
        self.assessments.append(record)
        try:
            scores = await self.evaluate(texts)
            scores = Scores(node=scores.node, edge=scores.edge)
            record["scores"] = scores
            return Thought(thought, scores.node, scores.edge)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def build_tree(
        self,
        source: Provider,
        strategy: Literal["level", "leaf"],
        depth: int,
        limit: int,
        growth_range: tuple[float, float],
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> Tree:
        """Grow a weighted binary tree breadth-first or best-leaf-first."""
        frontier: list[Chain] = [()]
        leaves: list[Chain] = []
        expansions = 0
        low, high = growth_range
        while frontier and expansions < limit:
            index = 0
            if strategy == "leaf":
                index = max(
                    range(len(frontier)),
                    key=lambda i: frontier[i][-1].weight if frontier[i] else 0,
                )
            chain = frontier.pop(index)
            if len(chain) >= depth:
                leaves.append(chain)
                continue
            for _ in range(2):
                text = await self._call(self.generate, chain, source=source)
                thought = await self._assess(chain, text)
                child = chain + (thought,)
                if (
                    len(child) < depth
                    and low <= thought.node <= high
                    and low <= thought.edge <= high
                ):
                    frontier.append(child)
                else:
                    leaves.append(child)
            expansions += 1
        # Unexpanded frontier nodes are leaves when the expansion budget ends.
        return Tree(strategy, tuple(leaves + frontier), expansions, temperature, top_p)

    async def _forest(self, count, depth, limit, growth_range, rng) -> tuple[Tree, ...]:
        settings = []
        for index in range(count):
            source, temperature, top_p = self.provider, None, None
            if self.tree_provider is not None:
                temperature = rng.choice((0.2, 0.4, 0.6, 0.7, 0.9, 1.1, 1.5))
                top_p = rng.choice((0.1, 0.3, 0.5, 0.7, 0.9))
                source = self.tree_provider(temperature, top_p)
            strategy = "level" if index % 2 == 0 else "leaf"
            settings.append((source, strategy, depth, limit, growth_range, temperature, top_p))
        tasks = [asyncio.create_task(self.build_tree(*setting)) for setting in settings]
        try:
            return tuple(await asyncio.gather(*tasks))
        except BaseException:
            # Finish cancellation before returning so no tree keeps spending after failure.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def aggregate(
        self,
        chains: list[Chain],
        strategy: Literal["greedy", "best_first"],
        depth: int,
        threshold: float,
    ) -> Chain:
        """Select the best chain or greedily join successors of similar thoughts."""
        if strategy == "best_first":
            return max(chains, key=lambda chain: sum(step.weight for step in chain))
        starts = [chain[0] for chain in chains if chain]
        if not starts or depth == 0:
            return ()
        result = [max(starts, key=lambda thought: thought.weight)]
        seen = {result[0].text}
        similarities = {}
        while len(result) < depth:
            candidates = []
            for chain in chains:
                for parent, child in zip(chain, chain[1:]):
                    if child.text in seen:
                        continue
                    pair = (result[-1].text, parent.text)
                    if pair not in similarities:
                        similarities[pair] = (
                            1.0 if pair[0] == pair[1] else await self._call(self.similarity, *pair)
                        )
                    if similarities[pair] > threshold:
                        candidates.append(child)
            if not candidates:
                break
            result.append(max(candidates, key=lambda thought: thought.weight))
            seen.add(result[-1].text)
        return tuple(result)
