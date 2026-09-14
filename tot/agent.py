"""Search textual thought trees with beam BFS or pruned, backtracking DFS."""

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Thoughts(BaseModel, extra="forbid"):
    thoughts: list[Text]


class Value(BaseModel, extra="forbid"):
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class Vote(BaseModel, extra="forbid"):
    choice: int = Field(strict=True, ge=1)


@dataclass(frozen=True)
class State:
    thoughts: tuple[str, ...] = ()
    value: float = 0.0


@dataclass(frozen=True)
class Solution:
    state: State
    answer: str


@dataclass(frozen=True)
class Result:
    solutions: tuple[Solution, ...]
    best_state: State
    expansions: int
    budget_exhausted: bool


class TreeOfThoughts:
    """Own one search at a time; task context defines what a thought means.

    An optional async evaluator replaces LM evaluation and scores a State, with
    higher finite values preferred. Provider configuration and retries belong to
    the caller. Calls are independent and sequential, without shared sessions.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[State], Awaitable[float]] | None = None,
        *,
        thought: str = "One coherent intermediate step toward solving the task",
        criteria: str = "Feasibility, progress, and consistency with all task constraints",
        answer_format: str = "A complete answer satisfying the task",
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.thought, self.criteria, self.answer_format = thought, criteria, answer_format
        self.calls: list[dict] = []
        self.assessments: list[dict] = []
        self.history: list[tuple[State, ...]] = []

    @prompt(template="sample.j2")
    async def sample(self, state: State, *, generated: str) -> str:
        """Sample one continuation independently of sibling candidates."""
        if not generated.strip():
            raise ValueError("generated thought is blank")
        return generated

    @prompt(template="propose.j2", output_type=Thoughts)
    async def propose(self, state: State, count: int, *, generated: Thoughts) -> list[str]:
        """Propose alternatives together, allowing an empty list for a dead end."""
        if len(generated.thoughts) > count:
            raise ValueError("proposal exceeds candidate limit")
        return generated.thoughts

    @prompt(template="value.j2", output_type=Value)
    async def value(self, state: State, *, generated: Value) -> float:
        """Estimate the prospect of completing this partial solution."""
        return generated.score

    @prompt(template="vote.j2", output_type=Vote)
    async def vote(self, states: list[State], *, generated: Vote) -> int:
        """Choose a candidate by its one-based position."""
        if generated.choice > len(states):
            raise ValueError("vote names an unknown candidate")
        return generated.choice - 1

    @prompt(template="finish.j2")
    async def finish(self, state: State, *, generated: str) -> str:
        """Generate the final artifact from a terminal search state."""
        if not generated.strip():
            raise ValueError("generated answer is blank")
        return generated

    async def run(
        self,
        *,
        search: Literal["bfs", "dfs"] = "bfs",
        depth: int = 3,
        breadth: int = 5,
        candidates: int = 5,
        generation: Literal["sample", "propose"] = "propose",
        evaluation: Literal["value", "vote"] = "value",
        evaluation_samples: int = 3,
        threshold: float = 0.0,
        max_expansions: int = 100,
    ) -> Result:
        """Run Algorithm 1 or 2 with fixed thought depth and an expansion budget.

        BFS returns one final answer; DFS records every surviving depth-limit leaf.
        Empty/pruned trees return no solutions. Budget exhaustion returns partial
        results explicitly. Failures abort without retries; records remain on the agent.
        """
        self.depth, self.breadth, self.candidates = depth, breadth, candidates
        self.generation, self.evaluation = generation, evaluation
        self.evaluation_samples, self.threshold = evaluation_samples, threshold
        self.max_expansions = max_expansions
        self.calls, self.assessments, self.history = [], [], []
        self.expansions, self.budget_exhausted, self.best_state = 0, False, State()
        solutions = await {"bfs": self._bfs, "dfs": self._dfs}[search]()
        return Result(tuple(solutions), self.best_state, self.expansions, self.budget_exhausted)

    async def _bfs(self) -> list[Solution]:
        frontier = [State()]
        for _ in range(self.depth):
            # Finish whole levels so an incomplete expansion cannot bias beam selection.
            if self.expansions + len(frontier) > self.max_expansions:
                self.budget_exhausted = True
                return []
            children = []
            for state in frontier:
                children.extend(await self._expand(state))
            frontier = (await self._rank(children))[: self.breadth]
            self.history.append(tuple(frontier))
            if not frontier:
                return []
            self.best_state = frontier[0]
        return [Solution(frontier[0], await self._invoke(self.finish, frontier[0]))]

    async def _dfs(self) -> list[Solution]:
        stack, solutions = [State()], []
        while stack:
            state = stack.pop()
            # The paper's crossword fallback chooses the first deepest explored state.
            if len(state.thoughts) > len(self.best_state.thoughts):
                self.best_state = state
            if len(state.thoughts) == self.depth:
                solutions.append(Solution(state, await self._invoke(self.finish, state)))
                continue
            if self.expansions >= self.max_expansions:
                self.budget_exhausted = True
                break
            ranked = await self._rank(await self._expand(state))
            retained = [child for child in ranked if child.value > self.threshold]
            self.history.append(tuple(retained))
            stack.extend(reversed(retained))
        return solutions

    async def _expand(self, state: State) -> list[State]:
        self.expansions += 1
        if self.generation == "sample":
            thoughts = [await self._invoke(self.sample, state) for _ in range(self.candidates)]
        else:
            thoughts = await self._invoke(self.propose, state, self.candidates)
        return [State(state.thoughts + (text,)) for text in dict.fromkeys(thoughts)]

    async def _rank(self, states: list[State]) -> list[State]:
        if not states:
            return []
        if self.evaluate is not None:
            scores = [await self._assess(state) for state in states]
        elif self.evaluation == "vote":
            scores = [0.0] * len(states)
            for _ in range(self.evaluation_samples):
                scores[await self._invoke(self.vote, states)] += 1 / self.evaluation_samples
        else:
            scores = []
            for state in states:
                values = [
                    await self._invoke(self.value, state) for _ in range(self.evaluation_samples)
                ]
                scores.append(sum(values) / self.evaluation_samples)
        scored = [replace(state, value=score) for state, score in zip(states, scores)]
        return sorted(scored, key=lambda state: state.value, reverse=True)

    async def _assess(self, state: State) -> float:
        record = {"state": state}
        self.assessments.append(record)
        try:
            score = float(await self.evaluate(state))
            record["score"] = score
            if not math.isfinite(score):
                raise ValueError("evaluation score must be finite")
            return score
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _invoke(self, operation, *args):
        record = {"operation": operation.__name__}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Forward the Slick call, recording raw text before its decorator parses it."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("ToT prompt operations require text, not tool requests")
        return response, requests
