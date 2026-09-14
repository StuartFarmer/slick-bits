"""Plan prompt improvements with PromptAgent's error-driven Monte Carlo tree search."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Feedback:
    """Task execution evidence for one sampled training batch."""

    errors: tuple[str, ...] = ()
    correct: tuple[str, ...] = ()


@dataclass
class Node:
    """One prompt state; backed-up values are suffix sums along visited paths."""

    prompt: str
    score: float
    parent: int | None
    depth: int
    gradient: str = ""
    children: list[int] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    terminal: bool = False

    @property
    def q(self) -> float:
        return sum(self.returns) / len(self.returns) if self.returns else self.score


def checked_prompts(response: str, count: int) -> list[str]:
    items = [s.strip() for s in re.findall(r"<START>(.*?)<END>", response, re.DOTALL)]
    if (
        len(items) != count
        or response.count("<START>") != count
        or response.count("<END>") != count
        or not all(items)
    ):
        raise ValueError(f"expected {count} nonempty <START>...<END> items; response={response!r}")
    return items


class PromptAgent:
    """Own UCT selection, gradient expansion, greedy simulation and reward backup.

    evaluate measures a prompt on a fixed validation set, returning a finite
    higher-is-better score. observe executes a sampled training batch and returns
    evidence. Neither callback implements search. All failures propagate.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        observe: Callable[[str, Sequence[str]], Awaitable[Feedback]],
    ):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate
        self.observe = observe

    @prompt(template="reflect_errors.j2")
    async def reflect_errors(
        self, instruction: str, examples: Sequence[str], *, generated: str
    ) -> str:
        """Explain failures before proposing a state transition."""
        if not generated.strip():
            raise ValueError(f"empty error reflection; response={generated!r}")
        return generated.strip()

    @prompt(template="reflect_correct.j2")
    async def reflect_correct(
        self, instruction: str, examples: Sequence[str], *, generated: str
    ) -> str:
        """Extract strengths from an entirely correct training batch."""
        if not generated.strip():
            raise ValueError(f"empty success reflection; response={generated!r}")
        return generated.strip()

    @prompt(template="revise.j2")
    async def revise(
        self,
        instruction: str,
        examples: Sequence[str],
        gradient: str,
        trajectory: Sequence[str],
        count: int,
        *,
        generated: str,
    ) -> list[str]:
        """Generate successors using error feedback and the ancestor trajectory."""
        return checked_prompts(generated, count)

    @prompt(template="extend.j2")
    async def extend(
        self,
        instruction: str,
        examples: Sequence[str],
        gradient: str,
        trajectory: Sequence[str],
        count: int,
        *,
        generated: str,
    ) -> list[str]:
        """Generate successors by extending the strengths of a correct prompt."""
        return checked_prompts(generated, count)

    async def _score(self, instruction):
        self.evaluations += 1
        score = float(await self.evaluate(instruction))
        if not math.isfinite(score):
            raise ValueError("evaluation score must be finite")
        return score

    def _terminal(self, node):
        parent_score = self.nodes[node.parent].score if node.parent is not None else self.root_score
        return (
            node.terminal
            or node.depth >= self.depth_limit
            or (node.depth > self.min_depth and node.score < (self.root_score + parent_score) / 2)
        )

    def _uct(self, index):
        node = self.nodes[index]
        parent_visits = len(self.nodes[node.parent].returns)
        return node.q + self.exploration * math.sqrt(
            math.log(parent_visits + 1) / max(1, len(node.returns))
        )

    def _select(self):
        path = [0]
        while self.nodes[path[-1]].children and not self._terminal(self.nodes[path[-1]]):
            path.append(max(self.nodes[path[-1]].children, key=self._uct))
        return path

    async def _expand(self, index):
        node = self.nodes[index]
        if self._terminal(node):
            node.terminal = True
            return
        ancestors = []
        cursor = index
        while cursor is not None:
            ancestors.append(self.nodes[cursor].prompt)
            cursor = self.nodes[cursor].parent
        for _ in range(self.expand_width):
            batch = self.rng.sample(self.examples, min(len(self.examples), self.batch_size))
            self.observations += 1
            evidence = await self.observe(node.prompt, batch)
            if evidence.errors:
                examples, reflect, improve = evidence.errors, self.reflect_errors, self.revise
            else:
                examples, reflect, improve = evidence.correct, self.reflect_correct, self.extend
            self.optimizer_calls += 1
            gradient = await reflect(node.prompt, examples, provider=self.provider)
            self.optimizer_calls += 1
            proposals = await improve(
                node.prompt,
                examples,
                gradient,
                ancestors[::-1],
                self.proposals_per_batch,
                provider=self.provider,
            )
            for instruction in proposals:
                child = Node(
                    instruction, await self._score(instruction), index, node.depth + 1, gradient
                )
                child.terminal = self._terminal(child)
                node.children.append(len(self.nodes))
                self.nodes.append(child)

    async def _simulate(self, path):
        while True:
            node = self.nodes[path[-1]]
            if node.depth > self.min_depth and node.score > self.threshold:
                self.threshold = node.score
                return
            self.threshold = max(self.threshold, node.score)
            if self._terminal(node):
                return
            if not node.children:
                await self._expand(path[-1])
            if not node.children:
                node.terminal = True
                return
            path.append(max(node.children, key=lambda i: self.nodes[i].score))

    def _backpropagate(self, path):
        total = 0.0
        for index in reversed(path):
            total += self.nodes[index].score
            self.nodes[index].returns.append(total)

    async def run(
        self,
        initial_prompt: str,
        examples: Sequence[str],
        *,
        iterations: int = 12,
        expand_width: int = 3,
        proposals_per_batch: int = 1,
        batch_size: int = 5,
        depth_limit: int = 8,
        min_depth: int = 2,
        exploration: float = 2.5,
        seed: int = 0,
    ) -> dict:
        """Search and select the best node on the path with greatest mean reward."""
        self.rng = random.Random(seed)
        self.examples = list(examples)
        self.expand_width, self.proposals_per_batch = expand_width, proposals_per_batch
        self.batch_size, self.depth_limit, self.min_depth = batch_size, depth_limit, min_depth
        self.exploration = exploration
        self.evaluations = self.observations = self.optimizer_calls = 0
        self.root_score = await self._score(initial_prompt)
        self.threshold = self.root_score
        self.nodes = [Node(initial_prompt, self.root_score, None, 0)]
        paths = []
        for _ in range(iterations):
            path = self._select()
            await self._expand(path[-1])
            await self._simulate(path)
            self._backpropagate(path)
            paths.append(path)
        best_path = max(paths or [[0]], key=lambda p: sum(self.nodes[i].score for i in p) / len(p))
        return {
            "best": max((self.nodes[i] for i in best_path), key=lambda n: n.score),
            "best_global": max(self.nodes, key=lambda n: n.score),
            "best_path": best_path,
            "paths": paths,
            "nodes": self.nodes,
            "evaluations": self.evaluations,
            "observations": self.observations,
            "optimizer_calls": self.optimizer_calls,
        }
