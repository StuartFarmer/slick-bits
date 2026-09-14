"""Refine hierarchical prompts through structural and error-guided critic/actor edits."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class InvalidAction(ValueError):
    """A generated edit references an invalid tree location or operation."""


class Section(BaseModel, extra="forbid"):
    title: Text
    body: str = ""
    examples: list[Text] = Field(default_factory=list, max_length=6)
    children: list["Section"] = Field(default_factory=list)


class Feedback(BaseModel, extra="forbid"):
    references: list[list[Text]] = Field(min_length=1)
    feedback: Text


class Assessment(BaseModel, extra="forbid"):
    feedback: list[Feedback]


class Action(BaseModel, extra="forbid"):
    kind: Literal["rephrase", "create", "delete", "merge", "reorder", "examples"]
    path: list[Text]
    body: str = ""
    section: Section | None = None
    sources: list[list[Text]] = Field(default_factory=list)
    order: list[Text] = Field(default_factory=list)
    position: Annotated[int, Field(strict=True)] = 0
    instruction: str = ""


class Actions(BaseModel, extra="forbid"):
    actions: list[Action]


class Examples(BaseModel, extra="forbid"):
    examples: list[Text] = Field(max_length=6)


def render_sections(sections: Sequence[Section], level: int = 1) -> str:
    return "\n\n".join(
        "\n".join(
            ["#" * level + " " + node.title, node.body]
            + ["Example: " + example for example in node.examples]
            + [render_sections(node.children, level + 1)]
        ).strip()
        for node in sections
    )


class SCULPT:
    """Keep tree edits local and select the strongest parent/child beam.

    Supply a hierarchy of sections, an async scalar evaluator, and an async
    errors(prompt) callback returning observed failures as plain dictionaries.
    Evaluation scores are finite, higher-is-better, and cached within each run.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        errors: Callable[[str], Awaitable[Sequence[dict]]],
        render: Callable[[Sequence[Section]], str] = render_sections,
    ):
        self.task, self.provider = task, provider
        self.evaluate, self.errors, self.render = evaluate, errors, render

    @prompt(template="assess_structure.j2", output_type=Assessment)
    async def assess_structure(self, tree: list[dict], *, generated: Assessment) -> Assessment:
        return generated

    @prompt(template="assess_errors.j2", output_type=Assessment)
    async def assess_errors(
        self, tree: list[dict], errors: list[dict], *, generated: Assessment
    ) -> Assessment:
        return generated

    @prompt(template="act.j2", output_type=Actions)
    async def act(self, tree: list[dict], feedback: list[dict], *, generated: Actions) -> Actions:
        return generated

    @prompt(template="update_examples.j2", output_type=Examples)
    async def update_examples(
        self, section: dict, instruction: str, *, generated: Examples
    ) -> Examples:
        return generated

    @staticmethod
    def _node(tree, path):
        siblings = tree
        for title in path:
            matches = [node for node in siblings if node.title == title]
            if len(matches) != 1:
                raise InvalidAction(f"section path is absent or ambiguous: {path!r}")
            node = matches[0]
            siblings = node.children
        if not path:
            raise InvalidAction("an action requires a section path")
        return node

    def _siblings(self, tree, parent_path):
        return self._node(tree, parent_path).children if parent_path else tree

    async def _apply(self, tree, actions):
        changed = [node.model_copy(deep=True) for node in tree]
        for action in actions.actions:
            path = action.path
            if action.kind == "rephrase":
                self._node(changed, path).body = action.body
            elif action.kind == "delete":
                self._siblings(changed, path[:-1]).remove(self._node(changed, path))
            elif action.kind in ("create", "merge"):
                if action.section is None:
                    raise InvalidAction("create/merge requires section content")
                target = self._siblings(changed, path)
                if not 0 <= action.position <= len(target):
                    raise InvalidAction("new section position is outside its parent")
                if action.kind == "merge":
                    if len(action.sources) < 2:
                        raise InvalidAction("merge requires at least two source sections")
                    if len({tuple(path) for path in action.sources}) != len(action.sources):
                        raise InvalidAction("merge sources must be distinct")
                    sources = [
                        (self._siblings(changed, p[:-1]), self._node(changed, p))
                        for p in action.sources
                    ]
                    if any(parent is not target for parent, _ in sources):
                        raise InvalidAction("merge sources must be siblings at the destination")
                    for parent, source in sources:
                        parent.remove(source)
                if any(node.title == action.section.title for node in target):
                    raise InvalidAction("new section title already exists at the destination")
                target.insert(action.position, action.section.model_copy(deep=True))
            elif action.kind == "reorder":
                siblings = self._siblings(changed, path)
                by_title = {node.title: node for node in siblings}
                if len(action.order) != len(siblings) or set(action.order) != set(by_title):
                    raise InvalidAction("reorder must list every sibling exactly once")
                siblings[:] = [by_title[title] for title in action.order]
            else:
                node = self._node(changed, path)
                self.optimizer_calls += 1
                examples = await self.update_examples(
                    node.model_dump(), action.instruction, provider=self.provider
                )
                node.examples = examples.examples
        if not self.render(changed).strip():
            raise InvalidAction("actions removed the whole prompt")
        return changed

    async def _revise(self, tree, feedback, stage):
        self.optimizer_calls += 1
        actions = await self.act(
            [node.model_dump() for node in tree],
            [item.model_dump() for item in feedback],
            provider=self.provider,
        )
        try:
            changed = await self._apply(tree, actions)
        except InvalidAction as exc:
            self.history.append(
                {"stage": stage, "actions": actions.model_dump(), "rejection": str(exc)}
            )
            return None
        self.history.append({"stage": stage, "actions": actions.model_dump(), "rejection": ""})
        return changed

    def _groups(self, feedback, tree):
        groups = {}
        for item in feedback:
            headers = set()
            for path in item.references:
                self._node(tree, path)
                headers.add(path[0])
            for header in sorted(headers):
                group = groups.setdefault(header, [])
                if item not in group:
                    group.append(item)
        return list(groups.values())

    async def _expand(self, tree, error_batches, errors_per_batch):
        self.optimizer_calls += 1
        structure = await self.assess_structure(
            [node.model_dump() for node in tree], provider=self.provider
        )
        proposals = []
        if structure.feedback:
            revised = await self._revise(tree, structure.feedback, "structure")
            if revised is not None:
                tree = revised
                proposals.append(tree)
        self.error_evaluations += 1
        errors = list(await self.errors(self.render(tree)))
        feedback = []
        for _ in range(error_batches):
            if not errors:
                break
            sampled = self.rng.sample(errors, min(len(errors), errors_per_batch))
            self.optimizer_calls += 1
            assessment = await self.assess_errors(
                [node.model_dump() for node in tree], sampled, provider=self.provider
            )
            feedback.extend(assessment.feedback)
        try:
            groups = self._groups(feedback, tree)
        except InvalidAction as exc:
            self.history.append({"stage": "error references", "rejection": str(exc)})
            return proposals
        for group in groups:
            revised = await self._revise(tree, group, "errors")
            if revised is not None:
                proposals.append(revised)
        return proposals

    async def _score(self, tree):
        text = self.render(tree)
        if text not in self.cache:
            self.evaluations += 1
            score = float(await self.evaluate(text))
            if not math.isfinite(score):
                raise ValueError("fitness must be finite")
            self.cache[text] = score
        return {"tree": tree, "prompt": text, "score": self.cache[text]}

    async def run(
        self,
        sections: Sequence[Section],
        *,
        iterations: int = 5,
        beam_size: int = 4,
        error_batches: int = 2,
        errors_per_batch: int = 4,
        seed: int = 0,
    ) -> dict:
        self.rng, self.cache, self.history = random.Random(seed), {}, []
        self.evaluations = self.error_evaluations = self.optimizer_calls = 0
        beam = [await self._score([node.model_copy(deep=True) for node in sections])]
        for _ in range(iterations):
            expanded = beam.copy()
            seen = {item["prompt"] for item in beam}
            for parent in beam:
                for tree in await self._expand(parent["tree"], error_batches, errors_per_batch):
                    text = self.render(tree)
                    if text not in seen:
                        seen.add(text)
                        expanded.append(await self._score(tree))
            beam = sorted(expanded, key=lambda item: item["score"], reverse=True)[:beam_size]
        return {
            "best": beam[0],
            "beam": beam,
            "history": self.history.copy(),
            "evaluations": self.evaluations,
            "error_evaluations": self.error_evaluations,
            "optimizer_calls": self.optimizer_calls,
        }
