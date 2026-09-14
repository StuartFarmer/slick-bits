"""Task-facet learning through grouped feedback, conceptual edits and beam search."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel
from slick import prompt
from slick.providers import Provider


class Groups(BaseModel):
    groups: list[list[int]]


def checked(text):
    if not text.strip():
        raise ValueError(f"empty generated text; response={text!r}")
    return text.strip()


class UniPrompt:
    """`errors(prompt, batch)` returns serialized failures including reference outputs.

    `evaluate(prompt)` scores on a separate validation set. Larger is better.
    Training data are opaque serialized examples interpreted by the caller.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        errors: Callable[[str, tuple[str, ...]], Awaitable[list[str]]],
        evaluate: Callable[[str], Awaitable[float]],
    ):
        self.task, self.provider = task, provider
        self.errors, self.evaluate = errors, evaluate

    @prompt(template="feedback.j2")
    async def feedback(
        self, *, instruction: str, failures: list[str], history: list, generated: str
    ) -> str:
        return checked(generated)

    @prompt(template="cluster.j2", output_type=Groups)
    async def cluster(
        self, *, feedbacks: list[str], group_count: int, generated: Groups
    ) -> list[list[int]]:
        indices = [index for group in generated.groups for index in group]
        if sorted(indices) != list(range(len(feedbacks))) or not all(generated.groups):
            raise ValueError(
                f"generated groups must partition all feedback indices; response={generated!r}"
            )
        return generated.groups

    @prompt(template="aggregate.j2")
    async def aggregate(self, *, feedbacks: list[str], failures: list[str], generated: str) -> str:
        return checked(generated)

    @prompt(template="add.j2")
    async def add(self, *, feedbacks: list[str], failures: list[str], generated: str) -> str:
        return checked(generated)

    @prompt(template="delete.j2")
    async def delete(self, *, feedbacks: list[str], failures: list[str], generated: str) -> str:
        return checked(generated)

    @prompt(template="set.j2")
    async def set_content(
        self, *, feedbacks: list[str], failures: list[str], generated: str
    ) -> str:
        return checked(generated)

    @prompt(template="apply.j2")
    async def apply(self, *, instruction: str, edits: str, generated: str) -> str:
        return checked(generated)

    async def _measure(self, instruction):
        self.evaluations += 1
        score = float(await self.evaluate(instruction))
        if not math.isfinite(score):
            raise ValueError("measured validation score must be finite")
        return {"prompt": instruction, "score": score}

    async def _failures(self, instruction, batch):
        self.error_calls += 1
        return await self.errors(instruction, tuple(batch))

    async def _group(self, instruction, training, group_count):
        feedbacks = []
        for example in training:
            failures = await self._failures(instruction, [example])
            if failures:
                self.optimizer_calls += 1
                feedbacks.append(
                    await self.feedback(
                        instruction=instruction,
                        failures=failures,
                        history=[],
                        provider=self.provider,
                    )
                )
            else:
                feedbacks.append("Correct response")
        self.optimizer_calls += 1
        return await self.cluster(
            feedbacks=feedbacks, group_count=group_count, provider=self.provider
        )

    async def _edit_group(self, training, indices, history, batch_size, minibatch_size, epsilon):
        parent = self.beam[0]
        selected = self.rng.sample(indices, min(len(indices), batch_size * minibatch_size))
        feedbacks, failures = [], []
        for start in range(0, len(selected), minibatch_size):
            batch = [training[index] for index in selected[start : start + minibatch_size]]
            errors = await self._failures(parent["prompt"], batch)
            if errors:
                self.optimizer_calls += 1
                feedbacks.append(
                    await self.feedback(
                        instruction=parent["prompt"],
                        failures=errors,
                        history=history,
                        provider=self.provider,
                    )
                )
                failures.extend(errors)
        if not feedbacks:
            return
        operation = self.aggregate
        if self.rng.random() < epsilon:
            operation = self.rng.choice([self.add, self.delete, self.set_content])
        self.optimizer_calls += 1
        edits = await operation(feedbacks=feedbacks, failures=failures, provider=self.provider)
        children = []
        for candidate in self.beam:
            self.optimizer_calls += 1
            instruction = await self.apply(
                instruction=candidate["prompt"], edits=edits, provider=self.provider
            )
            child = await self._measure(instruction)
            children.append(child)
            history.append(
                {
                    "edits": edits,
                    "parent": candidate["prompt"],
                    "delta": child["score"] - candidate["score"],
                }
            )
        self.beam = sorted(self.beam + children, key=lambda item: item["score"], reverse=True)[
            : self.width
        ]

    async def run(
        self,
        initial: str,
        training: Sequence[str],
        *,
        epochs=5,
        beam_width=3,
        batch_size=7,
        minibatch_size=5,
        epsilon=0.5,
        group_count=5,
        group_every=2,
        seed=0,
    ) -> dict:
        self.rng, self.width = random.Random(seed), beam_width
        self.optimizer_calls = self.error_calls = self.evaluations = 0
        self.beam = [await self._measure(initial)]
        history = []
        for epoch in range(epochs):
            if epoch % group_every == 0:
                groups = await self._group(self.beam[0]["prompt"], training, group_count)
                group_history = [[] for _ in groups]
            for indices, edits in zip(groups, group_history, strict=True):
                await self._edit_group(
                    training, indices, edits, batch_size, minibatch_size, epsilon
                )
            history.append(
                {
                    "epoch": epoch,
                    "groups": groups,
                    "beam": self.beam.copy(),
                    "edits": [items.copy() for items in group_history],
                }
            )
        return {
            "best": self.beam[0],
            "beam": self.beam,
            "history": history,
            "optimizer_calls": self.optimizer_calls,
            "error_calls": self.error_calls,
            "evaluations": self.evaluations,
        }
