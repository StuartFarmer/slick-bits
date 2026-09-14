"""Run sequential multi-agent debate with adaptive judging and final extraction.

Adapted from Skytliang/Multi-Agents-Debate; see README.md for provenance.
Copyright (C) 2023 The MAD Team
SPDX-License-Identifier: GPL-3.0-or-later
"""

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Judgment(BaseModel, extra="forbid"):
    """An empty answer means the discriminative judge has not decided."""

    answer: Annotated[str, StringConstraints(strip_whitespace=True)]
    reason: Text


@dataclass(frozen=True)
class Turn:
    round: int
    speaker: Literal["affirmative", "negative"]
    content: str


@dataclass(frozen=True)
class Result:
    answer: str
    reason: str
    rounds: int
    stop_reason: Literal["judge", "round_limit"]
    turns: tuple[Turn, ...]
    calls: int


def nonblank(generated: str) -> str:
    """Reject empty generated prose without changing artifact whitespace."""
    if not generated.strip():
        raise ValueError("generated text is blank")
    return generated


class MultiAgentDebate:
    """Own a two-debater MAD run and its explicitly rendered histories.

    No retries, tools, or execution of generated content. All errors propagate;
    completed turns, judgments, and raw call records survive failure. Each run
    resets records. Use one run at a time per instance and configure Slick's
    template root before use. Model configuration belongs to the caller.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        negative_provider: Provider | None = None,
        judge_provider: Provider | None = None,
        criteria: str = "Correctness and consistency with the task constraints",
        answer_format: str = "A complete answer satisfying the task",
    ):
        self.task, self.criteria, self.answer_format = task, criteria, answer_format
        self.providers = {
            "affirmative": provider,
            "negative": provider if negative_provider is None else negative_provider,
            "judge": provider if judge_provider is None else judge_provider,
        }
        self.turns: list[Turn] = []
        self.judgments: list[Judgment] = []
        self.calls: list[dict] = []
        self.round, self.max_rounds = 0, 3

    @prompt(template="affirm.j2")
    async def affirm(self, *, generated: str) -> str:
        """Generate the affirmative opening answer."""
        return nonblank(generated)

    @prompt(template="oppose.j2")
    async def oppose(self, *, generated: str) -> str:
        """Challenge the affirmative opening with an alternative answer."""
        return nonblank(generated)

    @prompt(template="rebut.j2")
    async def rebut(self, speaker: str, *, generated: str) -> str:
        """Respond to the latest opponent while retaining the debate history."""
        return nonblank(generated)

    @prompt(template="discriminate.j2", output_type=Judgment)
    async def discriminate(self, *, generated: Judgment) -> Judgment:
        """Decide whether a final answer has emerged after a complete round."""
        return generated

    @prompt(template="extract.j2")
    async def extract(self, *, generated: str) -> str:
        """Collect answer candidates from the entire debate for the final judge."""
        return nonblank(generated)

    @prompt(template="select.j2", output_type=Judgment)
    async def select(self, candidates: str, *, generated: Judgment) -> Judgment:
        """Conclude the debate; the final judge cannot abstain."""
        nonblank(generated.answer)
        return generated

    async def run(self, *, max_rounds: int = 3) -> Result:
        """Stop on a judge answer, otherwise extract and select at the round limit.

        Each complete round costs three provider calls; fallback costs two more.
        A zero-round run goes directly to candidate generation and selection.
        A judge's decision is not an independently verified correctness result.
        """
        self.turns, self.judgments, self.calls = [], [], []
        self.round, self.max_rounds = 0, max_rounds
        stop_reason = "round_limit"
        for self.round in range(1, max_rounds + 1):
            await self._debate_round()
            judgment = await self._invoke(self.discriminate, "judge")
            self.judgments.append(judgment)
            if judgment.answer:
                stop_reason = "judge"
                break
        else:
            judgment = await self._conclude()
        return Result(
            judgment.answer,
            judgment.reason,
            self.round,
            stop_reason,
            tuple(self.turns),
            len(self.calls),
        )

    async def _debate_round(self):
        for speaker, opening in (("affirmative", self.affirm), ("negative", self.oppose)):
            if self.round == 1:
                answer = await self._invoke(opening, speaker)
            else:
                answer = await self._invoke(self.rebut, speaker, speaker)
            # Append immediately: the next speaker must see this round's latest answer.
            self.turns.append(Turn(self.round, speaker, answer))

    async def _conclude(self) -> Judgment:
        candidates = await self._invoke(self.extract, "judge")
        return await self._invoke(self.select, "judge", candidates)

    async def _invoke(self, operation, speaker, *args):
        record = {"operation": operation.__name__, "speaker": speaker, "round": self.round}
        self.calls.append(record)
        try:
            return await operation(*args, provider=self)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def acall(self, context, *, tools=None, tool_results=None):
        """Record raw provider output before Slick parsing or postprocessing."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.providers[record["speaker"]].acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("MAD prompt operations require text, not tool requests")
        return response, requests
