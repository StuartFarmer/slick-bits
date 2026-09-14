"""Refine independent answers through repeated exchanges of previous-round peer responses."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from slick import Session, prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Result:
    """Responses retain provider order; rounds include the initial independent answers."""

    responses: tuple[str, ...]
    rounds: tuple[tuple[str, ...], ...]
    calls: int


class MultiAgentDebate:
    """Own one conversation per provider entry and a fixed number of debate rounds.

    Repeat a provider in the sequence to sample independent instances of one model.
    Calls run sequentially, but all revisions read the same previous-round snapshot.
    Errors propagate without retries; calls, generations, sessions, and completed
    rounds remain inspectable after failure. Each run resets this state.
    """

    def __init__(
        self,
        task: str,
        providers: Sequence[Provider],
        *,
        debate_rounds: int = 2,
        style: Literal["short", "long"] = "long",
        summarizer: Provider | None = None,
    ):
        self.task = task
        self.providers = tuple(providers)
        self.debate_rounds = debate_rounds
        self.style = style
        self.summarizer = summarizer
        self.sessions: list[Session] = []
        self.rounds: list[tuple[str, ...]] = []
        self.generations: list[str] = []
        self.calls = 0

    def _record(self, generated: str) -> str:
        self.generations.append(generated)
        if not generated.strip():
            raise ValueError("debate generated an empty response")
        return generated

    @prompt(template="initialize.j2", max_turns=1)
    async def initialize(self, *, generated: str) -> str:
        """Generate an independent answer using the caller's task prompt."""
        return self._record(generated)

    @prompt(template="revise_long.j2", max_turns=1)
    async def revise_long(self, peers: tuple[str, ...], *, generated: str) -> str:
        """Treat peer responses as additional advice."""
        return self._record(generated)

    @prompt(template="revise_short.j2", max_turns=1)
    async def revise_short(self, peers: tuple[str, ...], *, generated: str) -> str:
        """Encourage faster agreement with peer responses."""
        return self._record(generated)

    @prompt(template="reflect.j2", max_turns=1)
    async def reflect(self, *, generated: str) -> str:
        """Double-check the previous answer when there are no peers."""
        return self._record(generated)

    @prompt(template="summarize.j2")
    async def summarize(self, peers: tuple[str, ...], *, generated: str) -> str:
        """Compress this agent's peer responses without adding conversation history."""
        return self._record(generated)

    async def _initialize_agents(self) -> tuple[str, ...]:
        responses = []
        for session in self.sessions:
            self.calls += 1
            responses.append(await self.initialize(session=session))
        return tuple(responses)

    async def _debate_round(self, previous: tuple[str, ...]) -> tuple[str, ...]:
        revise = {"short": self.revise_short, "long": self.revise_long}[self.style]
        responses = []
        for index, session in enumerate(self.sessions):
            peers = previous[:index] + previous[index + 1 :]
            if peers and self.summarizer is not None:
                self.calls += 1
                peers = (await self.summarize(peers, provider=self.summarizer),)
            self.calls += 1
            response = (
                await revise(peers, session=session)
                if peers
                else await self.reflect(session=session)
            )
            responses.append(response)
        return tuple(responses)

    async def run(self) -> Result:
        """Run initialization followed by debate_rounds revisions, without early stopping.

        No evaluator or judge participates in the paper's debate loop. Callers own
        answer extraction, selection, and external evaluation of the final responses.
        Use one run at a time per instance and configure Slick's template root first.
        """
        self.sessions = [Session(provider=provider) for provider in self.providers]
        self.rounds = []
        self.generations = []
        self.calls = 0
        self.rounds.append(await self._initialize_agents())
        for _ in range(self.debate_rounds):
            self.rounds.append(await self._debate_round(self.rounds[-1]))
        return Result(self.rounds[-1], tuple(self.rounds), self.calls)
