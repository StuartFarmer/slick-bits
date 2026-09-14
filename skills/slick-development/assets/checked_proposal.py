"""Propose a candidate and revise it using caller-supplied feedback and evaluation."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints
from slick import Session, prompt

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Proposal(BaseModel, extra="forbid"):
    description: Text
    content: Text


class Assessment(BaseModel, extra="forbid"):
    proposal: Proposal
    score: float = Field(allow_inf_nan=False)


class ProposalDesigner:
    """Keep task context; the caller owns evaluation and provider retries."""

    def __init__(self, task: str, provider, evaluate: Callable[[str], Awaitable[float]]):
        self.task = task
        self.provider = provider
        self.evaluate = evaluate

    @prompt(template="propose.j2", output_type=Proposal)
    async def propose(self, feedback: str, *, generated: Proposal) -> Proposal:
        return generated

    @prompt(template="revise.j2", output_type=Proposal)
    async def revise(self, draft: Proposal, feedback: str, *, generated: Proposal) -> Proposal:
        return generated

    async def assess(self, proposal: Proposal) -> Assessment:
        return Assessment(proposal=proposal, score=await self.evaluate(proposal.content))

    async def run(self, feedback: Sequence[str], *, session: Session | None = None) -> Assessment:
        """Propose, assess, and revise; retain strict score improvements."""
        first, *revisions = feedback
        execution = {"session": session} if session is not None else {"provider": self.provider}
        best = await self.assess(await self.propose(first, **execution))
        for message in revisions:
            proposal = await self.revise(best.proposal, message, **execution)
            assessment = await self.assess(proposal)
            if assessment.score > best.score:
                best = assessment
        return best
