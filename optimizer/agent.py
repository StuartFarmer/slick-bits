"""Propose, evaluate, and revise textual candidates using explicit feedback."""

import json
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints
from slick import Session, prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Proposal(BaseModel, extra="forbid", frozen=True):
    description: Text
    content: Text


class Individual(Proposal):
    fitness: float = Field(allow_inf_nan=False)


class Optimizer:
    """Own one improvement dialogue with caller-supplied task, feedback and scoring."""

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str], Awaitable[float]],
        *,
        maximize: bool = True,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.maximize = maximize
        self.score_direction = "Higher" if maximize else "Lower"
        self.history: list[dict] = []

    @prompt(template="propose.j2", output_type=Proposal)
    async def propose(self, *, generated: Proposal) -> Proposal:
        """Propose an initial candidate appropriate to the task."""
        return generated

    @prompt(template="revise.j2", output_type=Proposal)
    async def revise(
        self,
        incumbent: Individual,
        feedback: str,
        history: str,
        *,
        generated: Proposal,
    ) -> Proposal:
        """Revise the current best using supplied feedback and previous outcomes."""
        return generated

    async def run(
        self,
        feedback: Sequence[str] = (),
        *,
        session: Session | None = None,
    ) -> Individual:
        """Make one proposal and one revision attempt per feedback item.

        Keep only strict fitness improvements. Malformed proposals, invalid scores
        and timeouts reject a revision; an invalid initial proposal aborts. Other
        errors propagate. Calls are sequential; Session ownership stays with the caller.
        """
        execution = {"session": session} if session is not None else {"provider": self.provider}
        self.history = []
        incumbent = await self._turn(None, None, execution)
        if incumbent is None:
            raise RuntimeError("initial proposal failed; inspect history")
        for instruction in feedback:
            incumbent = await self._turn(incumbent, instruction, execution)
        return incumbent

    async def _assess(self, proposal: Proposal) -> Individual:
        fitness = float(await self.evaluate(proposal.content))
        if not math.isfinite(fitness):
            raise ValueError("fitness must be finite")
        return Individual(**proposal.model_dump(), fitness=fitness)

    async def _turn(self, incumbent, feedback, execution) -> Individual | None:
        record = {"turn": len(self.history), "feedback": feedback, "accepted": False}
        try:
            if incumbent is None:
                proposal = await self.propose(**execution)
            else:
                proposal = await self.revise(
                    incumbent, feedback, json.dumps(self.history), **execution
                )
            record["proposal"] = proposal.model_dump()
            candidate = await self._assess(proposal)
            record["fitness"] = candidate.fitness
            if incumbent is None or (
                candidate.fitness > incumbent.fitness
                if self.maximize
                else candidate.fitness < incumbent.fitness
            ):
                incumbent = candidate
                record["accepted"] = True
        except (ValueError, TimeoutError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.history.append(record)
        return incumbent
