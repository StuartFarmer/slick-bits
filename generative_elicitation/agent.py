"""Elicit preferences through informative questions or generated edge cases."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Response(BaseModel, extra="forbid"):
    text: Text


class GenerativeElicitation:
    def __init__(
        self,
        task: str,
        provider: Provider,
        oracle: Callable[[str], Awaitable[str]],
        *,
        representation: str = "instruction",
    ):
        self.task, self.provider, self.oracle = task, provider, oracle
        self.representation = representation

    @prompt(template="question.j2", output_type=Response)
    async def question(
        self, history: list[tuple[str, str]], question_type: str, *, generated: Response
    ) -> str:
        return generated.text

    @prompt(template="edge_case.j2", output_type=Response)
    async def edge_case(self, history: list[tuple[str, str]], *, generated: Response) -> str:
        return generated.text

    @prompt(template="hypothesis.j2", output_type=Response)
    async def hypothesis(self, history: list[tuple[str, str]], *, generated: Response) -> str:
        return generated.text

    @prompt(template="predict.j2", output_type=Response)
    async def predict(
        self, history: list[tuple[str, str]], query: str, *, generated: Response
    ) -> str:
        return generated.text

    async def run(
        self,
        *,
        queries: int = 10,
        mode: Literal["question", "edge_case"] = "question",
        question_type: str = "open-ended question",
        history: Sequence[tuple[str, str]] = (),
    ) -> dict:
        interactions = list(history)
        for _ in range(queries):
            if mode == "edge_case":
                query = await self.edge_case(interactions, provider=self.provider)
            else:
                query = await self.question(interactions, question_type, provider=self.provider)
            answer = await self.oracle(query)
            interactions.append((query, answer))
        hypothesis = await self.hypothesis(interactions, provider=self.provider)
        return {"hypothesis": hypothesis, "history": interactions, "oracle_calls": queries}
