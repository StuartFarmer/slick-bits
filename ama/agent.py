"""Ask and answer through reusable prompt chains, then aggregate their votes."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints
from slick import prompt
from slick.providers import Provider

from .aggregation import WeakSupervision, WSConfig, majority_vote


@dataclass(frozen=True)
class Example:
    text: str
    context: str = ""


@dataclass(frozen=True)
class Chain:
    """A fixed functional prompt pair reused across every example."""

    style: Literal["yes_no", "wh", "cloze", "identity"] = "yes_no"
    question_examples: str = ""
    answer_examples: str = ""


@dataclass(frozen=True)
class Trace:
    example: Example
    chain: int
    question: str
    answer: str


class Label(BaseModel, extra="forbid"):
    label: str | None


class OpenAnswer(BaseModel, extra="forbid"):
    answer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | None


@dataclass(frozen=True)
class Result:
    predictions: tuple[str | None, ...]
    votes: tuple[tuple[str | None, ...], ...]
    traces: tuple[Trace, ...]
    calls: int
    probabilities: tuple[tuple[float, ...], ...] = ()
    dependencies: tuple[tuple[int, int], ...] = ()


Mapper = Callable[[Example, Trace], Awaitable[str | None]]
DEFAULT_CHAINS = (Chain("yes_no"), Chain("wh"), Chain("cloze"))


class AMA:
    """Own independent generation calls and batch aggregation; no gold labels.

    Configure Slick's template root once before use. Errors propagate without
    retries. Calls count attempted generations; raw prose and completed QA
    traces survive failure. Provider logging must retain rejected mapping JSON.
    One run at a time per instance; each run resets records and the label model.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        labels: Sequence[str] = (),
        chains: Sequence[Chain] = DEFAULT_CHAINS,
        map_answer: Mapper | None = None,
        ws_config: WSConfig = WSConfig(),
    ):
        self.task = task
        self.provider = provider
        self.labels = tuple(labels)
        self.chains = tuple(chains)
        self.map_answer = map_answer
        self.ws_config = ws_config
        self.calls = 0
        self.generations: list[str] = []
        self.traces: list[Trace] = []
        self.label_model: WeakSupervision | None = None

    def _record_text(self, generated: str, operation: str) -> str:
        self.generations.append(generated)
        text = generated.strip()
        if not text:
            raise ValueError(f"blank {operation}; response={generated!r}")
        return text

    @prompt(template="question_yes_no.j2")
    async def question_yes_no(self, example: Example, chain: Chain, *, generated: str) -> str:
        return self._record_text(generated, "question")

    @prompt(template="question_wh.j2")
    async def question_wh(self, example: Example, chain: Chain, *, generated: str) -> str:
        return self._record_text(generated, "question")

    @prompt(template="question_cloze.j2")
    async def question_cloze(self, example: Example, chain: Chain, *, generated: str) -> str:
        return self._record_text(generated, "question")

    @prompt(template="answer.j2")
    async def answer(self, example: Example, question: str, chain: Chain, *, generated: str) -> str:
        return self._record_text(generated, "answer")

    @prompt(template="map_label.j2", output_type=Label)
    async def map_label(self, trace: Trace, *, generated: Label) -> Label:
        if generated.label is not None and generated.label not in self.labels:
            raise ValueError(f"unknown generated label: {generated.label!r}")
        return generated

    @prompt(template="map_answer.j2", output_type=OpenAnswer)
    async def map_open_answer(self, trace: Trace, *, generated: OpenAnswer) -> OpenAnswer:
        return generated

    async def collect(self, examples: Sequence[Example]) -> tuple[tuple[str | None, ...], ...]:
        """Collect a stable example-by-chain matrix, appending execution records."""
        questioners = {
            "yes_no": self.question_yes_no,
            "wh": self.question_wh,
            "cloze": self.question_cloze,
        }
        rows = []
        for example in examples:
            votes = []
            for index, chain in enumerate(self.chains):
                question = example.text
                if chain.style != "identity":
                    self.calls += 1
                    question = await questioners[chain.style](
                        example, chain, provider=self.provider
                    )
                self.calls += 1
                answer = await self.answer(example, question, chain, provider=self.provider)
                trace = Trace(example, index, question, answer)
                self.traces.append(trace)
                if self.map_answer is not None:
                    vote = await self.map_answer(example, trace)
                elif self.labels:
                    self.calls += 1
                    vote = (await self.map_label(trace, provider=self.provider)).label
                else:
                    self.calls += 1
                    vote = (await self.map_open_answer(trace, provider=self.provider)).answer
                if self.labels and vote is not None and vote not in self.labels:
                    raise ValueError(f"mapped answer is outside the label set: {vote!r}")
                votes.append(vote)
            rows.append(tuple(votes))
        return tuple(rows)

    async def run(
        self,
        examples: Sequence[Example],
        *,
        unlabeled: Sequence[Example] = (),
        aggregation: Literal["auto", "majority", "weak_supervision"] = "auto",
    ) -> Result:
        """Fit WS on additional unlabeled examples plus target votes (transductive).

        Auto uses WS for fixed labels and majority vote for open answers. WS
        requires a batch with at least three chains; it never silently falls
        back to majority vote. Extra unlabeled examples are used only for WS.
        """
        self.calls, self.generations, self.traces = 0, [], []
        self.label_model = None
        if not examples:
            return Result((), (), (), 0)
        use_ws = aggregation == "weak_supervision" or (aggregation == "auto" and bool(self.labels))
        votes = await self.collect(examples)
        if not use_ws:
            return Result(tuple(map(majority_vote, votes)), votes, tuple(self.traces), self.calls)
        extra_votes = await self.collect(unlabeled)
        self.label_model = WeakSupervision(self.labels, self.ws_config)
        self.label_model.fit(extra_votes + votes)
        probabilities = self.label_model.predict_proba(votes)
        predictions = tuple(self.labels[index] for index in probabilities.argmax(axis=1))
        return Result(
            predictions,
            votes,
            tuple(self.traces),
            self.calls,
            tuple(map(tuple, probabilities.tolist())),
            self.label_model.dependencies,
        )
