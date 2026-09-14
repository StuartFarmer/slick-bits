"""Run evidence-grounded debate with inference-time persuasion optimization.

Adapted from UCL DARK's llm_debate (MIT); see README.md and LICENSE.
"""

import asyncio
import html
import math
import re
import string
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from slick import Prompt, prompt
from slick.providers import Provider

Logprobs = Callable[[str], Awaitable[Mapping[str, float]]]
QuoteVerifier = Callable[[str], Awaitable[bool]]
_QUOTE_TAG = re.compile(r"<(/?)(?:[vu][_.])?quote\s*>", re.IGNORECASE)
_THINKING_TAG = re.compile(r"<(/?)thinking\b[^>]*>", re.IGNORECASE)
DUMMY_ARGUMENT = "My answer is the best choice and my opponent is wrong."


@dataclass(frozen=True)
class Problem:
    question: str
    answers: tuple[str, str]
    evidence: str
    criteria: str = "Correctness supported by the supplied evidence"


@dataclass(frozen=True)
class Debater:
    provider: Provider
    best_of: int = 1
    critiques: int = 0
    critic: Provider | None = None


@dataclass(frozen=True)
class Turn:
    round: int
    side: int | None
    text: str


@dataclass(frozen=True)
class View:
    """Public judge input deliberately has no private evidence field."""

    question: str
    answers: tuple[str, str]
    criteria: str
    transcript: str
    protocol: str


@dataclass
class Call:
    """Per-call adapter keeps concurrent raw responses attached to their operation."""

    operation: str
    side: int | None
    round: int
    source: Provider | None = None
    context: str = ""
    response: str | Mapping[str, float] | None = None
    error: str | None = None

    async def acall(self, context, *, tools=None, tool_results=None):
        self.context = context
        response, requests = await self.source.acall(
            context,
            tools=tools,
            tool_results=tool_results,
        )
        self.response = response
        if requests:
            raise ValueError("Debate generation expects text, not tool requests")
        return response, []


@dataclass(frozen=True)
class Result:
    choice: int | None
    answer: str | None
    approval: tuple[float, float]
    votes: tuple[int, ...]
    turns: tuple[Turn, ...]
    calls: int


class CandidateError(ValueError):
    """Generated content cannot safely enter the public transcript."""


def normalize(text: str) -> str:
    """Match the upstream case, punctuation, and whitespace normalization."""
    text = text.translate(str.maketrans({"”": '"', "“": '"', "’": "'", "‘": "'"}))
    return " ".join(text.translate(str.maketrans("", "", string.punctuation)).lower().split())


def strip_thinking(response: str) -> str:
    """Remove nested private blocks, including an unfinished trailing block."""
    pieces, end, depth = [], 0, 0
    for match in _THINKING_TAG.finditer(response):
        if depth == 0:
            pieces.append(response[end : match.start()])
        if match[1]:
            if depth == 0:
                raise CandidateError("Unmatched closing thinking tag")
            depth -= 1
        else:
            depth += 1
        end = match.end()
    if depth == 0:
        pieces.append(response[end:])
    return "".join(pieces)


def extract(response: str, tag: str) -> str:
    """Require a single nonempty public block; never promote a scratchpad."""
    public = strip_thinking(response)
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", public, re.DOTALL)
    if len(matches) != 1 or not matches[0].strip() or f"<{tag}>" in matches[0]:
        raise CandidateError(f"Expected one nonempty <{tag}> block")
    return matches[0].strip()


def truncate(text: str, limit: int) -> str:
    """Upstream-style word truncation, closing any partially retained quote."""
    text = _QUOTE_TAG.sub(r"<\1quote>", text)
    words = text.split()
    if len(words) <= limit:
        return text
    result = " ".join(words[:limit])
    if result.count("<quote>") > result.count("</quote>"):
        result += "</quote>"
    return result + "...<TRUNCATED>"


async def verify_quotes(text: str, verify: QuoteVerifier) -> str:
    """Recheck all quote tags; escape other markup so it cannot forge a verdict."""
    text = _QUOTE_TAG.sub(r"<\1quote>", strip_thinking(text))
    pieces, end = [], 0
    for match in re.finditer(r"<quote>(.*?)</quote>", text, re.DOTALL):
        pieces.append(html.escape(text[end : match.start()], quote=False))
        quote = match[1]
        verified = bool(normalize(quote)) and await verify(quote)
        tag = "v_quote" if verified else "u_quote"
        pieces.append(f"<{tag}>{html.escape(quote, quote=False)}</{tag}>")
        end = match.end()
    pieces.append(html.escape(text[end:], quote=False))
    return "".join(pieces)


class PersuasiveDebate:
    """Own one independent debate at a time; each run resets the audit records.

    Configuration and provider construction belong to the caller. Preference
    callbacks return first-token log probabilities (A/B or Y/N). No implicit
    transport retries, Sessions, ground-truth labels, or candidate execution.
    Configure Slick's process-global template root before calling run().
    """

    def __init__(
        self,
        problem: Problem,
        debaters: tuple[Debater, Debater],
        judge: Provider,
        *,
        preference: Logprobs | None = None,
        critique_preference: Logprobs | None = None,
        verify_quote: QuoteVerifier | None = None,
        protocol: Literal["debate", "interactive_debate", "consultancy"] = "debate",
        consultant: int = 0,
        min_words: int = 70,
        target_words: int = 100,
        max_words: int = 150,
        candidates_per_sample: int = 3,
    ):
        self.problem, self.debaters, self.judge = problem, debaters, judge
        self.preference = preference
        self.critique_preference = (
            preference if critique_preference is None else critique_preference
        )
        self.verify_quote = self._source_quote if verify_quote is None else verify_quote
        self.protocol, self.consultant = protocol, consultant
        self.word_factor = 2 if protocol == "consultancy" else 1
        self.min_words, self.target_words, self.max_words = min_words, target_words, max_words
        self.candidates_per_sample = candidates_per_sample
        self.turns: list[Turn] = []
        self.calls: list[Call] = []
        self.failures: list[str] = []
        self.round = 0

    @prompt(template="open.j2")
    async def open_argument(self, side: int, view: View, *, generated: str) -> str:
        return extract(generated, "argument")

    @prompt(template="challenge.j2")
    async def challenge(self, side: int, view: View, *, generated: str) -> str:
        return extract(generated, "argument")

    @prompt(template="rebut.j2")
    async def rebut(self, side: int, view: View, *, generated: str) -> str:
        return extract(generated, "argument")

    @prompt(template="respond.j2")
    async def respond(self, side: int, view: View, *, generated: str) -> str:
        return extract(generated, "argument")

    @prompt(template="revise.j2")
    async def revise(
        self,
        side: int,
        view: View,
        argument: str,
        critique: str,
        *,
        generated: str,
    ) -> str:
        return extract(generated, "argument")

    @prompt(template="critique.j2")
    async def critique(self, side: int, view: View, argument: str, *, generated: str) -> str:
        return extract(generated, "critique")

    @prompt(template="ask.j2")
    async def ask(self, view: View, *, generated: str) -> str:
        return extract(generated, "question")

    @prompt(template="judge.j2")
    async def judge_answer(self, view: View, *, generated: str) -> int:
        public = strip_thinking(generated).strip()
        choices = re.findall(r"^Answer:\s*([AB])\s*$", public, re.MULTILINE)
        if len(choices) != 1:
            raise CandidateError("Judge must return exactly one Answer: A or Answer: B")
        return "AB".index(choices[0])

    async def run(self, *, rounds: int = 3, votes: int = 1) -> Result:
        """Generate rounds, judge both orders, and return canonical answer votes.

        votes is the number of independent judgments per order. Exact ties
        return choice=None. approval is vote share, not calibrated confidence.
        Completed rounds and raw calls survive exceptions for caller inspection.
        """
        self.turns, self.calls, self.failures = [], [], []
        self.round = 0
        for self.round in range(1, rounds + 1):
            await self._debate_round()
            if self.protocol != "debate" and self.round < rounds:
                # Stable answer text, not A/B labels, keeps questions swappable.
                view = self._view(tuple(self.turns), (0, 1), self.problem.answers)
                question = await self._invoke(self.ask, self.judge, None, view)
                self.turns.append(Turn(self.round, None, html.escape(question, quote=False)))
        judgments = await self._judge_orders(votes)
        approval = judgments.count(0) / len(judgments)
        choice = None if approval == 0.5 else int(approval < 0.5)
        return Result(
            choice,
            None if choice is None else self.problem.answers[choice],
            (approval, 1 - approval),
            judgments,
            tuple(self.turns),
            len(self.calls),
        )

    async def _debate_round(self):
        snapshot = tuple(self.turns)
        sides = (self.consultant,) if self.protocol == "consultancy" else (0, 1)
        tasks = [asyncio.create_task(self._argument(side, snapshot)) for side in sides]
        # Finish siblings before raising; never publish half a simultaneous round.
        await asyncio.gather(*tasks, return_exceptions=True)
        arguments = [task.result() for task in tasks]
        self.turns.extend(Turn(self.round, side, text) for side, text in zip(sides, arguments))

    async def _argument(self, side: int, snapshot: tuple[Turn, ...]) -> str:
        expert = self.debaters[side]
        view = self._view(snapshot, (side, 1 - side), ("You", "Opponent"))
        if self.round == 1:
            operation = self.open_argument
        elif self.protocol == "consultancy":
            operation = self.respond
        else:
            operation = self.challenge if self.round == 2 else self.rebut
        candidates = await self._sample(operation, side, view)
        argument = await self._select(candidates, side, snapshot)
        if expert.critiques:
            critique = await self._best_critique(side, view, argument)
            try:
                revisions = await self._sample(self.revise, side, view, argument, critique)
            except CandidateError as exc:
                self.failures.append(f"round {self.round}, side {side}: retained original: {exc}")
            else:
                argument = await self._select(revisions, side, snapshot)
        return argument

    async def _sample(self, operation, side: int, view: View, *args) -> list[str]:
        expert = self.debaters[side]
        factor = self.word_factor
        valid, invalid = [], []
        for _ in range(expert.best_of * self.candidates_per_sample):
            try:
                argument = await self._invoke(operation, expert.provider, side, side, view, *args)
            except CandidateError as exc:
                self.failures.append(f"round {self.round}, side {side}: {exc}")
                continue
            if operation == self.revise and "critique" in argument.lower():
                self.failures.append(
                    f"round {self.round}, side {side}: refinement mentions critique"
                )
                continue
            words = len(argument.split())
            argument = _QUOTE_TAG.sub(r"<\1quote>", argument)
            # Upstream prefers length-compliant candidates containing a quote.
            accepted = (
                self.min_words * factor <= words <= self.max_words * factor
                and "<quote>" in argument
            )
            processed = await verify_quotes(
                truncate(argument, self.max_words * factor), self.verify_quote
            )
            if accepted:
                valid.append(processed)
            else:
                invalid.append(processed)
                self.failures.append(f"round {self.round}, side {side}: word/quote filter fallback")
        candidates = (valid + invalid)[: expert.best_of]
        if len(candidates) < expert.best_of:
            raise CandidateError("Rejection pool exhausted before enough extractable arguments")
        return candidates

    async def _select(self, candidates: list[str], side: int, snapshot: tuple[Turn, ...]) -> str:
        if len(candidates) == 1:
            return candidates[0]
        scores = []
        for candidate in candidates:
            current = [Turn(self.round, side, candidate)]
            if self.protocol != "consultancy":
                current.append(Turn(self.round, 1 - side, DUMMY_ARGUMENT))
            view = self._view(snapshot + tuple(current))
            scores.append(
                await self._score(
                    "prefer_argument.j2",
                    self.preference,
                    "AB"[side],
                    side,
                    view=view,
                )
            )
        return candidates[max(range(len(candidates)), key=scores.__getitem__)]

    async def _best_critique(self, side: int, view: View, argument: str) -> str:
        expert = self.debaters[side]
        source = expert.provider if expert.critic is None else expert.critic
        critiques, scores = [], []
        limit = self.max_words * (2 if self.protocol == "consultancy" else 1)
        for _ in range(expert.critiques):
            text = await self._invoke(self.critique, source, side, side, view, argument)
            text = await verify_quotes(truncate(text, limit), self.verify_quote)
            critiques.append(text)
            if expert.critiques > 1:
                scores.append(
                    await self._score(
                        "prefer_critique.j2",
                        self.critique_preference,
                        "Y",
                        side,
                        view=view,
                        argument=argument,
                        critique=text,
                    )
                )
        return (
            critiques[max(range(len(critiques)), key=scores.__getitem__)]
            if scores
            else critiques[0]
        )

    async def _judge_orders(self, votes: int) -> tuple[int, ...]:
        judgments = []
        for order in ((0, 1), (1, 0)):
            view = self._view(tuple(self.turns), order)
            for _ in range(votes):
                position = await self._invoke(self.judge_answer, self.judge, None, view)
                judgments.append(order[position])
        return tuple(judgments)

    def _view(
        self,
        turns: tuple[Turn, ...],
        order: tuple[int, int] = (0, 1),
        labels: tuple[str, str] = ("Debater A", "Debater B"),
    ) -> View:
        lines = []
        for round_number in dict.fromkeys(turn.round for turn in turns):
            lines.append(f"Round {round_number}:")
            for side, label in zip(order, labels):
                for turn in turns:
                    if turn.round == round_number and turn.side == side:
                        lines.append(f"{label}: {turn.text}")
            for turn in turns:
                if turn.round == round_number and turn.side is None:
                    lines.append(f"Judge: {turn.text}")
        return View(
            self.problem.question,
            tuple(self.problem.answers[side] for side in order),
            self.problem.criteria,
            "\n".join(lines),
            self.protocol,
        )

    async def _source_quote(self, quote: str) -> bool:
        return normalize(quote) in normalize(self.problem.evidence)

    async def _invoke(self, operation, source, side, *args):
        call = Call(operation.__name__, side, self.round, source)
        self.calls.append(call)
        try:
            return await operation(*args, provider=call)
        except Exception as exc:
            call.error = f"{type(exc).__name__}: {exc}"
            raise

    async def _score(self, template, scorer, target, side, **variables) -> float:
        call = Call(template, side, self.round)
        self.calls.append(call)
        try:
            call.context = Prompt(template)(**variables)
            probabilities = await scorer(call.context)
            call.response = dict(probabilities)
            score = probabilities.get(target, -100.0)
            if not math.isfinite(score) or score > 0:
                raise ValueError(
                    "Preference score must be a finite, nonpositive token log probability"
                )
            return score
        except Exception as exc:
            call.error = f"{type(exc).__name__}: {exc}"
            raise
