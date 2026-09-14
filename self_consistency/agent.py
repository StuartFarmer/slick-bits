"""Sample independent reasoning paths and marginalize them by counting final answers."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Sample:
    response: str
    answer: str | None
    error: str | None = None


@dataclass(frozen=True)
class Result:
    answer: str
    counts: dict[str, int]
    consistency: float
    samples: tuple[Sample, ...]


def extract_answer(response: str) -> str:
    """Read the last 'The answer is ' suffix before the next Q:, removing one period."""
    response = response.partition("Q:")[0]
    _, marker, answer = response.rpartition("The answer is ")
    answer = answer.strip().removesuffix(".").strip()
    if not marker or not answer:
        raise ValueError("missing or empty final answer")
    return answer


class SelfConsistency:
    """Own fixed-budget sampling and unweighted answer voting from Section 2.

    Use a stateless provider configured for stochastic decoding. The synchronous
    extractor defines answer identity; ValueError rejects one sampled response.
    Model settings, transport retries, and any external evaluation belong to callers.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        examples: str = "",
        extract_answer: Callable[[str], str] = extract_answer,
    ):
        self.task = task
        self.provider = provider
        self.examples = examples
        self.extract_answer = extract_answer
        self.samples: list[Sample] = []
        self.calls: list[dict] = []

    @prompt(template="generate.j2")
    async def generate(self, input: str, *, generated: str) -> str:
        """Generate one textual path; extraction happens after raw response recording."""
        return generated

    async def run(self, input: str, *, samples: int = 40) -> Result:
        """Sample, then vote; ties select the first encountered valid answer.

        Invalid answers consume a sample without replacement. All-invalid runs
        raise ValueError. Provider/tool errors and non-ValueError extractor errors
        abort, retaining partial samples and call records. Runs reset state;
        use one run at a time per instance. There are no algorithm-owned retries.
        """
        self.samples = []
        self.calls = []
        await self.sample_paths(input, samples)
        return self.aggregate()

    async def sample_paths(self, input: str, count: int) -> None:
        """Sample the same prompt without sharing outputs or conversation history."""
        # ponytail: sequential calls; add bounded concurrency if latency matters.
        for _ in range(count):
            record = {"operation": "generate"}
            self.calls.append(record)
            try:
                response = await self.generate(input, provider=self)
                try:
                    answer = self.extract_answer(response)
                    if not answer.strip():
                        raise ValueError("empty extracted answer")
                except ValueError as exc:
                    self.samples.append(Sample(response, None, str(exc)))
                    record["rejection"] = str(exc)
                    continue
                self.samples.append(Sample(response, answer))
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise

    def aggregate(self) -> Result:
        """Count answer occurrences, including duplicate paths as independent draws."""
        counts = Counter(sample.answer for sample in self.samples if sample.answer is not None)
        if not counts:
            raise ValueError("no valid answers among sampled responses")
        answer, votes = counts.most_common(1)[0]
        return Result(answer, dict(counts), votes / len(self.samples), tuple(self.samples))

    async def acall(self, context, *, tools=None, tool_results=None):
        """Keep raw responses before parsing and reject unexpected tool requests."""
        record = self.calls[-1]
        record["prompt"] = context
        response, requests = await self.provider.acall(
            context, tools=tools, tool_results=tool_results
        )
        record["response"] = response
        if requests:
            raise ValueError("self-consistency requires text, not tool requests")
        return response, requests
