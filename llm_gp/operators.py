"""Paper Appendix 3 prompts, executed and parsed by Slick."""

import asyncio
import json
import math
import time

from pydantic import BaseModel, ConfigDict, Field
from slick import prompt
from slick.prompts import PromptError
from slick.providers import OpenAIAPI, Provider, ProviderError

from llm_gp.core import BudgetExceeded, fitness, parse_expression


class Expression(BaseModel):
    expression: str = Field(min_length=1, max_length=4096)


class Mutation(BaseModel):
    new_expression: str = Field(min_length=1, max_length=4096)


class Children(BaseModel):
    expressions: list[str] = Field(min_length=2, max_length=2)


class Choice(BaseModel):
    model_config = ConfigDict(strict=True)
    individuals: list[int]


@prompt(template="initialize.j2", output_type=Expression)
async def initialize_prompt(*, generated: Expression) -> Expression:
    """Generate an expression; Operators validates its grammar and handles fallback."""
    return generated


@prompt(template="mutate.j2", output_type=Mutation)
async def mutation_prompt(expression: str, samples: list[str], *, generated: Mutation) -> Mutation:
    """Generate a mutation; Operators validates its grammar and handles fallback."""
    return generated


@prompt(template="crossover.j2", output_type=Children)
async def crossover_prompt(
    expressions: list[str], samples: list[str], *, generated: Children
) -> Children:
    """Generate two children; Operators validates their grammar and handles fallback."""
    return generated


@prompt(template="choose.j2", output_type=Choice)
async def choice_prompt(
    operation: str, individuals: list[dict], count: int, samples: list[dict], *, generated: Choice
) -> Choice:
    """Parse integer IDs; Operators checks membership, count, and uniqueness."""
    return generated


class MeasuredOpenAI(OpenAIAPI):
    """Preserve SDK usage before Slick reduces the response to text and tool calls."""

    def __init__(self, model, *, temperature=0.8, **kwargs):
        super().__init__(model, **kwargs)
        self.temperature = temperature
        self.last_usage = None
        self.resolved_model = None

    def _encode(self, context, prepared, results):
        request = super()._encode(context, prepared, results)
        if self.temperature is not None:
            request["temperature"] = self.temperature
        return request

    async def _asend(self, request):
        response = await super()._asend(request)
        self.last_usage = response.usage.model_dump() if response.usage else None
        self.resolved_model = response.model
        if response.status != "completed":
            raise ProviderError(f"Incomplete response: {response.status}")
        return response


class Operators(Provider):
    """One sequential run; log each transport attempt and the checked fallback."""

    def __init__(
        self,
        backend,
        rng,
        *,
        n_shots=2,
        retries=2,
        retry_delay=1,
        max_calls=10000,
        seconds=60000,
        timeout=60,
        on_call=None,
    ):
        if any(type(n) is not int or n < 0 for n in (n_shots, retries)):
            raise ValueError("n_shots and retries must be nonnegative integers")
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if any(not math.isfinite(t) or t <= 0 for t in (seconds, timeout)):
            raise ValueError("seconds and timeout must be finite and positive")
        if not math.isfinite(retry_delay) or retry_delay < 0:
            raise ValueError("retry_delay must be finite and nonnegative")
        self.backend, self.rng = backend, rng
        self.n_shots, self.retries, self.retry_delay = n_shots, retries, retry_delay
        self.max_calls, self.timeout = max_calls, timeout
        self.deadline = time.perf_counter() + seconds
        # ponytail: retain tutorial-size logs; stream aggregates for larger studies.
        self.calls, self.operation, self.on_call = [], "", on_call

    async def acall(self, context, *, tools=None, tool_results=None):
        for attempt in range(self.retries + 1):
            remaining = self.deadline - time.perf_counter()
            if len(self.calls) >= self.max_calls or remaining <= 0:
                raise BudgetExceeded
            started = time.perf_counter()
            record = {
                "operation": self.operation,
                "attempt": attempt + 1,
                "prompt": context,
                "response": None,
                "error": None,
                "fallback": False,
                "usage": None,
                "resolved_model": None,
            }
            if hasattr(self.backend, "last_usage"):
                self.backend.last_usage = None
            try:
                response, requests = await asyncio.wait_for(
                    self.backend.acall(context), min(self.timeout, remaining)
                )
                record["response"] = response
                if requests:
                    raise PromptError("Unexpected tool requests")
                return response, []
            except (ProviderError, asyncio.TimeoutError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if attempt == self.retries:
                    raise
            finally:
                record["seconds"] = time.perf_counter() - started
                record["usage"] = getattr(self.backend, "last_usage", None)
                record["resolved_model"] = getattr(self.backend, "resolved_model", None)
                self.calls.append(record)
            delay = self.retry_delay * 2**attempt
            if time.perf_counter() + delay >= self.deadline:
                raise BudgetExceeded
            await asyncio.sleep(delay)

    async def request(self, operation, function, arguments, validate, fallback):
        self.operation = operation
        start = len(self.calls)
        try:
            response = await function(**arguments, provider=self)
            return validate(response)
        except (
            ValueError,
            SyntaxError,
            RecursionError,
            ProviderError,
            asyncio.TimeoutError,
            PromptError,
        ) as exc:
            self.calls[-1]["error"] = f"{type(exc).__name__}: {exc}"
            self.calls[-1]["fallback"] = True
            return fallback()
        finally:
            if self.on_call:
                for record in self.calls[start:]:
                    self.on_call(record)

    def samples(self, expressions):
        return self.rng.sample(expressions, min(self.n_shots, len(expressions)))

    async def initialize(self):
        def validate(response):
            parse_expression(response.expression)
            return response.expression

        return await self.request("initialization", initialize_prompt, {}, validate, lambda: "0")

    async def mutate(self, expression, samples):
        def validate(response):
            parse_expression(response.new_expression)
            return response.new_expression

        return await self.request(
            "mutation",
            mutation_prompt,
            {"expression": expression, "samples": self.samples(samples)},
            validate,
            lambda: expression,
        )

    async def crossover(self, expressions, samples):
        def validate(response):
            for expression in response.expressions:
                parse_expression(expression)
            return response.expressions

        return await self.request(
            "crossover",
            crossover_prompt,
            {"expressions": expressions, "samples": self.samples(samples)},
            validate,
            lambda: expressions[:],
        )

    async def choose(self, operation, population, count):
        if operation not in ("selection", "replacement", "best") or not population or count < 1:
            raise ValueError("invalid selection request")
        unique = operation != "selection"
        if unique and count > len(population):
            raise ValueError("cannot select more unique IDs than the population")
        individuals = [{"id": i, **p} for i, p in enumerate(population)]

        def validate(response):
            ids = response.individuals
            if len(ids) != count or any(i < 0 or i >= len(population) for i in ids):
                raise ValueError("wrong count or unknown individual ID")
            if unique and len(set(ids)) != count:
                raise ValueError("duplicate individual IDs")
            return [population[i] for i in ids]

        def fallback():
            return (
                sorted(population, key=fitness)[:count]
                if unique
                else self.rng.choices(population, k=count)
            )

        return await self.request(
            operation,
            choice_prompt,
            {
                "operation": operation,
                "individuals": individuals,
                "count": count,
                "samples": sorted(individuals, key=fitness)[: self.n_shots],
            },
            validate,
            fallback,
        )


class DemoProvider(Provider):
    """Canned outputs, including the target: tests plumbing, never LLM discovery."""

    provider, model = "demo", "canned-not-an-llm"

    def __init__(self):
        self.index = 0

    async def acall(self, context, *, tools=None, tool_results=None):
        if "Operation:" in context:
            # The demo reads only the explicitly delimited choice instruction.
            count = int(context.split("Select ")[1].split()[0])
            individuals = json.loads(
                context.split(" high quality elements from ")[1].split(".\n")[0]
            )
            ordered = sorted(individuals, key=fitness)
            response = {"individuals": [ordered[i % len(ordered)]["id"] for i in range(count)]}
        elif "Recombine the mathematical" in context:
            response = {"expressions": ["x0*x0 + x1*x1", "x0 + x1"]}
        elif "Rephrase the mathematical" in context:
            response = {"new_expression": "x0*x0 + x1*x1"}
        else:
            response = {"expression": ("x0", "x1", "x0*x1", "1", "0")[self.index % 5]}
            self.index += 1
        return json.dumps(response), []
