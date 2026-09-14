"""Compile and adapt SEED's hybrid pipelines for caller-defined input/output tasks."""

import ast
import asyncio
import json
import math
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, JsonValue, StringConstraints, ValidationError
from slick import Session, prompt
from slick.providers import Provider

from .batching import form_batches
from .planning import (
    Evaluator,
    Example,
    Module,
    Optimization,
    Prediction,
    Predictor,
    SeedOptimizer,
    checked_predictions,
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Answer(BaseModel, extra="forbid"):
    id: int = Field(strict=True, ge=0)
    value: JsonValue


class Batch(BaseModel, extra="forbid"):
    answers: list[Answer]


class Advice(BaseModel, extra="forbid"):
    strategies: list[Text] = Field(min_length=1)


class Code(BaseModel, extra="forbid"):
    source: str = Field(min_length=1)


class GeneratedExample(BaseModel, extra="forbid"):
    input: str
    output: JsonValue


class Cases(BaseModel, extra="forbid"):
    examples: list[GeneratedExample] = Field(min_length=1)


class CandidateRejected(Exception):
    """An isolated executor may raise this for invalid or infeasible generated code."""


@dataclass(frozen=True)
class Config:
    gap: float = 0.05
    optimizer: Literal["specialized", "generic", "exhaustive"] = "specialized"
    beam_size: int | None = 20
    code_branches: int = 4
    preserved_branches: int = 8
    evolution_iterations: int = 2
    advice_count: int = 2
    max_errors: int = 3
    max_codegen_calls: int = 64
    max_fallback: float = 0.95
    code_timeout: float = 10.0
    ensemble: Literal["sequential", "vote", "weighted"] = "vote"
    batch_size: int = 16
    batching: Literal["RND", "DIV", "PRX", "FAR", "CLS"] = "RND"
    few_shot: int = 5
    sample_modes: tuple[Literal["random", "balanced", "nearest"], ...] = ("random",)
    distance_thresholds: tuple[float, ...] = (0.0, 0.2, 0.4, 0.8)
    confidence_thresholds: tuple[float, ...] = (0.5, 0.8, 0.95)
    min_train: int = 32
    reoptimize_every: int = 128
    local_radius: int = 1
    llm_cost: float = 1.0
    code_cost: float = 0.01
    model_cost: float = 0.05
    cache_cost: float = 0.001
    random_seed: int = 0


@dataclass(frozen=True)
class VerifiedCode:
    source: str
    accuracy: float
    correct: frozenset[int]
    predictions: tuple[Prediction | None, ...]

    @property
    def failures(self):
        return {i for i, p in enumerate(self.predictions) if p is not None} - self.correct


def classification_confidence(probabilities: Sequence[float]) -> float:
    """Section 4.2: normalize maximum class probability above random guessing."""
    k = len(probabilities)
    return 1.0 if k == 1 else (k * max(probabilities) - 1) / (k - 1)


def generation_confidence(log_probabilities: Sequence[float]) -> float:
    """Inverse perplexity over generated tokens, excluding padding."""
    return math.exp(sum(log_probabilities) / len(log_probabilities))


def json_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def exact_match(expected, predicted):
    return json_key(expected) == json_key(predicted)


def unit_vectors(vectors, count):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or len(vectors) != count or not np.isfinite(vectors).all():
        raise ValueError("embed must return a finite matrix with one vector per input")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must have nonzero norm")
    return vectors / norms


class SEED:
    """Own code evolution, module compilation, label cache and dynamic plan state.

    Inputs are caller-serialized text; outputs are JSON values. evaluate owns the
    task metric. execute_code(source, inputs) owns execution isolation; it returns
    Predictions or None for abstentions. train_model(examples) returns a predictor
    with calibrated confidence. No generated code executes inside this class.

    Use one instance per task and validation split, sequentially. Callers own
    transport retries, model/execution resources, persistence and template-root setup.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Evaluator,
        *,
        embed: Callable[[Sequence[str]], Awaitable[np.ndarray]] | None = None,
        execute_code: Callable[[str, Sequence[str]], Awaitable[Sequence[Prediction | None]]]
        | None = None,
        train_model: Callable[[Sequence[Example]], Awaitable[Predictor]] | None = None,
        tools: Sequence = (),
        validate_output: Callable[[JsonValue], JsonValue] = lambda value: value,
        matches: Callable[[JsonValue, JsonValue], bool] = exact_match,
        config: Config = Config(),
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.embed, self.execute_code, self.train_model = embed, execute_code, train_model
        self.tools, self.validate_output, self.matches = tuple(tools), validate_output, matches
        self.config = config
        self.cache: dict[str, Example] = {}
        self.heldout: set[str] = set()
        self.attempts, self.code_evaluations, self.history = [], [], []
        self.runtime_failures = []
        self.routing = defaultdict(int)
        self.codegen_calls = self.llm_records = self.revision = 0
        self._last_refresh = 0
        self.codes: list[VerifiedCode] = []

    @prompt(template="query.j2", output_type=Batch, max_turns=12)
    async def query(self, inputs, demonstrations, *, generated: Batch) -> list[Prediction]:
        ids = [answer.id for answer in generated.answers]
        if sorted(ids) != list(range(len(inputs))):
            raise ValueError("answer IDs must cover each input exactly once")
        predictions = [
            Prediction(self.validate_output(answer.value))
            for answer in sorted(generated.answers, key=lambda answer: answer.id)
        ]
        for prediction in predictions:
            json_key(prediction.value)
        return predictions

    @prompt(template="advice.j2", output_type=Advice)
    async def advise(self, examples, count, *, generated: Advice) -> Advice:
        return generated

    @prompt(template="generate.j2", output_type=Code)
    async def generate(self, advice, examples, *, generated: Code) -> str:
        return self._check_code(generated.source)

    @prompt(template="repair_advice.j2", output_type=Advice)
    async def advise_repair(self, code, errors, count, *, generated: Advice) -> Advice:
        return generated

    @prompt(template="repair.j2", output_type=Code)
    async def repair(self, code, errors, advice, *, generated: Code) -> str:
        return self._check_code(generated.source)

    @prompt(template="examples.j2", output_type=Cases)
    async def generate_examples(self, count, *, generated: Cases) -> list[Example]:
        examples = [Example(e.input, self.validate_output(e.output)) for e in generated.examples]
        try:
            for example in examples:
                json_key(example.output)
        except ValueError as error:
            raise CandidateRejected(
                "generated examples must contain finite JSON outputs"
            ) from error
        return examples

    def _check_code(self, source):
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            raise CandidateRejected(str(error)) from error
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        # This validates generated syntax/interface only, never execution safety.
        if not any(node.name == "solve" and len(node.args.args) == 1 for node in functions):
            raise CandidateRejected("code must define a synchronous solve(input) function")
        return source

    async def _call(self, operation, *args, tool_query=False):
        record = {"operation": operation.__name__}
        self.attempts.append(record)
        session = Session(provider=self.provider, tools=self.tools if tool_query else ())
        try:
            result = await operation(*args, session=session)
            record["status"] = "accepted"
            return result
        except Exception as error:
            record["error"] = f"{error.__class__.__name__}: {error}"
            raise
        finally:
            record["raw"] = session.to_dict()

    async def _code_call(self, operation, *args):
        if self.codegen_calls >= self.config.max_codegen_calls:
            return None
        self.codegen_calls += 1
        try:
            return await self._call(operation, *args)
        except (ValidationError, CandidateRejected):
            return None

    async def _verify(self, source, examples):
        record = {"source": source}
        self.code_evaluations.append(record)
        try:
            outputs = await asyncio.wait_for(
                self.execute_code(source, [e.input for e in examples]),
                timeout=self.config.code_timeout,
            )
            outputs = checked_predictions(outputs, len(examples))
        except (TimeoutError, CandidateRejected) as error:
            record["error"] = f"{error.__class__.__name__}: {error}"
            return None
        answered = [i for i, p in enumerate(outputs) if p is not None]
        score = (
            float(
                await self.evaluate([examples[i] for i in answered], [outputs[i] for i in answered])
            )
            if answered
            else 0.0
        )
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("measured code effectiveness must be finite and in [0, 1]")
        correct = frozenset(
            i
            for i, (e, p) in enumerate(zip(examples, outputs))
            if p is not None and self.matches(e.output, p.value)
        )
        record["score"] = score
        return VerifiedCode(source, score, correct, outputs)

    def _filter_codes(self, candidates):
        ranked = sorted(candidates, key=lambda c: (-c.accuracy, -len(c.correct), len(c.failures)))
        unique = {code.source: code for code in ranked}
        viable = [
            c
            for c in unique.values()
            if sum(p is None for p in c.predictions) / len(c.predictions) < self.config.max_fallback
        ]
        kept = []
        for candidate in viable:
            # Adapted from upstream CodeGenAgent.codeg_profile_dominated: a wider
            # correct set alone must not eliminate a more cautious, precise snippet.
            def dominates(other):
                no_worse = (
                    candidate.correct <= other.correct
                    and other.failures <= candidate.failures
                    and other.accuracy >= candidate.accuracy
                )
                better = (
                    candidate.correct < other.correct
                    or other.failures < candidate.failures
                    or other.accuracy > candidate.accuracy
                )
                return no_worse and (better or other in kept)

            if any(dominates(other) for other in viable if other is not candidate):
                continue
            kept.append(candidate)
        return kept[: self.config.preserved_branches]

    async def _initialize_code(self, examples):
        advice = await self._code_call(self.advise, examples, self.config.code_branches)
        if advice is None:
            return []
        candidates = []
        for strategy in advice.strategies[: self.config.code_branches]:
            source = await self._code_call(self.generate, strategy, examples)
            if source is not None:
                candidate = await self._verify(source, examples)
                if candidate is not None:
                    candidates.append(candidate)
        return candidates

    async def _evolve_code(self, population, examples):
        candidates = list(population)
        for parent in population:
            errors = [
                {
                    "input": e.input,
                    "expected": e.output,
                    "prediction": p.value if p is not None else None,
                    "abstained": p is None,
                }
                for i, (e, p) in enumerate(zip(examples, parent.predictions))
                if i not in parent.correct
            ]
            groups = [[error] for error in errors[: self.config.max_errors]]
            if len(errors) > 1:
                groups.append(errors)
            for group in groups:
                advice = await self._code_call(
                    self.advise_repair, parent.source, group, self.config.advice_count
                )
                if advice is None:
                    continue
                for strategy in advice.strategies[: self.config.advice_count]:
                    source = await self._code_call(self.repair, parent.source, group, strategy)
                    if source is not None and source not in {c.source for c in candidates}:
                        candidate = await self._verify(source, examples)
                        if candidate is not None:
                            candidates.append(candidate)
        return self._filter_codes(candidates)

    async def _compile_code(self, examples):
        if self.execute_code is None or self.config.code_branches == 0:
            return
        if not examples:
            examples = await self._code_call(self.generate_examples, 5)
            if examples is None:
                return
        examples = tuple(e for e in examples if e.input not in self.heldout)
        if not examples:
            return
        population = await self._initialize_code(examples)
        for _ in range(self.config.evolution_iterations):
            evolved = await self._evolve_code(population, examples)
            if [c.source for c in evolved] == [c.source for c in population]:
                break
            population = evolved
        self.codes = self._filter_codes(population)[: self.config.code_branches]

    async def _ensemble(self, codes, inputs):
        outputs = [None] * len(inputs)
        votes = [defaultdict(float) for _ in inputs]
        values = [{} for _ in inputs]
        for code in codes:
            indices = list(range(len(inputs)))
            if self.config.ensemble == "sequential":
                indices = [i for i in indices if outputs[i] is None]
            if not indices:
                break
            try:
                predictions = await asyncio.wait_for(
                    self.execute_code(code.source, [inputs[i] for i in indices]),
                    timeout=self.config.code_timeout,
                )
                predictions = checked_predictions(predictions, len(indices))
            except (TimeoutError, CandidateRejected) as error:
                self.runtime_failures.append({"source": code.source, "error": str(error)})
                continue
            for i, prediction in zip(indices, predictions):
                if prediction is None:
                    continue
                if self.config.ensemble == "sequential":
                    outputs[i] = prediction
                else:
                    key = json_key(prediction.value)
                    votes[i][key] += code.accuracy if self.config.ensemble == "weighted" else 1
                    values[i][key] = prediction
        if self.config.ensemble != "sequential":
            for i, vote in enumerate(votes):
                if vote:
                    best = max(vote.values())
                    winners = [k for k in vote if vote[k] == best]
                    if len(winners) == 1:
                        outputs[i] = values[i][winners[0]]
        return outputs

    async def _demonstrations(self, inputs, examples, mode):
        if not examples or self.config.few_shot == 0:
            return []
        count = min(len(examples), self.config.few_shot)
        if mode == "random":
            # Fixed sampling makes a frozen LLM configuration stable across reoptimization.
            selected = random.Random(self.config.random_seed).sample(list(examples), count)
        elif mode == "balanced":
            groups = defaultdict(list)
            for example in examples:
                groups[json_key(example.output)].append(example)
            rng = random.Random(self.config.random_seed)
            for group in groups.values():
                rng.shuffle(group)
            selected = []
            while len(selected) < count:
                for group in groups.values():
                    if group and len(selected) < count:
                        selected.append(group.pop())
        else:
            vectors = unit_vectors(
                await self.embed([e.input for e in examples] + list(inputs)),
                len(examples) + len(inputs),
            )
            similarity = (vectors[: len(examples)] @ vectors[len(examples) :].T).mean(axis=1)
            order = np.argsort(-similarity, kind="stable")
            selected = [examples[i] for i in order[:count]]
        return [{"input": e.input, "output": e.output} for e in selected]

    async def _batches(self, inputs):
        vectors = None
        if inputs and self.config.batching != "RND":
            vectors = unit_vectors(await self.embed(inputs), len(inputs))
        return form_batches(
            len(inputs),
            self.config.batch_size,
            self.config.batching,
            vectors,
            seed=self.config.random_seed,
        )

    async def _query_inputs(self, inputs, examples, mode, *, cache_labels=False):
        outputs = [None] * len(inputs)
        for indices in await self._batches(inputs):
            batch = [inputs[i] for i in indices]
            demos = await self._demonstrations(batch, examples, mode)
            self.llm_records += len(batch)
            predictions = await self._call(self.query, batch, demos, tool_query=True)
            for i, prediction in zip(indices, predictions):
                outputs[i] = prediction
                if cache_labels and inputs[i] not in self.heldout:
                    self.cache[inputs[i]] = Example(inputs[i], prediction.value)
        return outputs

    async def label(self, inputs: Sequence[str]) -> list[Prediction]:
        """LLM-annotate records, atomically caching each successful batch, except held-out inputs."""
        return await self._query_inputs(
            inputs, tuple(self.cache.values()), "random", cache_labels=True
        )

    def _static_modules(self, examples):
        modules = []
        for mode in self.config.sample_modes:

            async def llm(inputs, mode=mode):
                return await self._query_inputs(inputs, examples, mode)

            modules.append(Module(f"llm:{mode}", "LLM", self.config.llm_cost, llm))
        for count in sorted({1, len(self.codes)} - {0}):
            if not self.codes:
                break

            async def code(inputs, codes=tuple(self.codes[:count])):
                return await self._ensemble(codes, inputs)

            modules.append(Module(f"code:{count}", "CodeGen", self.config.code_cost * count, code))
        return modules

    def _thresholds(self, grid, family):
        if not self.history:
            return grid
        previous = next(
            (m.threshold for m in self.history[-1].best.modules if m.family == family), None
        )
        if previous is None:
            return grid
        center = min(range(len(grid)), key=lambda i: abs(grid[i] - previous))
        radius = self.config.local_radius
        return grid[max(0, center - radius) : center + radius + 1]

    async def _adaptive_modules(self):
        examples = tuple(self.cache.values())
        modules = []
        if not examples:
            return modules
        if self.train_model is not None and len(examples) >= self.config.min_train:
            predictor = await self.train_model(examples)
            # One raw prediction per batch, shared by all confidence thresholds.
            measured = {}

            async def model(inputs):
                key = tuple(inputs)
                if key not in measured:
                    measured[key] = checked_predictions(await predictor(inputs), len(inputs))
                return measured[key]

            for threshold in self._thresholds(self.config.confidence_thresholds, "ModelGen"):

                async def gated(inputs, threshold=threshold):
                    return [
                        p if p is not None and p.confidence >= threshold else None
                        for p in await model(inputs)
                    ]

                modules.append(
                    Module(
                        f"model:{self.revision}:{threshold}",
                        "ModelGen",
                        self.config.model_cost,
                        gated,
                        threshold,
                    )
                )
        if self.embed is not None:
            vectors = unit_vectors(await self.embed([e.input for e in examples]), len(examples))
            exact = {e.input: e.output for e in examples}
            neighbors = {}

            async def nearest(inputs):
                key = tuple(inputs)
                if key not in neighbors:
                    queries = unit_vectors(await self.embed(inputs), len(inputs))
                    # ponytail: linear scan; use a vector index when cache size warrants it.
                    distances = np.clip(1 - queries @ vectors.T, 0, 2)
                    indices = distances.argmin(axis=1)
                    neighbors[key] = [
                        (float(distances[i, j]), examples[j].output) for i, j in enumerate(indices)
                    ]
                return neighbors[key]

            for threshold in self._thresholds(self.config.distance_thresholds, "CacheReuse"):

                async def cached(inputs, threshold=threshold):
                    found = await nearest(inputs)
                    return [
                        Prediction(exact[x])
                        if x in exact
                        else Prediction(value)
                        if distance <= threshold
                        else None
                        for x, (distance, value) in zip(inputs, found)
                    ]

                modules.append(
                    Module(
                        f"cache:{self.revision}:{threshold}",
                        "CacheReuse",
                        self.config.cache_cost,
                        cached,
                        threshold,
                    )
                )
        return modules

    async def reoptimize(self) -> Optimization:
        """Refit adaptive modules on cached labels; reuse static validation measurements."""
        self.revision += 1
        modules = self.static_modules + await self._adaptive_modules()
        result = await self.optimizer.run(
            modules,
            gap=self.config.gap,
            mode=self.config.optimizer,
            beam_size=self.config.beam_size,
        )
        self.plan = result.best
        self.history.append(result)
        self._last_refresh = len(self.cache)
        return result

    async def run(
        self,
        validation: Sequence[Example],
        *,
        examples: Sequence[Example] = (),
        unlabeled: Sequence[str] = (),
    ) -> Optimization:
        """Compile from development examples and score plans on a fixed held-out split."""
        self.heldout = {e.input for e in validation}
        self.cache = {k: e for k, e in self.cache.items() if k not in self.heldout}
        examples = tuple(e for e in examples if e.input not in self.heldout)
        self.cache.update({e.input: e for e in examples})
        await self._compile_code(examples)
        self.static_modules = self._static_modules(examples)
        self.optimizer = SeedOptimizer(validation, self.evaluate)
        await self.label([x for x in unlabeled if x not in self.heldout])
        return await self.reoptimize()

    async def _execute_plan(self, inputs):
        outputs = [None] * len(inputs)
        for module in self.plan.modules:
            pending = [i for i, p in enumerate(outputs) if p is None]
            if not pending:
                break
            self.routing[module.family] += len(pending)
            predictions = checked_predictions(
                await module.predict([inputs[i] for i in pending]), len(pending)
            )
            for i, prediction in zip(pending, predictions):
                outputs[i] = prediction
                if (
                    module.family == "LLM"
                    and prediction is not None
                    and inputs[i] not in self.heldout
                ):
                    self.cache[inputs[i]] = Example(inputs[i], prediction.value)
        return outputs

    async def predict(self, inputs: Sequence[str]) -> list[Prediction]:
        """Route batches and reoptimize after enough new LLM labels accumulate."""
        outputs = [None] * len(inputs)
        for indices in await self._batches(inputs):
            if len(self.cache) - self._last_refresh >= self.config.reoptimize_every:
                await self.reoptimize()
            predictions = await self._execute_plan([inputs[i] for i in indices])
            for i, prediction in zip(indices, predictions):
                outputs[i] = prediction
        return outputs
