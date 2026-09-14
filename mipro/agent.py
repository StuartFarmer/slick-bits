"""Jointly optimize module instructions and bootstrapped demonstrations with categorical TPE."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import logsumexp
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class ModulePrompt:
    name: str
    instruction: str
    demonstrations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Trace:
    """End-to-end teacher score and one executed demonstration per program module."""

    score: float
    demonstrations: tuple[str, ...]


class MIPRO:
    """Own bootstrap acceptance, grounded proposals and joint categorical acquisition.

    evaluate(program, examples) returns a finite higher-is-better end-to-end score.
    execute(program, example) runs a teacher and returns its score and module
    traces; it neither selects demonstrations nor implements prompt search.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[tuple[ModulePrompt, ...], Sequence[str]], Awaitable[float]],
        execute: Callable[[tuple[ModulePrompt, ...], str], Awaitable[Trace]],
    ):
        self.task, self.provider, self.evaluate, self.execute = task, provider, evaluate, execute

    @prompt(template="summarize.j2")
    async def summarize(self, examples: Sequence[str], *, generated: str) -> str:
        """Describe the dataset to ground module-specific instruction proposals."""
        if not generated.strip():
            raise ValueError(f"empty task summary; response={generated!r}")
        return generated.strip()

    @prompt(template="propose.j2")
    async def propose(
        self,
        module: ModulePrompt,
        program: Sequence[ModulePrompt],
        summary: str,
        demonstrations: Sequence[str],
        *,
        generated: str,
    ) -> str:
        """Propose an instruction using program structure, data and executed examples."""
        if not generated.strip():
            raise ValueError(f"empty instruction; response={generated!r}")
        return generated.strip()

    async def _bootstrap(self, modules, training, candidate_sets, max_demos, attempts, threshold):
        accepted = [[] for _ in modules]
        examples = list(training)
        self.rng.shuffle(examples)
        for example in examples[:attempts]:
            self.bootstrap_calls += 1
            trace = await self.execute(modules, example)
            if not math.isfinite(trace.score):
                raise ValueError("teacher score must be finite")
            if trace.score >= threshold:
                for i in range(len(modules)):
                    accepted[i].append(trace.demonstrations[i])
        self.demo_candidates = []
        for module, pool in zip(modules, accepted):
            options = [module.demonstrations]
            for _ in range(candidate_sets - 1):
                size = self.rng.randint(1, min(max_demos, len(pool))) if pool and max_demos else 0
                options.append(tuple(self.rng.sample(pool, size)))
            self.demo_candidates.append(options)

    async def _propose_instructions(self, modules, training, count):
        self.optimizer_calls += 1
        summary = await self.summarize(training, provider=self.provider)
        self.instruction_candidates = []
        for i, module in enumerate(modules):
            proposals = [module.instruction]
            for j in range(count - 1):
                self.optimizer_calls += 1
                demonstrations = self.demo_candidates[i][(j + 1) % len(self.demo_candidates[i])]
                proposals.append(
                    await self.propose(
                        module, modules, summary, demonstrations, provider=self.provider
                    )
                )
            self.instruction_candidates.append(proposals)

    def _program(self, configuration):
        return tuple(
            replace(
                module,
                instruction=self.instruction_candidates[i][configuration[2 * i]],
                demonstrations=self.demo_candidates[i][configuration[2 * i + 1]],
            )
            for i, module in enumerate(self.modules)
        )

    def _density(self, points, observations, smoothing):
        # A mixture over complete configurations retains module/field dependencies.
        terms = []
        for observed in observations:
            probability = smoothing / self.cardinalities + (1 - smoothing) * (points == observed)
            terms.append(np.log(probability).sum(axis=1))
        terms.append(np.full(len(points), -np.log(self.cardinalities).sum()))
        return logsumexp(np.array(terms), axis=0) - math.log(len(terms))

    def _acquire(self, startup_trials, proposals, quantile, smoothing):
        if len(self.trials) < startup_trials:
            return tuple(int(self.nprng.integers(n)) for n in self.cardinalities)
        ranked = sorted(self.trials, key=lambda trial: trial["score"], reverse=True)
        split = max(1, math.ceil(len(ranked) * quantile))
        good = np.array([t["configuration"] for t in ranked[:split]])
        bad = np.array([t["configuration"] for t in ranked[split:]])
        candidates = []
        for _ in range(proposals):
            component = int(self.nprng.integers(len(good) + 1))
            candidate = []
            for dimension, size in enumerate(self.cardinalities):
                if component == len(good) or self.nprng.random() < smoothing:
                    candidate.append(int(self.nprng.integers(size)))
                else:
                    candidate.append(int(good[component, dimension]))
            candidates.append(candidate)
        points = np.array(candidates)
        acquisition = self._density(points, good, smoothing) - self._density(points, bad, smoothing)
        return tuple(int(i) for i in points[int(np.argmax(acquisition))])

    async def _measure(self, configuration, examples, full):
        self.evaluations += 1
        self.example_evaluations += len(examples)
        program = self._program(configuration)
        score = float(await self.evaluate(program, examples))
        if not math.isfinite(score):
            raise ValueError("program score must be finite")
        trial = {"configuration": configuration, "program": program, "score": score, "full": full}
        self.trials.append(trial)
        if full:
            self.full_configurations.add(configuration)
        return trial

    async def _full_check(self, validation):
        scores = {}
        for trial in self.trials:
            key = trial["configuration"]
            if key not in self.full_configurations:
                scores.setdefault(key, []).append(trial["score"])
        if scores:
            key = max(scores, key=lambda k: sum(scores[k]) / len(scores[k]))
            await self._measure(key, validation, True)

    async def run(
        self,
        modules: Sequence[ModulePrompt],
        training: Sequence[str],
        validation: Sequence[str],
        *,
        candidate_sets: int = 5,
        instructions_per_module: int = 5,
        max_bootstrapped_demos: int = 4,
        bootstrap_attempts: int = 20,
        bootstrap_threshold: float = 1,
        iterations: int = 20,
        minibatch_size: int = 20,
        full_eval_interval: int = 5,
        startup_trials: int = 10,
        acquisition_samples: int = 24,
        good_quantile: float = 0.25,
        smoothing: float = 0.2,
        seed: int = 0,
    ) -> dict:
        """Bootstrap traces, ground instructions, search joint assignments and fully rank finalists."""
        self.modules = tuple(modules)
        self.rng, self.nprng = random.Random(seed), np.random.default_rng(seed)
        self.optimizer_calls = self.bootstrap_calls = self.evaluations = (
            self.example_evaluations
        ) = 0
        self.trials, self.full_configurations = [], set()
        await self._bootstrap(
            self.modules,
            training,
            candidate_sets,
            max_bootstrapped_demos,
            bootstrap_attempts,
            bootstrap_threshold,
        )
        await self._propose_instructions(self.modules, training, instructions_per_module)
        self.cardinalities = np.array(
            [size for _ in modules for size in (instructions_per_module, candidate_sets)]
        )
        baseline = (0,) * len(self.cardinalities)
        await self._measure(baseline, validation, True)
        for iteration in range(iterations):
            configuration = self._acquire(
                startup_trials, acquisition_samples, good_quantile, smoothing
            )
            batch = self.rng.sample(list(validation), min(len(validation), minibatch_size))
            await self._measure(configuration, batch, len(batch) == len(validation))
            if (iteration + 1) % full_eval_interval == 0:
                await self._full_check(validation)
        await self._full_check(validation)
        return {
            "best": max((t for t in self.trials if t["full"]), key=lambda t: t["score"]),
            "trials": self.trials,
            "instruction_candidates": self.instruction_candidates,
            "demo_candidates": self.demo_candidates,
            "optimizer_calls": self.optimizer_calls,
            "bootstrap_calls": self.bootstrap_calls,
            "evaluations": self.evaluations,
            "example_evaluations": self.example_evaluations,
        }
