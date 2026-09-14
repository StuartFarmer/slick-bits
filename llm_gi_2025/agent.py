"""Search reversible patches using the journal paper's LLM mutation operator."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from slick import prompt
from slick.providers import Provider

from llm_gi.agent import CandidateRejected, Evaluation, Target, nonblank


@dataclass(frozen=True)
class Edit:
    """A frozen replacement addressed by a caller's stable block identifier."""

    target: str
    replacement: str


@dataclass(frozen=True)
class Candidate:
    content: str
    fitness: float
    patch: tuple[Edit, ...] = ()


def whole_artifact(content: str) -> Mapping[str, Mapping[str, Target]]:
    return {"artifact": {"body": Target(0, len(content))}}


class GeneticImprovement:
    """Own mutation, patch replay, independent target searches, and audit records.

    Target discovery maps method IDs to block IDs and current string spans.
    Evaluation receives the complete artifact and the selected method ID.
    The caller owns parsing, profiling, execution isolation, and measurement.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str, str], Awaitable[Evaluation]],
        *,
        language: str = "text",
        requirements: str = "Preserve the fragment's interface and behavior.",
        targets: Callable[[str], Mapping[str, Mapping[str, Target]]] = whole_artifact,
        validate: Callable[[str], None] = nonblank,
        examples: Mapping[str, Sequence[str]] | None = None,
        classic: Callable[[str, random.Random], str] | None = None,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.language, self.requirements = language, requirements
        self.targets, self.validate = targets, validate
        self.examples = {} if examples is None else examples
        self.classic = classic
        self.history: list[dict] = []
        self.best: dict[str, Candidate | None] = {}
        self.baselines: dict[str, Candidate] = {}
        self.evaluations = self.optimizer_calls = 0

    @prompt(template="basic.j2")
    async def basic(self, content: str, *, generated: str) -> str:
        return generated

    @prompt(template="small_changes.j2")
    async def small_changes(self, content: str, examples: Sequence[str], *, generated: str) -> str:
        return generated

    @prompt(template="structural_changes.j2")
    async def structural_changes(
        self, content: str, examples: Sequence[str], *, generated: str
    ) -> str:
        return generated

    async def run(
        self,
        initial: str,
        *,
        mode: Literal["local", "random"] = "local",
        prompt_style: Literal[
            "BASIC", "SMALL_CHANGES", "STRUCTURAL_CHANGES", "STATEMENT"
        ] = "BASIC",
        neighborhood: Literal["paper", "artifact"] = "paper",
        budget: int | None = None,
        seed: int = 123,
    ) -> dict[str, Candidate | None]:
        """Return winners per method; local budget is per method, random is total.

        Local defaults to 100 slots including each baseline. Random defaults to
        1000 independent one-edit patches, sampling methods then blocks uniformly.
        Invalid proposals consume a slot. No retries, deduplication, or sessions.
        """
        local = {"local": True, "random": False}[mode]
        remove_edits = {"paper": True, "artifact": False}[neighborhood]
        budget = (100 if local else 1000) if budget is None else budget
        self.history, self.baselines, self.seen = [], {}, set()
        self.evaluations = self.optimizer_calls = 0
        self.rng = random.Random(seed)
        methods = tuple(self.targets(initial))
        self.best = dict.fromkeys(methods)
        if local:
            for method in methods:
                await self._baseline(initial, method)
                for _ in range(budget - 1):
                    await self._attempt(
                        initial, method, self.best[method], prompt_style, remove_edits
                    )
        else:
            for _ in range(budget):
                method = self.rng.choice(methods)
                await self._attempt(initial, method, None, prompt_style, False)
        return self.best.copy()

    async def _baseline(self, initial: str, method: str) -> None:
        record = {
            "method": method,
            "operation": "baseline",
            "content": initial,
            "patch": (),
            "accepted": False,
        }
        self.history.append(record)
        candidate = await self._assess(initial, method, (), record)
        if candidate is None:
            raise CandidateRejected(f"baseline failed for {method}: {record['status']}")
        self.baselines[method] = self.best[method] = candidate
        record.update(status="baseline", accepted=True)

    async def _attempt(self, initial, method, parent, prompt_style, remove_edits):
        content = initial if parent is None else parent.content
        patch = () if parent is None else parent.patch
        record = {
            "method": method,
            "parent": content,
            "prompt_style": prompt_style,
            "accepted": False,
        }
        self.history.append(record)
        try:
            if patch and remove_edits and self.rng.random() > 0.5:
                index = self.rng.randrange(len(patch))
                patch = patch[:index] + patch[index + 1 :]
                record.update(operation="remove", removed_index=index, patch=patch)
                content = self._apply(initial, method, patch)
            else:
                record["operation"] = "add"
                edit, content = await self._propose(content, method, prompt_style, record)
                patch = (*patch, edit)
            key = (method, patch)
            record.update(content=content, patch=patch, valid=True, unique=key not in self.seen)
            self.seen.add(key)
            candidate = await self._assess(content, method, patch, record)
            incumbent = self.best[method]
            if candidate is not None and (
                incumbent is None or candidate.fitness < incumbent.fitness
            ):
                self.best[method] = candidate
                record["accepted"] = True
        except CandidateRejected as exc:
            record.update(status="invalid", error=str(exc))
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise

    async def _propose(self, content, method, prompt_style, record):
        blocks = self.targets(content).get(method, {})
        if not blocks:
            raise CandidateRejected("no eligible blocks")
        target = self.rng.choice(tuple(blocks))
        span = blocks[target]
        fragment = content[span.start : span.end]
        record.update(target=target, fragment=fragment)
        if prompt_style == "STATEMENT":
            edit = Edit(target, self.classic(fragment, self.rng))
            return edit, self._apply(content, method, (edit,))
        self.optimizer_calls += 1
        if prompt_style == "BASIC":
            raw = await self.basic(fragment, provider=self.provider)
        else:
            operation = {
                "SMALL_CHANGES": self.small_changes,
                "STRUCTURAL_CHANGES": self.structural_changes,
            }[prompt_style]
            raw = await operation(fragment, self.examples[prompt_style], provider=self.provider)
        record.update(raw_response=raw, invalid_suggestions=[])
        # The paper describes labeled or unlabeled fences; keep contents verbatim.
        for match in re.finditer(r"```([^\r\n`]*)\r?\n(.*?)```", raw, re.DOTALL):
            if match.group(1).strip() not in ("", self.language):
                continue
            edit = Edit(target, match.group(2))
            try:
                return edit, self._apply(content, method, (edit,))
            except CandidateRejected as exc:
                record["invalid_suggestions"].append(str(exc))
        raise CandidateRejected("no parseable replacement")

    def _apply(self, content: str, method: str, patch: tuple[Edit, ...]) -> str:
        for edit in patch:
            blocks = self.targets(content).get(method, {})
            if edit.target not in blocks:
                raise CandidateRejected(f"patch target disappeared: {edit.target}")
            span = blocks[edit.target]
            nonblank(edit.replacement)
            content = content[: span.start] + edit.replacement + content[span.end :]
        self.validate(content)
        return content

    async def _assess(self, content, method, patch, record):
        self.evaluations += 1
        try:
            result = await self.evaluate(content, method)
        except CandidateRejected as exc:
            record.update(status="rejected", error=str(exc))
            return None
        except TimeoutError as exc:
            record.update(status="timeout", error=str(exc))
            return None
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        record["evaluation"] = result
        if not result.compiled:
            record["status"] = "compile_failed"
        elif not result.passed:
            record["status"] = "test_failed"
        elif result.fitness is None or not math.isfinite(result.fitness):
            record["status"] = "nonfinite"
        else:
            record.update(status="passed", fitness=result.fitness)
            return Candidate(content, result.fitness, patch)
        return None
