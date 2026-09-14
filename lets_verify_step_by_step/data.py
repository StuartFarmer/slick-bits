"""Adapters for the official PRM800K phase-2 labels and scored evaluation samples."""

import gzip
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from .agent import TrainingExample, process_examples


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream a downloaded official JSONL file, optionally gzip compressed."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def phase2_examples(row: dict) -> tuple[TrainingExample, ...]:
    """Use original phase-2 completions through the first error, without repairs.

    Skip QC, screening, incomplete work, and flagged/null ratings. Phase 1 uses
    branching/human continuations and is deliberately a separate unsupported
    format here, rather than silently treating those branches as one solution.
    """
    if row.get("is_quality_control_question") or row.get("is_initial_screening_question"):
        return ()
    if row["generation"] is None:
        raise ValueError("This adapter requires phase-2 PRM800K labels")
    if row["label"]["finish_reason"] not in ("solution", "found_error"):
        return ()
    steps, labels = [], []
    for step in row["label"]["steps"]:
        completion = step["completions"][0]
        if completion.get("flagged") or completion["rating"] is None:
            return ()
        if not completion["text"].strip():
            raise ValueError("PRM800K contains an empty labeled step")
        steps.append(completion["text"])
        labels.append(completion["rating"])
        if labels[-1] == -1:
            break
    return process_examples(row["question"]["problem"], steps, labels)


def scored_sample_trial(
    rows: Iterable[dict],
    *,
    n: int,
    samples_per_problem: int = 1860,
    method: Literal["prm", "orm"] = "prm",
    rng: random.Random,
) -> float:
    """One official best-of-N trial, adapted from prm800k/eval/eval.py (MIT).

    Preserve missing-completion padding, shuffle-before-filter ordering, answer
    key fallback, highest-score selection and problem-level accuracy. Caller
    owns repeated trials and RNG state. See UPSTREAM_LICENSE for attribution.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["problem"]].append(row)
    correct = 0
    for samples in grouped.values():
        padded = samples + [None] * (samples_per_problem - len(samples))
        rng.shuffle(padded)
        eligible = [
            sample
            for sample in padded[:n]
            if sample is not None and sample.get("answer", sample.get("given_answer")) is not None
        ]
        best = max(eligible, key=lambda sample: sample[f"{method}_score"], default=None)
        correct += best is not None and best["is_correct"]
    return correct / len(grouped)
