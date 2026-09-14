"""Problem-agnostic Darwin Gödel Machine."""

from .agent import (
    DGM,
    SEED_AGENT,
    Attempt,
    Candidate,
    CandidateRejected,
    Config,
    Evaluation,
    ExecutionError,
    Generation,
    Proposal,
    Result,
)

__all__ = [
    "DGM",
    "SEED_AGENT",
    "Attempt",
    "Candidate",
    "CandidateRejected",
    "Config",
    "Evaluation",
    "ExecutionError",
    "Generation",
    "Proposal",
    "Result",
]
