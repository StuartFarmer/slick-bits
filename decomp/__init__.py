"""Decomposed Prompting with Slick and replaceable sub-task handlers."""

from .agent import Decomp, Execution, Program, Result
from .library import paper_agent

__all__ = ["Decomp", "Execution", "Program", "Result", "paper_agent"]
