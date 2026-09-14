"""Measure exact and structural recurrence without influencing the mutation chain."""

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

import networkx as nx


def tokens(text: str) -> list[str]:
    """Generic lexical tokens; supply a language tokenizer for domain-specific analysis."""
    return re.findall(r"\w+|[^\w\s]", text)


def normalized_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """Token Levenshtein distance divided by the longer sequence length."""
    denominator = max(len(a), len(b))
    if not denominator:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        following = [i]
        for j, right in enumerate(b, 1):
            following.append(min(row[j] + 1, following[-1] + 1, row[j - 1] + (left != right)))
        row = following
    return row[-1] / denominator


def graph_summary(states: Sequence[str]) -> dict:
    """Count all simple directed node cycles; parallel edges retain transition frequency."""
    visits, transitions = Counter(states), Counter(zip(states, states[1:]))
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(visits)
    graph.add_edges_from(zip(states, states[1:]))
    seen, cumulative = set(), []
    for state in states:
        seen.add(state)
        cumulative.append(len(seen))
    total_degree = 2 * sum(transitions.values())
    degree_entropy, successor_entropy = 0.0, 0.0
    for state, degree in graph.degree():
        if degree:
            p = degree / total_degree
            degree_entropy -= p * math.log2(p)
        outgoing = graph.out_degree(state)
        for target in graph.successors(state):
            p = transitions[state, target] / outgoing
            successor_entropy -= p * math.log2(p)
    count = len(visits)
    # ponytail: exact cycle enumeration can be exponential; analyze shorter chains if needed.
    lengths = Counter(len(cycle) for cycle in nx.simple_cycles(graph))
    return {
        "visits": dict(visits),
        "transitions": dict(transitions),
        "cumulative_unique": tuple(cumulative),
        "revisit_fraction": (len(states) - count) / (len(states) - 1) if len(states) > 1 else 0.0,
        "cycle_lengths": dict(sorted(lengths.items())),
        "mean_degree_entropy": degree_entropy / count if count else 0.0,
        "mean_successor_entropy": successor_entropy / count if count else 0.0,
    }


def analyze(
    programs: Sequence[str],
    *,
    skeleton: Callable[[str], str],
    tokenize: Callable[[str], Sequence[str]] = tokens,
) -> dict:
    """Include the initial state in counts; distances correspond only to mutations."""
    programs = tuple(programs)
    tokenized = [tokenize(program) for program in programs]
    return {
        "programs": graph_summary(programs),
        "skeletons": graph_summary(tuple(skeleton(program) for program in programs)),
        "successive_distances": tuple(
            normalized_distance(a, b) for a, b in zip(tokenized, tokenized[1:])
        ),
    }


def pairwise_distances(
    programs: Sequence[str], *, tokenize: Callable[[str], Sequence[str]] = tokens
) -> tuple[tuple[float, ...], ...]:
    """Compute the symmetric distance matrix for the paper's trajectory heatmaps."""
    tokenized = [tokenize(program) for program in programs]
    # ponytail: quadratic matrix storage; request a trajectory window for large chains.
    matrix = [[0.0] * len(programs) for _ in programs]
    for i, a in enumerate(tokenized):
        for j in range(i):
            matrix[i][j] = matrix[j][i] = normalized_distance(a, tokenized[j])
    return tuple(tuple(row) for row in matrix)
