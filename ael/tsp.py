"""Euclidean TSP, exact reference tours, and the isolated candidate worker."""

import contextlib
import inspect
import json
import math
import operator
import os
import random
import sys
import time

import numpy as np


def distances(coordinates):
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("coordinates must contain at least two 2D points")
    if not np.isfinite(points).all():
        raise ValueError("coordinates must be finite")
    return np.linalg.norm(points[:, None] - points[None, :], axis=2)


def greedy(current_node, destination_node, unvisited_nodes, distance_matrix):
    return min(unvisited_nodes, key=lambda n: distance_matrix[current_node, n])


def paper_heuristic(
    current_node, destination_node, unvisited_nodes, distance_matrix, threshold=0.7
):
    """Figure 6, with the singleton case made explicit to avoid empty means."""
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    scores = {}
    for node in unvisited_nodes:
        others = [distance_matrix[node, n] for n in unvisited_nodes if n != node]
        scores[node] = (
            0.4 * distance_matrix[current_node, node]
            - 0.3 * np.mean(others)
            + 0.2 * np.std(others)
            - 0.1 * distance_matrix[destination_node, node]
        )
    if min(scores.values()) > threshold:
        return greedy(current_node, destination_node, unvisited_nodes, distance_matrix)
    return min(scores, key=scores.get)


def construct_tour(selector, matrix):
    # Immutable backing prevents a candidate from changing the evaluation distances.
    matrix = np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(matrix.shape)
    unvisited = list(range(1, len(matrix)))
    tour = [0]
    while unvisited:
        node = selector(tour[-1], 0, unvisited.copy(), matrix)
        if isinstance(node, (bool, np.bool_)):
            raise ValueError("selector returned a boolean instead of a node ID")
        try:
            node = operator.index(node)
        except TypeError as exc:
            raise ValueError("selector must return an integer node ID") from exc
        if node not in unvisited:
            raise ValueError(f"selector returned an unavailable node: {node}")
        tour.append(node)
        unvisited.remove(node)
    return tour + [0]


def tour_length(tour, matrix):
    n = len(matrix)
    if (
        len(tour) != n + 1
        or any(type(v) is not int for v in tour)
        or tour[0] != 0
        or tour[-1] != 0
        or sorted(tour[:-1]) != list(range(n))
    ):
        raise ValueError("tour must visit every node once and return to node zero")
    return float(sum(matrix[a, b] for a, b in zip(tour, tour[1:])))


def exact_length(matrix, timeout=60):
    """Undirected degree-two MILP with iterative subtour cuts; require optimality."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, vstack

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("solver timeout must be positive and finite")
    n = len(matrix)
    if n == 2:
        return float(2 * matrix[0, 1])
    deadline = time.monotonic() + timeout
    left, right = np.triu_indices(n, 1)
    m = len(left)
    constraints = csr_matrix(
        (np.ones(2 * m), (np.concatenate([left, right]), np.tile(np.arange(m), 2))),
        shape=(n, m),
    )
    lower, upper = [2.0] * n, [2.0] * n
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("exact TSP reference exceeded its deadline")
        result = milp(
            matrix[left, right],
            integrality=np.ones(m),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(constraints, lower, upper),
            options={"time_limit": remaining, "mip_rel_gap": 0},
        )
        if result.status != 0 or result.x is None:
            raise RuntimeError(f"reference was not proven optimal: {result.message}")
        neighbors = [[] for _ in range(n)]
        for edge in np.flatnonzero(result.x > 0.5):
            a, b = int(left[edge]), int(right[edge])
            neighbors[a].append(b)
            neighbors[b].append(a)
        if any(len(nodes) != 2 for nodes in neighbors):
            raise RuntimeError("solver returned an invalid degree-two solution")
        components, unseen = [], set(range(n))
        while unseen:
            stack, component = [min(unseen)], set()
            while stack:
                node = stack.pop()
                if node in unseen:
                    unseen.remove(node)
                    component.add(node)
                    stack.extend(neighbors[node])
            components.append(component)
        if len(components) == 1:
            return float(matrix[left, right][result.x > 0.5].sum())
        for component in components:
            inside = np.isin(left, list(component)) & np.isin(right, list(component))
            constraints = vstack([constraints, csr_matrix(inside.astype(float))], format="csr")
            lower.append(-np.inf)
            upper.append(len(component) - 1)


def worker():
    request = json.load(sys.stdin)
    random.seed(request["seed"])
    np.random.seed(request["seed"] % (2**32))
    namespace = {"__name__": "candidate"}
    # Candidate stdout is diagnostic noise, not the worker's JSON protocol.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        exec(compile(request["code"], "<candidate>", "exec"), namespace)
        selector = namespace["select_next_node"]
        inspect.signature(selector).bind(0, 0, [1], np.zeros((2, 2)))
        tours = [construct_tour(selector, distances(points)) for points in request["coordinates"]]
    json.dump({"tours": tours}, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    worker()
