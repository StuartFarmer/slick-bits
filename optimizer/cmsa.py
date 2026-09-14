"""CMSA for maximum independent set; construction behavior follows the authors' archive."""

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

VARIANTS = ("cmsa", "v1", "v2", "v1-perf", "v2-perf")


@dataclass(frozen=True)
class Config:
    variant: str = "cmsa"
    n_solutions: int = 10
    age_max: int = 10
    determinism_rate: float = 0.8
    candidate_list_size: int = 5
    time_limit: float = 5.0
    solver_time_limit: float = 1.0
    iterations: int | None = None

    def validate(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant: {self.variant}")
        for name in ("n_solutions", "age_max", "candidate_list_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.iterations is not None and (
            type(self.iterations) is not int or self.iterations < 1
        ):
            raise ValueError("iterations must be a positive integer or None")
        if not 0 <= self.determinism_rate <= 1:
            raise ValueError("determinism_rate must be in [0, 1]")
        for value in (self.time_limit, self.solver_time_limit):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("time limits must be finite and positive")


def validate_graph(neighbors):
    for v, adjacent in enumerate(neighbors):
        if any(type(u) is not int or not 0 <= u < len(neighbors) for u in adjacent):
            raise ValueError("vertex IDs must be integers in [0, n)")
        if v in adjacent or any(v not in neighbors[u] for u in adjacent):
            raise ValueError("graph must be undirected with no self loops")


def graph_neighbors(graph):
    if graph.is_directed() or graph.is_multigraph() or set(graph) != set(range(len(graph))):
        raise ValueError("expected a simple undirected graph with vertices 0..n-1")
    neighbors = [set(graph[v]) for v in range(len(graph))]
    validate_graph(neighbors)
    return neighbors


def read_graph(path):
    """Read authors' n + zero-based edge pairs, or DIMACS p edge / one-based e lines."""
    lines = [
        line.split()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("c", "#"))
    ]
    if not lines:
        raise ValueError("empty graph file; use 0 for an empty graph")
    header, *edges = lines
    dimacs = header[0] == "p"
    if dimacs:
        if len(header) != 4 or header[1] not in ("edge", "col"):
            raise ValueError("expected DIMACS p edge n m header")
        n, m = int(header[2]), int(header[3])
        if m < 0 or len(edges) != m:
            raise ValueError("edge count differs from DIMACS header")
    else:
        if len(header) != 1:
            raise ValueError("first line must contain the vertex count")
        n = int(header[0])
    if n < 0:
        raise ValueError("negative vertex count")
    neighbors = [set() for _ in range(n)]
    for edge in edges:
        if dimacs:
            if len(edge) != 3 or edge[0] != "e":
                raise ValueError("expected DIMACS e u v")
            edge = edge[1:]
        if len(edge) != 2:
            raise ValueError("expected an edge pair")
        u, v = (int(s) - int(dimacs) for s in edge)
        if not 0 <= u < n or not 0 <= v < n or u == v:
            raise ValueError("invalid endpoint or self loop")
        if v in neighbors[u]:
            raise ValueError("duplicate edge")
        neighbors[u].add(v)
        neighbors[v].add(u)
    return neighbors


def independent(neighbors, vertices):
    return all(type(v) is int and 0 <= v < len(neighbors) for v in vertices) and all(
        not neighbors[v].intersection(vertices) for v in vertices
    )


def selection_probabilities(active, neighbors, age, entropy):
    weights = [1 / (2 + age[v]) + 1 / (1 + len(neighbors[v])) for v in active]
    total = sum(weights)
    probabilities = [w / total for w in weights]
    if entropy:
        h = -sum(p * math.log(p) for p in probabilities)
        # Normalize over AVAILABLE vertices: sum(p + H) = 1 + len(active) * H.
        probabilities = [(p + h) / (1 + len(active) * h) for p in probabilities]
    return probabilities


def generate_solution(
    neighbors,
    age,
    rng,
    *,
    variant="cmsa",
    determinism_rate=0.8,
    candidate_list_size=5,
    deadline=math.inf,
    order=None,
):
    """Construct an independent set, marking newly selected ages zero in-place.

    Degrees are static. A deadline may return a partial independent set.
    The deterministic branch follows the released code (minimum degree), not
    the paper's contradictory argmin(weight) equation. PERF changes availability
    storage only, preserving candidate order and random draws.
    """
    if variant not in VARIANTS or not 0 <= determinism_rate <= 1:
        raise ValueError("invalid construction variant or determinism rate")
    if type(candidate_list_size) is not int or candidate_list_size < 1:
        raise ValueError("candidate_list_size must be a positive integer")
    if len(age) != len(neighbors) or any(type(a) is not int or a < -1 for a in age):
        raise ValueError("one integer age >= -1 is required per vertex")
    active = (
        list(order)
        if order is not None
        else sorted(range(len(neighbors)), key=lambda v: (len(neighbors[v]), v))
    )
    available = (1 << len(neighbors)) - 1
    solution = set()
    # ponytail: active-list scans can be quadratic; profile before adding indexed sampling.
    while active and time.perf_counter() < deadline:
        if rng.random() <= determinism_rate:
            vertex = active[0]
        elif variant == "cmsa":
            vertex = rng.choice(active[:candidate_list_size])
        else:
            probabilities = selection_probabilities(
                active, neighbors, age, variant.startswith("v2")
            )
            vertex = rng.choices(active, weights=probabilities, k=1)[0]
        solution.add(vertex)
        if age[vertex] == -1:
            age[vertex] = 0
        if variant.endswith("-perf"):
            available &= ~(1 << vertex)
            for u in neighbors[vertex]:
                available &= ~(1 << u)
            active = [v for v in active if available & (1 << v)]
        else:
            active = [v for v in active if v != vertex and v not in neighbors[vertex]]
    return solution


def solve_reduced(neighbors, pool, time_limit):
    """Maximize sum(x_v), x_u+x_v <= 1 for edges, binary x, v in pool.

    Return (None, False) when no incumbent is available within the budget.
    Optimality here refers only to the reduced graph.
    """
    deadline = time.perf_counter() + time_limit
    if not pool:
        return set(), True
    vertices = sorted(pool)
    positions = {v: i for i, v in enumerate(vertices)}
    edges = [
        (positions[u], positions[v])
        for u in vertices
        for v in neighbors[u]
        if u < v and v in positions
    ]
    if not edges:
        return set(pool), True
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return None, False
    rows = np.repeat(np.arange(len(edges), dtype=np.int32), 2)
    columns = np.asarray(edges, dtype=np.int32).ravel()
    matrix = coo_matrix(
        (np.ones(len(columns)), (rows, columns)), shape=(len(edges), len(vertices))
    ).tocsc()
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return None, False
    result = milp(
        -np.ones(len(vertices)),
        integrality=np.ones(len(vertices)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, -np.inf, 1),
        options={"time_limit": remaining, "mip_rel_gap": 0.0},
    )
    if result.status not in (0, 1):
        raise RuntimeError(f"MIS solver failed: {result.message}")
    if result.x is None:
        return None, False
    if not np.all(np.isfinite(result.x)) or not np.allclose(result.x, np.rint(result.x), atol=1e-6):
        raise RuntimeError("MIS solver returned nonintegral values")
    solution = {v for v, x in zip(vertices, result.x) if x > 0.5}
    if not independent(neighbors, solution):
        raise RuntimeError("MIS solver returned an infeasible solution")
    return solution, result.status == 0


def adapt(age, solution, age_max):
    for v, value in enumerate(age):
        if value >= 0:
            age[v] = 0 if v in solution else value + 1
            if age[v] >= age_max:
                age[v] = -1


def cmsa(neighbors, config=None, seed=0, constructor=generate_solution):
    config = config or Config()
    config.validate()
    validate_graph(neighbors)
    neighbors = tuple(frozenset(adjacent) for adjacent in neighbors)
    start, cpu_start = time.perf_counter(), time.process_time()
    deadline = start + config.time_limit
    order = sorted(range(len(neighbors)), key=lambda v: (len(neighbors[v]), v))
    age, rng, best = [-1] * len(neighbors), random.Random(seed), set()
    history = [{"seconds": 0.0, "score": 0, "phase": "initial"}]
    iteration, solves, optimal_solves, timeouts = 0, 0, 0, 0
    while neighbors and time.perf_counter() < deadline:
        if config.iterations is not None and iteration >= config.iterations:
            break
        iteration += 1
        for _ in range(config.n_solutions):
            if time.perf_counter() >= deadline:
                break
            raw = constructor(
                neighbors,
                age.copy(),
                rng,
                variant=config.variant,
                determinism_rate=config.determinism_rate,
                candidate_list_size=config.candidate_list_size,
                deadline=deadline,
                order=order.copy(),
            )
            solution = set(raw)
            if len(solution) != len(raw) or not independent(neighbors, solution):
                raise ValueError("constructor returned an invalid independent set")
            # CMSA owns pool membership; a replacement constructor cannot overwrite ages.
            for v in solution:
                if age[v] == -1:
                    age[v] = 0
            if len(solution) > len(best):
                best = solution
                history.append(
                    {
                        "seconds": time.perf_counter() - start,
                        "score": len(best),
                        "phase": "construct",
                    }
                )
        remaining = min(config.solver_time_limit, deadline - time.perf_counter())
        if remaining <= 0:
            break
        pool = {v for v, value in enumerate(age) if value >= 0}
        solution, optimal = solve_reduced(neighbors, pool, remaining)
        solves += 1
        optimal_solves += int(optimal)
        timeouts += int(not optimal)
        if solution is not None:
            if len(solution) > len(best):
                best = solution
                history.append(
                    {
                        "seconds": time.perf_counter() - start,
                        "score": len(best),
                        "phase": "solve",
                    }
                )
            adapt(age, solution, config.age_max)
        if optimal and len(pool) == len(neighbors):
            break  # The full graph was solved, so global optimality is established.
    return {
        "vertices": sorted(best),
        "score": len(best),
        "seed": seed,
        "seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time() - cpu_start,
        "iterations": iteration,
        "solver_calls": solves,
        "optimal_subproblems": optimal_solves,
        "solver_timeouts": timeouts,
        "history": history,
        "config": asdict(config),
    }
