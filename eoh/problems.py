"""EoH fitness tasks and small, explicit guided local search implementations."""

import math
import time

import numpy as np

INTERFACES = {
    "binpacking": ("score", ("item", "bins")),
    "tsp": ("update_edge_distance", ("edge_distance", "local_opt_tour", "edge_n_used")),
    "flowshop": ("get_matrix_and_jobs", ("current_sequence", "time_matrix", "m", "n")),
}
TASKS = {
    "binpacking": """Design score(item, bins) to minimize bins used in online bin packing.
    item is a positive integer; bins is a NumPy vector of remaining capacities of
    feasible, already-open bins (>= item, including exact fits). Return a NumPy
    vector of scores of the same shape; the highest score wins, first index breaks
    ties. A new bin opens only when no open bin fits. Use no random components.""",
    "tsp": """Design update_edge_distance(edge_distance, local_opt_tour, edge_n_used)
    to escape local optima in TSP guided local search using 2-opt and relocate.
    Inputs are NumPy arrays: the original n*n distance matrix, a closed 1-D tour
    of node IDs (first node repeated at the end), and an n*n matrix counting how
    often tour edges have been used in perturbation. Return a finite n*n guiding
    distance matrix. Actual tour quality is always measured on original distances.""",
    "flowshop": """Design get_matrix_and_jobs(current_sequence, time_matrix, m, n)
    to escape local optima in permutation flow shop scheduling with swap and
    relocate. current_sequence is a NumPy array of job IDs, time_matrix is an n*m
    array of processing times; m and n are machine and job counts. Return a finite,
    nonnegative n*m new_matrix and a nonempty 1-D NumPy array perturb_jobs of unique
    job IDs in priority order. Moves involving those jobs are tried on new_matrix;
    local search then resumes on original times, minimizing original makespan.""",
}


def array(value, shape=None, *, nonnegative=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError("expected a finite real numeric array")
    result = raw.astype(float)
    if shape is not None and result.shape != shape:
        raise ValueError(f"expected shape {shape}, got {result.shape}")
    if nonnegative and np.any(result < 0):
        raise ValueError("values must be nonnegative")
    return result


def validate_dataset(dataset):
    problem = dataset["problem"]
    if problem not in INTERFACES or not dataset["instances"]:
        raise ValueError("choose a supported problem and a nonempty instance set")
    for instance in dataset["instances"]:
        if problem == "binpacking":
            capacity = instance["capacity"]
            items = array(instance["items"])
            if (
                type(capacity) is not int
                or capacity < 1
                or items.ndim != 1
                or not items.size
                or np.any(items < 1)
                or np.any(items > capacity)
                or np.any(items != np.floor(items))
            ):
                raise ValueError("packing needs positive integer items <= integer capacity")
            bound = (
                instance["lower_bound"]
                if "lower_bound" in instance
                else bin_lower_bound(items.astype(int), capacity)
            )
            if not math.isfinite(bound) or bound <= 0 or bound > len(items) or bound != int(bound):
                raise ValueError("invalid packing lower bound")
            instance["lower_bound"] = int(bound)
        else:
            matrix = array(instance["distances" if problem == "tsp" else "times"], nonnegative=True)
            if matrix.ndim != 2 or min(matrix.shape) < 1:
                raise ValueError("expected a nonempty matrix")
            if problem == "tsp" and (
                matrix.shape[0] < 3
                or matrix.shape[0] != matrix.shape[1]
                or not np.allclose(matrix, matrix.T)
                or np.any(np.diag(matrix) != 0)
            ):
                raise ValueError("TSP needs a symmetric n*n matrix, n >= 3, zero diagonal")
            for name in ("optimum", "reference"):
                if name in instance and (not math.isfinite(instance[name]) or instance[name] <= 0):
                    raise ValueError(f"{name} must be positive and finite")
    if problem == "tsp":
        known = ["optimum" in instance for instance in dataset["instances"]]
        if any(known) and not all(known):
            raise ValueError("supply optima for every TSP instance or none")
    return dataset


def bin_lower_bound(items, capacity):
    """Martello–Toth L2 for positive integer sizes, including the volume bound."""
    sizes = np.asarray(items, dtype=np.int64)
    bound = math.ceil(int(sizes.sum()) / capacity)
    # Only thresholds at which a group changes are needed, even for large capacities.
    thresholds = {
        0,
        *(int(x) + 1 for x in sizes if 2 * x <= capacity),
        *(capacity - int(x) + 1 for x in sizes if 2 * x > capacity),
    }
    for alpha in thresholds:
        if alpha > capacity // 2:
            continue
        j1 = sizes > capacity - alpha
        j2 = (sizes > capacity / 2) & ~j1
        j3 = (sizes >= alpha) & (sizes <= capacity / 2)
        leftover = int(sizes[j3].sum()) - int((capacity - sizes[j2]).sum())
        bound = max(bound, int(j1.sum() + j2.sum()) + max(0, math.ceil(leftover / capacity)))
    return bound


def first_fit(item, bins):
    return -np.arange(len(bins))


def best_fit(item, bins):
    return item - bins


def funsearch(item, bins):
    scores = (bins - bins.max()) ** 2 / item + bins**2 / item**2 + bins**2 / item**3
    scores[bins > item] *= -1
    scores[1:] -= scores[:-1].copy()
    return scores


def paper_binpacking(item, bins):
    diff = bins - item
    combined = (1 - diff / bins) * np.sqrt(diff)
    adjustment = np.where(diff > item * 3, combined + 0.8, combined + 0.3)
    decay = np.exp(-diff)  # Algebraic equivalent of Figure 6 without exp(diff) overflow.
    return bins * decay**2 / (1 + 0.7 * decay) + adjustment


def pack(items, capacity, score):
    remaining = np.empty(len(items), dtype=float)
    used = 0
    for item in items:
        feasible = np.flatnonzero(remaining[:used] >= item)
        if len(feasible):
            values = array(score(int(item), remaining[feasible].copy()), (len(feasible),))
            index = feasible[int(np.argmax(values))]
        else:
            index = used
            remaining[index] = capacity
            used += 1
        remaining[index] -= item
    return used


def tour_length(matrix, route):
    return float(matrix[route, np.roll(route, -1)].sum())


def makespan(matrix, sequence):
    end = np.zeros(matrix.shape[1])
    for job in sequence:
        end[0] += matrix[job, 0]
        for machine in range(1, len(end)):
            end[machine] = max(end[machine], end[machine - 1]) + matrix[job, machine]
    return float(end[-1])


def neighbors(sequence, problem, jobs=None):
    n = len(sequence)
    positions = range(n) if jobs is None else [sequence.index(job) for job in jobs]
    for i in positions:
        for j in range(n):
            if i == j:
                continue
            moved = sequence.copy()
            moved.insert(j, moved.pop(i))
            yield moved
            if problem == "tsp":
                if j > i + 1:
                    yield sequence[:i] + sequence[i : j + 1][::-1] + sequence[j + 1 :]
            elif jobs is not None or j > i:
                moved = sequence.copy()
                moved[i], moved[j] = moved[j], moved[i]
                yield moved


def local_search(matrix, sequence, problem, deadline, jobs=None, one_move=False):
    objective = tour_length if problem == "tsp" else makespan
    value = objective(matrix, sequence)
    # ponytail: exhaustive O(n^3) tour / O(n^3*m) schedule scans; use move deltas for scale.
    while time.perf_counter() < deadline:
        best, best_value = sequence, value
        for candidate in neighbors(sequence, problem, jobs):
            if time.perf_counter() >= deadline:
                break
            cost = objective(matrix, candidate)
            if cost < best_value - 1e-12:
                best, best_value = candidate, cost
        if best is sequence:
            break
        sequence, value = best, best_value
        if one_move:
            break
    return sequence, value


def tsp_penalty(edge_distance, local_opt_tour, edge_n_used):
    matrix = edge_distance.copy()
    a, b = local_opt_tour[:-1], local_opt_tour[1:]
    penalty = edge_distance.mean() / (1 + edge_n_used[a, b])
    matrix[a, b] += penalty
    matrix[b, a] += penalty
    return matrix


def flow_perturb(current_sequence, time_matrix, m, n):
    machines = np.random.choice(m, max(1, int(0.3 * m)), replace=False)
    weights = np.random.rand(len(machines))
    averages = np.average(time_matrix[:, machines], axis=1, weights=weights)
    jobs = np.argsort(averages)[-max(1, int(0.3 * n)) :]
    matrix = time_matrix.copy()
    matrix[jobs[:, None], machines] *= np.random.uniform(0.8, 1.2, (len(jobs), len(machines)))
    return matrix, jobs


def flow_output(output, shape):
    matrix, jobs = output
    matrix = array(matrix, shape, nonnegative=True)
    jobs = np.asarray(jobs)
    if (
        jobs.ndim != 1
        or not jobs.size
        or jobs.dtype.kind not in "iu"
        or len(np.unique(jobs)) != len(jobs)
        or np.any(jobs < 0)
        or np.any(jobs >= shape[0])
    ):
        raise ValueError("perturb_jobs must contain unique valid integer job IDs")
    return matrix, jobs.tolist()


def solve_tsp(matrix, update, *, iterations=1000, seconds=60):
    matrix = array(matrix, nonnegative=True)
    deadline = time.perf_counter() + seconds
    sequence = [0]
    unvisited = set(range(1, len(matrix)))
    while unvisited:
        node = min(unvisited, key=lambda node: (matrix[sequence[-1], node], node))
        sequence.append(node)
        unvisited.remove(node)
    sequence, value = local_search(matrix, sequence, "tsp", deadline)
    best, best_value = sequence.copy(), value
    used = np.zeros_like(matrix)
    for _ in range(iterations):
        if time.perf_counter() >= deadline:
            break
        closed = np.array([*sequence, sequence[0]])
        guiding = array(update(matrix.copy(), closed.copy(), used.copy()), matrix.shape)
        a, b = closed[:-1], closed[1:]
        used[a, b] += 1
        used[b, a] += 1
        sequence, _ = local_search(guiding, sequence, "tsp", deadline, one_move=True)
        sequence, value = local_search(matrix, sequence, "tsp", deadline)
        if value < best_value:
            best, best_value = sequence.copy(), value
    return best, best_value


def neh(matrix, deadline=math.inf):
    order = sorted(range(len(matrix)), key=lambda job: (-matrix[job].sum(), job))
    sequence = []
    for index, job in enumerate(order):
        if time.perf_counter() >= deadline:
            return sequence + order[index:]
        sequence = min(
            (sequence[:i] + [job] + sequence[i:] for i in range(len(sequence) + 1)),
            key=lambda candidate: makespan(matrix, candidate),
        )
    return sequence


def solve_flowshop(matrix, update, *, iterations=1000, seconds=60):
    matrix = array(matrix, nonnegative=True)
    deadline = time.perf_counter() + seconds
    sequence = neh(matrix, deadline)
    sequence, value = local_search(matrix, sequence, "flowshop", deadline)
    best, best_value = sequence.copy(), value
    n, m = matrix.shape
    for _ in range(iterations):
        if time.perf_counter() >= deadline:
            break
        guiding, jobs = flow_output(update(np.array(sequence), matrix.copy(), m, n), matrix.shape)
        sequence, _ = local_search(guiding, sequence, "flowshop", deadline, jobs, one_move=True)
        sequence, value = local_search(matrix, sequence, "flowshop", deadline)
        if value < best_value:
            best, best_value = sequence.copy(), value
    return best, best_value


def evaluate(dataset, function, *, iterations=1000, seconds=60, seed=0):
    validate_dataset(dataset)
    values, fitnesses, solutions = [], [], []
    problem = dataset["problem"]
    for index, instance in enumerate(dataset["instances"]):
        np.random.seed((seed + index) % (2**32))
        if problem == "binpacking":
            # Probe even when all incoming items require new bins.
            capacity = instance["capacity"]
            array(function(1, np.array([1.0, float(capacity)])), (2,))
            value = pack(instance["items"], capacity, function)
            if value < instance["lower_bound"]:
                raise ValueError("supplied lower bound exceeds a feasible bin count")
            fitness = instance["lower_bound"] / value
            solution = None
            metric = "mean_lower_bound_over_bins"
        elif problem == "tsp":
            matrix = array(instance["distances"])
            tour = np.array([*range(len(matrix)), 0])
            array(function(matrix.copy(), tour, np.zeros_like(matrix)), matrix.shape)
            solution, value = solve_tsp(matrix, function, iterations=iterations, seconds=seconds)
            fitness = -100 * (value / instance["optimum"] - 1) if "optimum" in instance else -value
            metric = (
                "negative_mean_gap_percent"
                if "optimum" in instance
                else "negative_mean_tour_length"
            )
        else:
            matrix = array(instance["times"])
            n, m = matrix.shape
            flow_output(function(np.arange(n), matrix.copy(), m, n), matrix.shape)
            solution, value = solve_flowshop(
                matrix, function, iterations=iterations, seconds=seconds
            )
            fitness = -value
            metric = "negative_mean_makespan"
        values.append(value)
        fitnesses.append(fitness)
        solutions.append(solution)
    return {
        "fitness": float(np.mean(fitnesses)),
        "metric": metric,
        "values": values,
        "solutions": solutions,
    }


def generate_dataset(problem, *, size, count, seed=0, capacity=100, machines=5):
    if size < 1 or count < 1 or capacity < 1 or machines < 0 or (problem == "tsp" and size < 3):
        raise ValueError("positive dataset dimensions required; TSP needs at least three cities")
    rng = np.random.default_rng(seed)
    instances = []
    for _ in range(count):
        if problem == "binpacking":
            items = np.clip(np.rint(rng.weibull(0.5, size) * capacity / 10), 1, capacity).astype(
                int
            )
            instances.append({"items": items.tolist(), "capacity": capacity})
        elif problem == "tsp":
            xy = rng.random((size, 2))
            instances.append(
                {"distances": np.linalg.norm(xy[:, None] - xy[None, :], axis=-1).tolist()}
            )
        elif problem == "flowshop":
            m = int(rng.integers(2, 21)) if machines == 0 else machines
            instances.append({"times": rng.random((size, m)).tolist()})
        else:
            raise ValueError(f"unknown problem: {problem}")
    return validate_dataset(
        {
            "problem": problem,
            "instances": instances,
            "generator": {"seed": seed, "synthetic": True},
        }
    )
