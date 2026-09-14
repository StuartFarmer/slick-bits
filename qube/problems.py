"""Deterministic evaluators for the three QUBE tasks (paper appendix D)."""

import itertools
import math

import numpy as np

SEEDS = {
    "binpack": "def priority(item: float, bins: np.ndarray) -> np.ndarray:\n    return np.zeros_like(bins)\n",
    "capset": "def priority(element: tuple[int, ...], n: int) -> float:\n    return 0.0\n",
    "tsp": "def update_dist(distance_matrix, current_route):\n    return np.zeros_like(distance_matrix)\n",
}

# Trusted, fixed alternatives for an offline integration demonstration, not LLM search.
DEMO_CODES = {
    "binpack": [
        SEEDS["binpack"],
        "def priority(item: float, bins: np.ndarray) -> np.ndarray:\n    return -(bins - item)\n",
        "def priority(item: float, bins: np.ndarray) -> np.ndarray:\n    return -np.sqrt(bins - item)\n",
    ],
    "capset": [
        SEEDS["capset"],
        "def priority(element: tuple[int, ...], n: int) -> float:\n    return -sum(x == 0 for x in element)\n",
        "def priority(element: tuple[int, ...], n: int) -> float:\n    return sum(element)\n",
    ],
    "tsp": [
        SEEDS["tsp"],
        "def update_dist(distance_matrix, current_route):\n"
        "    delta = np.zeros_like(distance_matrix)\n"
        "    for a, b in zip(current_route, current_route[1:] + current_route[:1]):\n"
        "        delta[a, b] = delta[b, a] = 0.1 * distance_matrix[a, b]\n"
        "    return delta\n",
    ],
}


def l2_bound(items, capacity):
    """Martello–Toth L2 via Korf (AAAI 2002), estimated wasted space."""
    items = sorted(items)
    if (
        not items
        or not math.isfinite(capacity)
        or capacity <= 0
        or any(not math.isfinite(x) or not 0 < x <= capacity for x in items)
    ):
        raise ValueError("items must be nonempty, finite, and fit the capacity")
    left, right = 0, len(items) - 1
    waste = carry = 0
    while left <= right:
        residual = capacity - items[right]
        right -= 1
        while left <= right and items[left] <= residual:
            carry += items[left]
            left += 1
        waste += max(0, residual - carry)
        carry = max(0, carry - residual)
    return math.ceil((sum(items) + waste) / capacity)


def weibull(items=1000, instances=5, seed=0):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(instances):
        sizes = np.clip(np.rint(rng.weibull(3, items) * 45), 1, 100).astype(int).tolist()
        result.append(
            {
                "id": f"weibull-{i}",
                "capacity": 100,
                "items": sizes,
                "lower_bound": l2_bound(sizes, 100),
            }
        )
    return result


def read_or_library(path):
    """OR-Library binpack text: count, then id/capacity/item-count/reference/items."""
    words = iter(path.read_text().split())
    try:
        count = int(next(words))
        result = []
        for _ in range(count):
            name = next(words)
            capacity, size, reference = (int(next(words)) for _ in range(3))
            if size < 1:
                raise ValueError("item count must be positive")
            items = [int(next(words)) for _ in range(size)]
            # OR files report best-known packings; compute L2 for the paper metric.
            result.append(
                {
                    "id": name,
                    "capacity": capacity,
                    "items": items,
                    "reference_bins": reference,
                    "lower_bound": l2_bound(items, capacity),
                }
            )
        if next(words, None) is not None or not result:
            raise ValueError("empty data or trailing OR-Library fields")
        return result
    except (StopIteration, RuntimeError) as exc:
        raise ValueError("truncated OR-Library file") from exc


def binpack(items, capacity, priority):
    bins = np.full(len(items), capacity, dtype=float)
    for item in items:
        valid = np.flatnonzero(bins >= item)
        scores = np.asarray(priority(float(item), bins[valid].copy()), dtype=float)
        if scores.shape != valid.shape or not np.all(np.isfinite(scores)):
            raise ValueError("priority must return one finite score per feasible bin")
        bins[valid[int(np.argmax(scores))]] -= item
    return int(np.count_nonzero(bins != capacity))


def capset(n, priority):
    if not isinstance(n, int) or not 1 <= n <= 10:
        raise ValueError("cap-set dimension must be an integer between 1 and 10")
    vectors = np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int32)
    powers = 3 ** np.arange(n - 1, -1, -1)
    priorities = np.array([priority(tuple(int(x) for x in v), n) for v in vectors], dtype=float)
    if priorities.shape != (len(vectors),) or not np.all(np.isfinite(priorities)):
        raise ValueError("priority must return a finite scalar for each vector")
    chosen = np.empty_like(vectors)
    size = 0
    while np.any(priorities != -np.inf):
        index = int(np.argmax(priorities))
        vector = vectors[index]
        blocked = ((-chosen[:size] - vector) % 3) @ powers
        priorities[blocked] = -np.inf
        priorities[index] = -np.inf
        chosen[size] = vector
        size += 1
    return chosen[:size]


def tour_length(route, distances):
    return float(sum(distances[a, b] for a, b in zip(route, route[1:] + route[:1])))


def two_opt(route, distances):
    route = list(route)
    # ponytail: full 2-opt scan; candidate edge lists if large TSP instances require it.
    while True:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                a, b, c, d = (
                    route[i - 1],
                    route[i],
                    route[j],
                    route[(j + 1) % len(route)],
                )
                delta = distances[a, c] + distances[b, d] - distances[a, b] - distances[c, d]
                if delta < -1e-12:
                    route[i : j + 1] = reversed(route[i : j + 1])
                    improved = True
        if not improved:
            return route


def guided_search(cities, update_dist, iterations=100):
    cities = np.asarray(cities, dtype=float)
    if (
        cities.ndim != 2
        or cities.shape[1] != 2
        or len(cities) < 3
        or not np.all(np.isfinite(cities))
    ):
        raise ValueError("cities must contain at least three finite 2D coordinates")
    original = np.linalg.norm(cities[:, None] - cities[None, :], axis=-1)
    distances = original.copy()
    route = list(range(len(cities)))
    best = route.copy()
    for _ in range(iterations):
        route = two_opt(route, distances)
        if tour_length(route, original) < tour_length(best, original):
            best = route.copy()
        update = np.asarray(update_dist(distances.copy(), best.copy()), dtype=float)
        if update.shape != distances.shape or not np.all(np.isfinite(update)):
            raise ValueError("update_dist must return a finite matrix of the same shape")
        if not np.allclose(update, update.T, rtol=0, atol=1e-12):
            raise ValueError("update_dist must preserve symmetric distances")
        distances += (update + update.T) / 2
        if not np.all(np.isfinite(distances)):
            raise ValueError("updated distances overflowed")
    return best, tour_length(best, original)


def validate_data(task, instances):
    if not isinstance(instances, list) or not instances:
        raise ValueError("dataset must be a nonempty JSON list")
    if task == "capset":
        if any(type(n) is not int or not 1 <= n <= 10 for n in instances):
            raise ValueError("cap-set dimensions must be integers from 1 to 10")
    elif task == "binpack":
        for instance in instances:
            bound = l2_bound(instance["items"], instance["capacity"])
            supplied = instance.get("lower_bound", bound)
            if not math.isfinite(supplied) or not 0 < supplied <= len(instance["items"]):
                raise ValueError("invalid bin packing lower bound")
            instance["lower_bound"] = supplied
    elif task == "tsp":
        for instance in instances:
            cities = np.asarray(instance["cities"], dtype=float)
            if (
                cities.ndim != 2
                or cities.shape[1] != 2
                or len(cities) < 3
                or not np.all(np.isfinite(cities))
            ):
                raise ValueError("TSP instances require at least three finite 2D cities")
            optimum = instance.get("optimum")
            if optimum is not None and (not math.isfinite(optimum) or optimum <= 0):
                raise ValueError("TSP optimum must be positive and finite")
        if any("optimum" in x for x in instances) and not all("optimum" in x for x in instances):
            raise ValueError("supply optima for every TSP instance or none")
    else:
        raise ValueError(f"unknown task: {task}")


def evaluate(task, heuristic, instances, iterations=100):
    validate_data(task, instances)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if task == "binpack":
        values = [float(binpack(x["items"], x["capacity"], heuristic)) for x in instances]
        lower = sum(x["lower_bound"] for x in instances)
        score = 1 - sum(values) / lower
    elif task == "capset":
        values = [float(len(capset(n, heuristic))) for n in instances]
        score = float(np.mean(values))
    else:
        values = [guided_search(x["cities"], heuristic, iterations)[1] for x in instances]
        score = (
            -float(np.mean([v / x["optimum"] - 1 for v, x in zip(values, instances)]))
            if "optimum" in instances[0]
            else -float(np.mean(values))
        )
    return {"signature": values, "score": score}
