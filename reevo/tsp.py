"""A small NumPy Ant System evaluator for ReEvo's TSP heuristic measures."""

import asyncio
import contextlib
import json
import math
import os
import random
import signal
import sys
import tempfile
from pathlib import Path

import numpy as np


def task_and_seed(black_box=False):
    from .core import Task

    if black_box:
        task = Task(
            "Solve a black-box graph optimization problem via stochastic solution sampling.",
            "The input edge_attr has shape (n_edges, 1). Return a finite nonnegative "
            "NumPy vector of shape (n_edges,), indicating how promising each edge is. "
            "Use a deterministic function.",
            "def heuristics(edge_attr: np.ndarray) -> np.ndarray:",
            black_box=True,
        )
        source = (
            "import numpy as np\n" + task.signature + "\n    return np.ones(edge_attr.shape[0])\n"
        )
    else:
        task = Task(
            "Solve the Traveling Salesman Problem with Ant Colony Optimization: "
            "minimize the length of a tour visiting every node once and returning to its start.",
            "Given a NumPy distance matrix of shape (n, n), return finite nonnegative "
            "heuristic measures of shape (n, n). Higher values favor selecting an edge. "
            "Diagonal entries are zero and self-edges are never selected. "
            "Use a deterministic NumPy function.",
            "def heuristics(distance_matrix: np.ndarray) -> np.ndarray:",
            initial_reflection="Combine useful factors; sparsify unpromising edges.",
        )
        source = (
            "import numpy as np\n" + task.signature + "\n    return np.ones_like(distance_matrix)\n"
        )
    return task, source


def make_instances(count, nodes, *, seed):
    if count < 1 or nodes < 3:
        raise ValueError("need at least one instance and three nodes")
    coordinates = np.random.default_rng(seed).random((count, nodes, 2))
    return np.linalg.norm(coordinates[:, :, None, :] - coordinates[:, None, :, :], axis=-1)


def aco(distances, heuristic, *, ants=30, iterations=100, seed=0):
    """Return the best closed-tour length. Alpha=beta=1, evaporation=0.1.

    Every ant deposits 1/tour_length on its edges. No local search is added.
    Zero-weight feasible choices fall back to uniform sampling.
    """
    if ants < 1 or iterations < 1:
        raise ValueError("ants and iterations must be positive")
    n = len(distances)
    eta = np.array(heuristic, dtype=float, copy=True)
    if eta.shape != (n, n) or not np.isfinite(eta).all() or (eta < 0).any():
        raise ValueError("heuristic must be a finite, nonnegative (n, n) matrix")
    np.fill_diagonal(eta, 0)
    eta /= eta.max() or 1  # Keep products finite without changing relative weights.
    pheromone = np.ones((n, n))
    rng = np.random.default_rng(seed)
    rows = np.arange(ants)
    best = math.inf
    for _ in range(iterations):
        tours = np.empty((ants, n), dtype=int)
        tours[:, 0] = rng.integers(n, size=ants)
        visited = np.zeros((ants, n), dtype=bool)
        visited[rows, tours[:, 0]] = True
        for step in range(1, n):
            current = tours[:, step - 1]
            weights = pheromone[current] * eta[current]
            weights[visited] = 0
            empty = weights.sum(axis=1) == 0
            weights[empty] = ~visited[empty]
            probabilities = weights / weights.sum(axis=1, keepdims=True)
            cumulative = np.cumsum(probabilities, axis=1)
            cumulative[:, -1] = 1
            chosen = (cumulative <= rng.random((ants, 1))).sum(axis=1)
            tours[:, step] = chosen
            visited[rows, chosen] = True
        following = np.roll(tours, -1, axis=1)
        lengths = distances[tours, following].sum(axis=1)
        best = min(best, float(lengths.min()))
        pheromone *= 0.9
        deposits = np.broadcast_to(1 / np.maximum(lengths[:, None], 1e-12), tours.shape)
        np.add.at(pheromone, (tours, following), deposits)
        np.add.at(pheromone, (following, tours), deposits)
    return best


class TSPEvaluator:
    """Evaluate generated Python in a child process with a wall-clock timeout.

    A subprocess is NOT a security sandbox. Run live searches in an isolated
    environment: generated code has the child process's filesystem permissions.
    The child receives no API credentials. Each candidate gets a temporary cwd.
    """

    def __init__(self, instances, *, black_box=False, ants=30, iterations=100, seed=0, timeout=30):
        instances = np.array(instances, dtype=float, copy=True)
        if (
            instances.ndim != 3
            or instances.shape[0] < 1
            or instances.shape[1] < 3
            or instances.shape[1] != instances.shape[2]
            or not np.isfinite(instances).all()
            or (instances < 0).any()
            or not np.allclose(instances, instances.transpose(0, 2, 1))
        ):
            raise ValueError("instances must be finite nonnegative symmetric distance matrices")
        if ants < 1 or iterations < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("ants, iterations and timeout must be positive")
        self.payload = dict(
            instances=instances.tolist(),
            black_box=black_box,
            ants=ants,
            iterations=iterations,
            seed=seed,
        )
        self.timeout = timeout

    async def __call__(self, code):
        # Strip inherited credentials; preserve Python package discovery for local checkouts.
        env = {
            key: os.environ[key]
            for key in ("PATH", "PYTHONPATH", "SYSTEMROOT")
            if key in os.environ
        }
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="reevo-") as directory:
            # File-backed stdout prevents noisy candidates from filling parent memory.
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    cwd=directory,
                    env=env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    await asyncio.wait_for(
                        process.communicate(
                            json.dumps(dict(self.payload, code=code), allow_nan=False).encode()
                        ),
                        timeout=self.timeout,
                    )
                    stdout.seek(0)
                    answer = stdout.read(8192).decode(errors="replace")
                    if process.returncode:
                        stderr.seek(0)
                        raise ValueError(
                            stderr.read(4096).decode(errors="replace").strip()
                            or f"evaluator exited with {process.returncode}"
                        )
                    value = float(json.loads(answer)["objective"])
                    if not math.isfinite(value):
                        raise ValueError("nonfinite evaluation")
                    return value
                finally:
                    # Also reap descendants when a candidate crashes, times out or is cancelled.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()


def worker():
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
    payload = json.load(sys.stdin)
    random.seed(payload["seed"])
    np.random.seed(payload["seed"] % 2**32)
    scope = {"np": np, "__name__": "generated_heuristic"}
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(payload["code"], "<heuristic>", "exec"), scope)
        scores = []
        for index, distances in enumerate(np.asarray(payload["instances"], dtype=float)):
            argument = distances.reshape(-1, 1) if payload["black_box"] else distances
            heuristic = np.asarray(scope["heuristics"](argument.copy()), dtype=float)
            if payload["black_box"]:
                if heuristic.shape != (distances.size,):
                    raise ValueError("black-box output must have shape (n_edges,)")
                heuristic = heuristic.reshape(distances.shape)
            scores.append(
                aco(
                    distances,
                    heuristic,
                    ants=payload["ants"],
                    iterations=payload["iterations"],
                    seed=payload["seed"] + index,
                )
            )
    print(json.dumps({"objective": float(np.mean(scores))}, allow_nan=False))


if __name__ == "__main__":
    worker()
