"""Compare CMSA constructions with paired graph/algorithm seeds and equal budgets."""

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import rankdata

from cmsa import VARIANTS, Config, cmsa, graph_neighbors, read_graph


def make_graph(family, n, density, seed):
    """Explicit synthetic benchmark mapping; these are not the authors' instances."""
    if n < 3 or not 0 < density <= 1 or not math.isfinite(density):
        raise ValueError("generated graphs require n >= 3 and density in (0, 1]")
    if family == "er":
        graph = nx.gnp_random_graph(n, density, seed=seed)
    elif family == "ba":
        graph = nx.barabasi_albert_graph(n, max(1, min(n - 1, round(density * n / 2))), seed=seed)
    elif family == "ws":
        k = max(2, min(n - 1, round(density * (n - 1))))
        k -= k % 2
        graph = nx.watts_strogatz_graph(n, k, 0.1, seed=seed)
    else:
        raise ValueError(f"unknown family: {family}")
    return graph_neighbors(graph)


def mean_ranks(scores):
    return np.mean(
        [rankdata(-np.asarray(row), method="average") for row in scores], axis=0
    ).tolist()


def load_constructor(path):
    """Explicitly load user-reviewed Python code; this is ordinary execution, not isolation."""
    spec = importlib.util.spec_from_file_location("cmsa_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError("constructor must be a Python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "generate_solution", None)):
        raise ValueError("candidate must define generate_solution")
    return module.generate_solution


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graph",
        type=Path,
        action="append",
        help="input file; repeat for multiple graphs",
    )
    p.add_argument("--families", nargs="+", choices=("er", "ba", "ws"), default=["er", "ba", "ws"])
    p.add_argument("--sizes", nargs="+", type=int, default=[60])
    p.add_argument("--densities", nargs="+", type=float, default=[0.1, 0.5])
    p.add_argument("--instances", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--time-limit", type=float, default=1.0, help="wall seconds per algorithm/graph")
    p.add_argument("--solver-time-limit", type=float, default=0.2)
    p.add_argument("--iterations", type=int, help="optional iteration cap")
    p.add_argument("--n-solutions", type=int, default=10)
    p.add_argument("--age-max", type=int, default=10)
    p.add_argument("--determinism-rate", type=float, default=0.8)
    p.add_argument("--candidate-list-size", type=int, default=5)
    p.add_argument(
        "--constructor",
        type=Path,
        help="execute a reviewed generate_solution alongside variants",
    )
    p.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    args = p.parse_args(argv)
    config = Config(
        time_limit=args.time_limit,
        solver_time_limit=args.solver_time_limit,
        iterations=args.iterations,
        n_solutions=args.n_solutions,
        age_max=args.age_max,
        determinism_rate=args.determinism_rate,
        candidate_list_size=args.candidate_list_size,
    )
    try:
        config.validate()
        if args.instances < 1:
            raise ValueError("instances must be positive")
        if len(set(args.variants)) != len(args.variants):
            raise ValueError("variants must be unique")
        if args.output.exists():
            raise ValueError("output directory already exists; choose a new --output")
        if args.graph:
            jobs = [("file", str(path), None, None, None) for path in args.graph]
        else:
            if any(n < 3 for n in args.sizes) or any(not 0 < d <= 1 for d in args.densities):
                raise ValueError("require sizes >= 3 and densities in (0, 1]")
            jobs = [
                (family, None, n, d, i)
                for family in args.families
                for n in args.sizes
                for d in args.densities
                for i in range(args.instances)
            ]
        constructor = load_constructor(args.constructor) if args.constructor else None
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    args.output.mkdir(parents=True)
    labels = [*args.variants, *(["candidate"] if constructor else [])]
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("slick-ai", "scipy", "numpy", "networkx")},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "graph_mapping": "ER p=d; BA m=round(d*n/2) clipped; WS even k~d*(n-1), beta=0.1",
        "budget_clock": "wall",
        "labels": labels,
    }
    metadata["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("cmsa.py", "experiment.py", "improve.py")
    }
    metadata["arguments"]["graph"] = [str(path) for path in args.graph] if args.graph else None
    if args.constructor:
        source = args.constructor.read_text()
        (args.output / "candidate.py").write_text(source)
        metadata["candidate_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    scores = []
    with (args.output / "results.jsonl").open("w") as output:
        for index, (family, path, n, density, replica) in enumerate(jobs):
            graph_seed = int.from_bytes(
                hashlib.sha256(
                    json.dumps([args.seed, family, path, n, density, replica]).encode()
                ).digest()[:4]
            )
            neighbors = read_graph(path) if path else make_graph(family, n, density, graph_seed)
            graph_id = f"{index:04d}-{family}"
            graph_path = args.output / f"{graph_id}.txt"
            with graph_path.open("w") as saved:
                saved.write(f"{len(neighbors)}\n")
                for u, adjacent in enumerate(neighbors):
                    for v in sorted(adjacent):
                        if u < v:
                            saved.write(f"{u} {v}\n")
            graph_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
            row = {}
            execution_order = labels.copy()
            random.Random(graph_seed).shuffle(execution_order)
            for label in execution_order:
                settings = replace(config, variant="cmsa" if label == "candidate" else label)
                kwargs = {"constructor": constructor} if label == "candidate" else {}
                result = cmsa(neighbors, settings, seed=graph_seed, **kwargs)
                result.update(
                    algorithm=label,
                    graph=graph_id,
                    graph_sha256=graph_hash,
                    family=family,
                    n=len(neighbors),
                    edges=sum(map(len, neighbors)) // 2,
                    density_parameter=density,
                    replica=replica,
                )
                output.write(json.dumps(result, allow_nan=False) + "\n")
                output.flush()
                row[label] = result["score"]
                print(
                    f"{graph_id} {label:9s} MIS={result['score']:4d} "
                    f"{result['seconds']:.3f}s iterations={result['iterations']}",
                    flush=True,
                )
            scores.append([row[label] for label in labels])
    summary = {
        "instances": len(scores),
        "mean_rank": dict(zip(labels, mean_ranks(scores))),
        "mean_score": dict(zip(labels, np.mean(scores, axis=0).tolist())),
        "note": "Descriptive paired ranks; no statistical significance claim.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
