"""Evaluate trusted candidate Python in a disposable process with a hard timeout.

This is failure containment, NOT a security sandbox. Code has the caller's OS rights.
"""

import ast
import asyncio
import json
import math
import os
import signal
import sys
import tempfile
from pathlib import Path

import numpy as np

import problems


def validate_code(code, problem):
    tree = ast.parse(code)
    compile(tree, "<heuristic>", "exec")
    name, parameters = problems.INTERFACES[problem]
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1:
        raise ValueError(f"define exactly one {name}{parameters}")
    fn = functions[0]
    if (
        [arg.arg for arg in fn.args.posonlyargs + fn.args.args] != list(parameters)
        or fn.decorator_list
        or any(default is None for default in fn.args.kw_defaults)
    ):
        raise ValueError(f"expected undecorated {name}{parameters}, no extra required arguments")


async def evaluate_candidate(code, dataset, *, timeout=60, iterations=1000, seconds=60, seed=0):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    try:
        validate_code(code, dataset["problem"])
    except (SyntaxError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    with tempfile.TemporaryDirectory(prefix="eoh-") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        request.write_text(
            json.dumps(
                {
                    "code": code,
                    "dataset": dataset,
                    "iterations": iterations,
                    "seconds": seconds,
                    "seed": seed,
                    "timeout": timeout,
                }
            )
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "SYSTEMROOT", "LANG", "LC_ALL", "TMPDIR"}
        }
        env.update(
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            PYTHONHASHSEED=str(seed % (2**32)),
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            str(request),
            str(response),
            cwd=directory,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except asyncio.TimeoutError:
            return {"error": f"evaluation exceeded {timeout:g}s"}
        finally:
            # Also reap descendants if the worker exited before them.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.returncode is None:
                process.kill()
            await process.wait()
        if process.returncode != 0 or not response.exists():
            return {"error": f"worker exited with status {process.returncode}"}
        try:
            result = json.loads(response.read_text())
            if "error" not in result and not math.isfinite(result["fitness"]):
                raise ValueError("nonfinite fitness")
            return result
        except (ValueError, KeyError, TypeError) as exc:
            return {"error": f"invalid worker response: {exc}"}


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2])
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
        cpu = max(1, math.ceil(request["timeout"]))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    try:
        problem = request["dataset"]["problem"]
        validate_code(request["code"], problem)
        namespace = {"np": np, "__name__": "eoh_candidate"}
        np.random.seed(request["seed"] % (2**32))
        exec(compile(request["code"], "<heuristic>", "exec"), namespace)
        result = problems.evaluate(
            request["dataset"],
            namespace[problems.INTERFACES[problem][0]],
            iterations=request["iterations"],
            seconds=request["seconds"],
            seed=request["seed"],
        )
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
    output.write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
