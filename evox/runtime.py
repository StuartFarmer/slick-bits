"""Optional local strategy execution in a fresh process; not a security sandbox."""

import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


async def run_python_strategy(
    code: str,
    population: list[dict],
    state: dict,
    seed: int,
    *,
    timeout: float = 5.0,
) -> dict:
    """Execute the selection interface with a deadline and no inherited environment.

    For untrusted execution, inject a container/remote sandbox with this same
    callback signature. Local processes still have the user's filesystem access.
    The evaluator has a separate caller-owned execution boundary.
    """
    payload = json.dumps(dict(code=code, population=population, state=state, seed=seed)).encode()
    with TemporaryDirectory(prefix="evox-strategy-") as directory:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(Path(__file__).with_name("_worker.py").resolve()),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=directory,
            env={},
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(payload), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            if process.returncode is None:
                process.kill()
            await process.communicate()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise ValueError("strategy execution timed out") from exc
    if process.returncode:
        raise ValueError(f"strategy process exited with status {process.returncode}")
    try:
        result = json.loads(stdout)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("strategy process returned invalid JSON") from exc
    if result.get("error"):
        raise ValueError(result["error"])
    return result["selection"]
